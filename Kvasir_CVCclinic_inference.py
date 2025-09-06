import os
import sys
import cv2
import time
import torch
import gc
import pickle
import timeit
import random
import numpy as np
import torch.nn as nn
import os.path as osp
from thop import profile
from torch.utils import data
from argparse import ArgumentParser
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from model import CGNet
from utils.modeltools import netParams
from utils.loss import CrossEntropyLoss2d
from utils.metric import ConfusionMatrix
from utils.metric import calculate_metrics
from vis_prediction import visualize_prediction
from utils.convert_state import convert_state_dict
from dataset.Kvasir_CVCclinic import Kvasir_CVCclinic_train_inform 
from dataset.Kvasir_CVCclinic import KvasirSEG_ClinicDB_Val_dataset

# module-level device (default CPU)
DEVICE = torch.device('cpu')

def test(args, test_loader, model, criterion):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
      criterion: loss function
    return: class IoU and mean IoU
    """
    #evaluation or test mode
    model.eval()
    total_batches = len(test_loader)

    # use incremental confusion matrix to avoid large memory usage
    ConfM = ConfusionMatrix(args.classes)

    # Initialize variables to measure latency and throughput
    total_inference_time = 0
    num_images = 0

    # FLOPS Calculation - skip on CPU unless explicitly requested
    if getattr(args, 'profile', False) or DEVICE.type == 'cuda':
        try:
            input_sample, _, _, _ = next(iter(test_loader))
            input_sample = input_sample.to(DEVICE)
            macs, params = profile(model, inputs=(input_sample,))
            flops = macs * 2  # MACs are half of FLOPs
            print(f"FLOPs: {flops:.2e}, Params: {params}")
        except Exception as e:
            print("Profiling skipped (error):", e)
    else:
        print("Profiling skipped on CPU (use --profile True to force)")
    
    for i, (input, label, size, name) in enumerate(test_loader):
        input_var = input.to(DEVICE)    # move to device
        
        start_time = time.time()
        output = model(input_var)       # (B, C, H, W)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        # ---- Process prediction & free output to reduce peak memory usage ---- #
        pred = output[0].cpu().detach().numpy()                 # (C, H, W)
        pred = np.argmax(pred, axis=0).astype(np.uint8)         # (H, W)
        del output
        torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        gc.collect()

        # ---- Process ground truth ---- #
        gt = label[0].cpu().numpy().astype(np.uint8)            # (H, W)

        latency = end_time - start_time
        total_inference_time += latency
        num_images += input.size(0)

        # For metrics: build per-sample confusion matrix and add to global ConfM
        gt_flat = gt.flatten().astype(np.int64)
        pred_flat = pred.flatten().astype(np.int64)

        # mask out invalid labels
        valid_mask = (gt_flat >= 0) & (gt_flat < args.classes) & (pred_flat >= 0) & (pred_flat < args.classes)
        if valid_mask.any():
            idx = gt_flat[valid_mask] * args.classes + pred_flat[valid_mask]
            counts = np.bincount(idx, minlength=args.classes * args.classes)
            m = counts.reshape((args.classes, args.classes)).astype(np.float64)
            ConfM.addM(m)

        # ---- Visualization ----
        if (i < 4) and args.no_visual is False:
            image = input[0].cpu().numpy().transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
            print(f"Visualizing: Image: {name[0]}, Mask: {name[0]}")
            visualize_prediction(image, gt, pred, name[0])
        
        print(f"Processed image {i}")

        
    # compute meanIoU from confusion matrix
    meanIoU, per_class_iu, M = ConfM.jaccard()
    # compute precision/recall/f1/accuracy from confusion matrix
    precision, recall, f1_score, accuracy = calculate_metrics(ConfM.M)

    # Calculate FPS and Throughput
    fps = num_images / total_inference_time
    throughput = fps  # For single-threaded inference

    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Throughput: {throughput:.2f} images/second")

    return meanIoU, per_class_iu, precision, recall, f1_score, accuracy


def test_model(args):
    """
    main function for testing 
    args:
       args: global arguments
    """
    print("=====> Check if the cached file exists ")
    if not os.path.isfile(args.inform_data_file):
        print("%s is not found" %(args.inform_data_file))
        dataCollect = Kvasir_CVCclinic_train_inform(args.data_dir, args.classes, train_set_file= args.dataset_list, 
                                        inform_data_file = args.inform_data_file) #collect mean std, weigth_class information
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        print("%s exists" %(args.inform_data_file))
        datas = pickle.load(open(args.inform_data_file, "rb"))
    
    print(args)
    global network_type

    # choose device
    global DEVICE
    DEVICE = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
    if DEVICE.type == 'cuda':
        print("=====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Limit CPU threads to avoid oversubscription on laptops
    if DEVICE.type == 'cpu':
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)

    # set CUDA seed only if GPU is selected
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    cudnn.enabled = (DEVICE.type == 'cuda')

    print("=====> DEVICE status")
    print(f"Running on device: {DEVICE}")

    M = args.M
    N = args.N

    model = CGNet.CGNET(classes= args.classes, M= M, N= N)

    network_type="CGNet"
    print("=====> Current Architecture:  CGNet_M%sN%s"%(M, N))
    total_paramters = netParams(model)
    print("The number of parameters: " + str(total_paramters))
    print("data['classWeights']: ", datas['classWeights'])
    weight = torch.from_numpy(datas['classWeights'])
    print("=====> Dataset statistics")
    print("Mean & Std: ", datas['mean'], datas['std'])
    
    # define optimization criteria
    criteria = CrossEntropyLoss2d(weight, args.ignore_label)

    # move model and criteria to chosen device
    model = model.to(DEVICE)
    criteria = criteria.to(DEVICE)
    
    # load test set
    train_transform= transforms.Compose([
        transforms.ToTensor()])
    loader_workers = args.num_workers if DEVICE.type == 'cuda' else 0
    pin_mem = True if DEVICE.type == 'cuda' else False
    testLoader = data.DataLoader(KvasirSEG_ClinicDB_Val_dataset(args.data_dir, args.test_data_list,  mean= datas['mean']),
                                  batch_size = args.batch_size, shuffle=True, num_workers=loader_workers, pin_memory=pin_mem, drop_last=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=====> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=DEVICE)
            #model.load_state_dict(convert_state_dict(checkpoint['model']))
            model.load_state_dict(checkpoint['model'])
        else:
            print("=====> No checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark= True

    print("=====> Beginning test")
    print("Length of testing set:", len(testLoader))
    mIOU_val, per_class_iu, precision, recall, f1_score, accuracy = test(args, testLoader, model, criteria)
    
    print("mIoU : ", mIOU_val)
    print("Per class IoU : ", per_class_iu)
    print("Precision : ", precision)
    print("Recall : ", recall)
    print("F1_score : ", f1_score)
    print("Accuracy : ", accuracy)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type = str, default = "CGNet", help = "model name: Context Guided Network")
    parser.add_argument('--dataset', type = str, default = "KvasirSeg_CVCClinic", help = "kvasir + cvc-clinic")
    parser.add_argument('--ignore_label', type = int, default = -1, help = "nClass")
    parser.add_argument('--data_dir', default = "dataset/Polyp_processed_1_0_data/TESTING_Image+Mask", help = "data directory")
    parser.add_argument('--test_data_list', default = "dataset/Polyp_processed_1_0_data/TESTING_Image+Mask/Polyp_Kvasir_SEG_test_list.txt", help= "data directory")  ## change the testing dataset as needed to verify 
    parser.add_argument('--scaleIn', type = int, default = 1, help = "for input image, default is 1, keep fixed size")  
    parser.add_argument('--num_workers', type = int, default = 1, help = "the number of parallel threads") 
    parser.add_argument('--batch_size', type = int, default = 1, help = "the batch size is set to 1 when testing")
    parser.add_argument('--resume', type = str, default = './checkpoint/KvasirSeg_CVCClinic/CGNet_M3N21bs8gpu1_ontrain/model_250.pth', 
                         help = "use this file to load last checkpoint for testing")
    parser.add_argument('--classes', type = int, default = 2, 
                         help = "the number of classes in the dataset. 2 classes for all the datasets respectively")
    parser.add_argument('--inform_data_file', default = "dataset/Polyp_processed_1_0_data/polyp_inform.pkl", 
                         help = "storing classes weights, mean and std")
    parser.add_argument('--M', type = int, default = 3,  help = "the number of block in stage 2")
    parser.add_argument('--N', type = int, default = 21, help = "the number of block in stage 3")
    parser.add_argument('--cuda', type = bool, default = False, help = "running on CPU or GPU")
    parser.add_argument('--profile', type = bool, default = False, help = 'run FLOPS profile (Only on GPU or if explicitly enabled)')
    parser.add_argument('--no_visual', type = bool, default = False, help = 'disable matplotlib visualization during test')
    parser.add_argument("--gpus", type = str, default = "0",  help = "gpu ids (default: 0)")
    parser.add_argument("--gpu_nums",  type = int, default=1 , help="the number of gpu")
    
    args, _ = parser.parse_known_args(sys.argv[1:])
    test_model(args)