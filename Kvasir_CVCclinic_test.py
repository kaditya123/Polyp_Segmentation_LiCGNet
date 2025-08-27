import os
import sys
import time
import torch
import pickle
import timeit
import random
import numpy as np
import torch.nn as nn
import os.path as osp
from thop import profile
from torch.utils import data
import matplotlib.pyplot as plt
from torch.autograd import Variable
from argparse import ArgumentParser
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import cv2

#user
from model import CGNet
from utils.metric import get_iou
from utils.modeltools import netParams
from utils.loss import CrossEntropyLoss2d
from utils.metric import calculate_metrics
from utils.convert_state import convert_state_dict
from dataset.Kvasir_CVCclinic import Kvasir_CVCclinic_train_inform 
from dataset.Kvasir_CVCclinic import KvasirSEG_ClinicDB_Val_dataset


def visualize_prediction(image, ground_truth_mask, predicted_mask, image_name):
    """
    Function to visualize the image, ground truth mask, and predicted mask.
    
    Args:
    - image: Input image
    - ground_truth_mask: Ground truth segmentation mask
    - predicted_mask: Predicted segmentation mask
    """
    # Minimal, robust display
    image = np.array(image)
    gt = np.array(ground_truth_mask)
    pred = np.array(predicted_mask)

    # Normalize image to uint8 [0,255]
    if image.dtype != np.uint8:
        im = image.astype(np.float32)
        im = (im - im.min()) / max(1e-6, (im.max() - im.min()))
        image_disp = (im * 255).astype(np.uint8)
    else:
        image_disp = image

    # If channels-first, convert to HWC
    if image_disp.ndim == 3 and image_disp.shape[0] in (1, 3) and image_disp.shape[0] != image_disp.shape[2]:
        image_disp = np.transpose(image_disp, (1, 2, 0))

    if image_disp.ndim == 2:
        image_disp = cv2.cvtColor(image_disp, cv2.COLOR_GRAY2RGB)

    # Prepare binary masks (0 or 255)
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    gt_mask = (gt > (0 if gt.max() <= 1 else (gt.max() / 2))).astype(np.uint8) * 255
    pred_mask = (pred > (0 if pred.max() <= 1 else (pred.max() / 2))).astype(np.uint8) * 255

    H, W = image_disp.shape[:2]
    if gt_mask.shape != (H, W):
        gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    if pred_mask.shape != (H, W):
        pred_mask = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Overlay prediction in red
    overlay = image_disp.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    idx = pred_mask > 127
    overlay[idx] = (overlay[idx].astype(np.float32) * 0.5 + red.astype(np.float32) * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image_disp); axes[0].set_title('Image'); axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title('GT'); axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='gray'); axes[2].set_title('Pred'); axes[2].axis('off')
    axes[3].imshow(overlay); axes[3].set_title('Overlay'); axes[3].axis('off')
    plt.tight_layout(); plt.show()


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
    all_preds = []
    all_labels = []
    data_list=[]

     # Initialize variables to measure latency and throughput
    total_inference_time = 0
    num_images = 0

     # FLOPS Calculation (using the first batch)
    input_sample, _, _, _ = next(iter(test_loader))
    input_sample = Variable(input_sample).cuda()  # Change to `.cuda()` if using GPU
    macs, params = profile(model, inputs=(input_sample,))
    flops = macs * 2  # MACs are half of FLOPs
    print(f"FLOPs: {flops:.2e}, Params: {params}")
    
    for i, (input, label, size, name) in enumerate(test_loader):
        input_var = input.cuda()   # Variable wrapper not needed in PyTorch >0.4
        start_time = time.time()
        
        output = model(input_var)  # (B, C, H, W)
        torch.cuda.synchronize()
        
        end_time = time.time()
        latency = end_time - start_time
        total_inference_time += latency
        num_images += input.size(0)

        # ---- Process ground truth ----
        gt = label[0].cpu().numpy().astype(np.uint8)   # (H, W)

        # ---- Process prediction ----
        pred = output[0].cpu().detach().numpy()        # (C, H, W)
        pred = np.argmax(pred, axis=0).astype(np.uint8)  # (H, W)

        # For metrics
        data_list.append([gt.flatten(), pred.flatten()])
        all_preds.extend(pred.flatten())
        all_labels.extend(gt.flatten())

        # ---- Visualization ----
        if i < 11:
            image = input[0].cpu().numpy().transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
            print(f"Visualizing: Image: {name[0]}, Mask: {name[0]}")
            visualize_prediction(image, gt, pred, name[0])

        
    meanIoU, per_class_iu= get_iou(data_list, args.classes)
    precision, recall, f1_score, accuracy = calculate_metrics(all_labels, all_preds)
################################################
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
     
    if args.cuda:
        print("=====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed) 
    cudnn.enabled = True

    M = args.M
    N = args.N

    model = CGNet.CGNET(classes= args.classes, M= M, N= N)

    network_type="CGNet"
    print("=====> current architeture:  CGNet_M%sN%s"%(M, N))
    total_paramters = netParams(model)
    print("the number of parameters: " + str(total_paramters))
    print("data['classWeights']: ", datas['classWeights'])
    weight = torch.from_numpy(datas['classWeights'])
    print("=====> Dataset statistics")
    print("mean and std: ", datas['mean'], datas['std'])
    
    # define optimization criteria
    criteria = CrossEntropyLoss2d(weight, args.ignore_label)
    if args.cuda:
        model = model.cuda()
        criteria = criteria.cuda()
    
    #load test set
    train_transform= transforms.Compose([
        transforms.ToTensor()])
    testLoader = data.DataLoader(KvasirSEG_ClinicDB_Val_dataset(args.data_dir, args.test_data_list,  mean= datas['mean']),
                                  batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=====> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume) ## map_location=torch.device('cpu'))  #### comment the map_location during the GPU usage ###
            #model.load_state_dict(convert_state_dict(checkpoint['model']))
            model.load_state_dict(checkpoint['model'])
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark= True

    print("=====> beginning test")
    print("length of test set:", len(testLoader))
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
    parser.add_argument('--cuda', type = bool, default = True, help = "running on CPU or GPU")
    parser.add_argument("--gpus", type = str, default = "0",  help = "gpu ids (default: 0)")
    parser.add_argument("--gpu_nums",  type = int, default=1 , help="the number of gpu")
    
    args, _ = parser.parse_known_args(sys.argv[1:])
    test_model(args)