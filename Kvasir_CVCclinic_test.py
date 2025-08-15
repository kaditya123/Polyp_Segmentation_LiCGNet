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

#user
from model import CGNet
from utils.metric import get_iou
from utils.modeltools import netParams
from utils.loss import CrossEntropyLoss2d
from utils.metric import calculate_metrics
from utils.convert_state import convert_state_dict
from dataset.Kvasir_CVCclinic import Kvasir_CVCclinic_train_inform 
from dataset.Kvasir_CVCclinic import KvasirSEG_ClinicDB_Test_dataset


def visualize_prediction(image, ground_truth_mask, predicted_mask, image_name):
    """
    Function to visualize the image, ground truth mask, and predicted mask.
    
    Args:
    - image: Input image
    - ground_truth_mask: Ground truth segmentation mask
    - predicted_mask: Predicted segmentation mask
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display the image
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    # Display the ground truth mask
    axes[1].imshow(ground_truth_mask, cmap='gray', vmin=0, vmax=1)  # Assuming 21 classes, adjust colormap accordingly
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Display the predicted mask
    axes[2].imshow(predicted_mask, cmap='gray', vmin=0, vmax=1)  # Assuming 21 classes, adjust colormap accordingly
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


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
    input_sample, _, _ = next(iter(test_loader))
    input_sample = Variable(input_sample).cuda()  # Change to `.cuda()` if using GPU
    macs, params = profile(model, inputs=(input_sample,))
    flops = macs * 2  # MACs are half of FLOPs
    print(f"FLOPs: {flops:.2e}, Params: {params}")
    
    for i, (input, label, size, name) in enumerate(test_loader):
        input_var = Variable(input, volatile=True).cuda()   ### change to cuda in GPU usage
        start_time = time.time()  # Start measuring time for latency
        
        output = model(input_var)
        torch.cuda.synchronize() 
        
        end_time = time.time()   # End measuring time for latency
        latency = end_time - start_time   # Calculate latency and accumulate total inference time
        total_inference_time += latency
        num_images += input.size(0)  # Batch size
        
        output= output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype = np.uint8)
        output= output.transpose(1,2,0)
        output= np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append( [gt.flatten(), output.flatten()])
        
        all_preds.extend(output.flatten())
        all_labels.extend(gt.flatten())
        
         # Visualize the prediction
        if i < 11:
            image = input[0].numpy().transpose(1, 2, 0)  # Assuming input is in the format (C, H, W)
            print(f"Visualizing: Image: {name[0]}, Mask: {name[0]}")
            visualize_prediction(image, gt, output, name[0])
        
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
    testLoader = data.DataLoader(KvasirSEG_ClinicDB_Test_dataset(args.data_dir, args.test_data_list,  mean= datas['mean']),
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