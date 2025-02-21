# test.py

# This file is used for evaluating the trained model on test data.
# This file should include:

# 1. Model loading
#    - Load the trained model from a checkpoint

# 2. Evaluation loop
#    - Iterate through test data
#    - Generate predictions using the loaded model
#    - Compute evaluation metrics


# A Example Usage:
# python test.py --model_path /path/to/model/checkpoint --config path/to/config.yaml
import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from easydict import EasyDict as edict
import numpy as np
from dataset import Human

def calcIOU_per_class(pred, mask, cls):
    """
    Calculate IoU for a specific class
    """
    pred_cls = (pred == cls)  # Binary mask for predicted pixels of class `cls`
    mask_cls = (mask == cls)  # Binary mask for ground truth pixels of class `cls`
    
    intersection = np.logical_and(pred_cls, mask_cls).sum()
    union = np.logical_or(pred_cls, mask_cls).sum()
    
    if union == 0:
        return float('nan')  # Avoid division by zero if there is no ground truth for this class
    else:
        return intersection / union


def test(dataLoader, netmodel, exp_args, n_classes = 8):
    # switch to eval mode
    netmodel.eval()
    softmax = nn.Softmax(dim=1)
    iou_per_class = np.zeros(n_classes)
    total_per_class = np.zeros(n_classes)

    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):  
        input_ori_var = Variable(input_ori.cuda())
        input_var = Variable(input.cuda())
        edge_var = Variable(edge.cuda())
        mask_var = Variable(mask.cuda())
        
        # compute output: loss part1
        if exp_args.addEdge == True:
            output_mask, output_edge = netmodel(input_ori_var)
        else:
            output_mask = netmodel(input_ori_var)
            
        # Get the predicted class by taking the argmax across the class dimension
        prob = softmax(output_mask)
        pred = torch.argmax(prob, dim=1)  # Shape: (batch_size, H, W)
        pred = pred[0].data.cpu().numpy() 

        mask_np = mask_var[0].data.cpu().numpy()  # Ground truth mask for the first batch

        # Calculate IoU for each class
        for cls in range(n_classes):
            iou_per_class[cls] += calcIOU_per_class(pred, mask_np, cls)
            total_per_class[cls] += 1

    # Average IoU per class
    mean_iou_per_class = iou_per_class / total_per_class
    mean_iou = np.nanmean(mean_iou_per_class)  # Mean IoU over all classes
         
    print(f"Mean IoU: {mean_iou}")
    return mean_iou


def main(args):
    cudnn.benchmark = True
    assert args.model in ['PortraitNet'], 'Error!, <model> should in [PortraitNet]'
    
    config_path = args.config_path
    print ('===========> loading config <============')
    print ("config path: ", config_path)
    with open(config_path,'rb') as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)
    
    print ('===========> loading data <===========')
    exp_args = edict()
    
    exp_args.task = cf['task'] # only support 'seg' now
    exp_args.datasetlist = cf['datasetlist']
    exp_args.model_root = cf['model_root'] 
    exp_args.data_root = cf['data_root']
    exp_args.file_root = cf['file_root']
    # the height of input images, default=224
    exp_args.input_height = cf['input_height']
    # the width of input images, default=224
    exp_args.input_width = cf['input_width']
    
    # if exp_args.video=True, add prior channel for input images, default=False
    exp_args.video = cf['video']
    # the probability to set empty prior channel, default=0.5
    exp_args.prior_prob = cf['prior_prob']
    
    # whether to add boundary auxiliary loss, default=False
    exp_args.addEdge = cf['addEdge']
    # the weight of boundary auxiliary loss, default=0.1
    exp_args.edgeRatio = cf['edgeRatio']
    # whether to add consistency constraint loss, default=False
    exp_args.stability = cf['stability']
    # whether to use KL loss in consistency constraint loss, default=True
    exp_args.use_kl = cf['use_kl']
    # temperature in consistency constraint loss, default=1
    exp_args.temperature = cf['temperature'] 
    # the weight of consistency constraint loss, default=2
    exp_args.alpha = cf['alpha'] 
    
    # input normalization parameters
    exp_args.padding_color = cf['padding_color']
    exp_args.img_scale = cf['img_scale']
    # BGR order, image mean, default=[103.94, 116.78, 123.68]
    exp_args.img_mean = cf['img_mean']
    # BGR order, image val, default=[1/0.017, 1/0.017, 1/0.017]
    exp_args.img_val = cf['img_val'] 
    
    # whether to use pretian model to init portraitnet
    exp_args.istrain = False
    exp_args.init = False
    exp_args.resume = True
    
    # if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
    exp_args.useUpsample = cf['useUpsample'] 
    # if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
    exp_args.useDeconvGroup = cf['useDeconvGroup'] 

    dataset_test = Human(exp_args)
    print(len(dataset_test))
    dataLoader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)
    print(len(dataLoader_test))
    print("finish load dataset ...")

    if args.model == 'PortraitNet':
        from potraitnet import PotraitNet
        netmodel = PotraitNet(n_class=8, 
                                        useUpsample=exp_args.useUpsample, 
                                        useDeconvGroup=exp_args.useDeconvGroup, 
                                        addEdge=exp_args.addEdge, 
                                        channelRatio=1.0, 
                                        minChannel=16, 
                                        weightInit=True,
                                        video=exp_args.video).cuda()
        print ("finish load PortraitNet ...")

    if exp_args.resume:
        bestModelFile = os.path.join(exp_args.model_root, 'model_best_easypotrait.pth.tar')
        if os.path.isfile(bestModelFile):
            checkpoint = torch.load(bestModelFile, weights_only=False)
            netmodel.load_state_dict(checkpoint['state_dict'])
            print ("minLoss: ", checkpoint['minLoss'], checkpoint['epoch'])
            print("=> loaded checkpoint '{}' (epoch {})".format(bestModelFile, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(bestModelFile))
    netmodel = netmodel.cuda()

    acc = test(dataLoader_test, netmodel, exp_args)
    print ("mean iou: ", acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing code')
    parser.add_argument('--model', default='PortraitNet', type=str, 
                        help='<model> should be PortraitNet')
    parser.add_argument('--config_path', 
                        default='./config/config_test_easypotrait.yaml', 
                        type=str, help='the config path of the model')

    args = parser.parse_args()
    
    main(args)
    

