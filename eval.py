from __future__ import division

import os
import random
import argparse
import time
from time import gmtime, strftime
import pytz
import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from data.coco import COCODataset
from data.voc0712 import VOCDetection
from data.stanford_dog import StanfordDogDataset
from data.transform import Augmentation, BaseTransform

from utils.misc import detection_collate
from utils.com_paras_flops import FLOPs_and_Params
from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator
from evaluator.StanfordDogapi_evaluator import StanfordDogAPIEvaluator
from utils.util import get_current_time

from models.build import build_yolo
from models.matcher import gt_creator


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # 基本参数
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')

    # 模型参数
    parser.add_argument('-v', '--version', default='yolo',
                        help='yolo')

    
    
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
   

    # 数据集参数
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('--root', default=None,
                        help='data root')

    parser.add_argument('--arch', default='resnet18',
                        help='backbone arch')
    parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
    parser.add_argument('--stride', default=32, type=int,
                    help='网络的最大步长')

    parser.add_argument('--input_size', default=0, type=int,
                    help='input image size')

    parser.add_argument('--inference', action='store_true', default=False,
                    help='inference model')

    return parser.parse_args()



def eval():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    current_time = get_current_time()
    log_dir = args.resume[:args.resume.find('weight')]
    
    
    if not args.gpu is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 是否使用cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.input_size:
        train_size = args.input_size
        val_size = args.input_size
    else:
        # 是否使用多尺度训练
        if args.multi_scale:
            print('use the multi-scale trick ...')
            train_size = 640
            val_size = 416
        else:
            train_size = 416
            val_size = 416

    # 构建dataset类和dataloader类
    num_classes, evaluator_val, evaluator_test = build_eval_dataset(args, device, val_size)
    

    # 构建我们的模型
    model = build_yolo(args, device, train_size, num_classes, trainable=False, pretrain = False)
    model.to(device)

    # 计算模型的FLOPs和参数量
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    FLOPs, Params = FLOPs_and_Params(model=model_copy, 
                        img_size=val_size, 
                        device=device)
    del model_copy

    
    # keep training
    if args.resume is not None:
        print('resume model weight from: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    if args.inference:
        save_dir = os.path.join(log_dir, 'infer')
        os.makedirs(save_dir, exist_ok=True)
        # avg_time_consumption = evaluator_val.inference(model, save_dir)
        avg_time_consumption = evaluator_test.inference(model, save_dir)
        logfile = os.path.join(log_dir, 'parameters.txt')
        # if os.path.exists(logfile):
        #     os.remove(logfile)
        log_file = open(logfile, "a+")
        log_file.write('\n')
        log_file.write('---------------Inference---------' + '\n')
        p=vars(args)
        
        log_file.write('current_time' + ':' + current_time + '\n')
        
        log_file.write(avg_time_consumption + '\n')
        log_file.write('---------------Inference END---------' + '\n')
        log_file.write('\n')
        log_file.close()
    else:
        os.makedirs(log_dir, exist_ok=True)
        logfile = os.path.join(log_dir, 'parameters.txt')
        # if os.path.exists(logfile):
        #     os.remove(logfile)
        log_file = open(logfile, "a+")
        # log_file.write('\n')
        log_file.write('\n')
        log_file.write('---------------Evaluation---------' + '\n')
        p=vars(args)
        
        log_file.write('current_time' + ':' + current_time + '\n')
        
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')
        log_file.write(FLOPs + '\n')
        log_file.write(Params + '\n')
        # 


        model.trainable = False
        model.set_grid(val_size)
        model.eval()

        # evaluate
        val_ap50_95, val_ap50 = evaluator_val.evaluate(model)
        test_ap50_95, test_ap50 = evaluator_test.evaluate(model)

        
        val_print = '[val  ][AP @[ IoU=0.50:0.95 | maxDets=100 ] = %.2f || AP @[ IoU=0.50 | maxDets=100 ] = %.2f'\
                    % (val_ap50_95, val_ap50)                       
        log_file.write(val_print + '\n')

        test_print = '[test ][AP @[ IoU=0.50:0.95 | maxDets=100 ] = %.2f || AP @[ IoU=0.50 | maxDets=100 ] = %.2f'\
                    % (test_ap50_95, test_ap50)          
        
        log_file.write(test_print + '\n')
        log_file.write('---------------Evaluation END-------------' + '\n')             
        log_file.write('\n')
        log_file.close()
    
    


def build_eval_dataset(args, device, val_size):
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    val_transform = BaseTransform(val_size, pixel_mean, pixel_std)
    
    # 构建dataset类和dataloader类
    if args.dataset == 'voc':
        data_root = os.path.join(args.root, 'VOCdevkit')
        # 加载voc数据集
        num_classes = 20

        evaluator_val = VOCAPIEvaluator(
            data_root=data_root,
            img_size=val_size,
            device=device,
            transform=val_transform
            )
        
        evaluator_test = None

    elif args.dataset == 'coco':
        # 加载COCO数据集
        data_root = os.path.join(args.root, 'COCO')
        num_classes = 80

        evaluator_val = COCOAPIEvaluator(
            data_dir=data_root,
            img_size=val_size,
            device=device,
            transform=val_transform
            )
        evaluator_test = None

    elif args.dataset == 'StanfordDog':
        # 加载StanfordDog数据集
        data_root = os.path.join(args.root, 'stanfordDogsDataset')
        num_classes = 120
        
        evaluator_val = StanfordDogAPIEvaluator(
            data_dir=data_root,
            img_size=val_size,
            device=device,
            transform=val_transform,
            image_set='val'
            )
            
        evaluator_test = StanfordDogAPIEvaluator(
            data_dir=data_root,
            img_size=val_size,
            device=device,
            transform=val_transform,
            image_set='test'
            )
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('Eval model on:', args.dataset)
    # print('The dataset size:', len(evaluator_val), len(evaluator_test))
    print("----------------------------------------------------------")


    return num_classes, evaluator_val, evaluator_test



if __name__ == '__main__':
    eval()
