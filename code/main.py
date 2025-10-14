from torchvision import models
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from arguments import args
import numpy as np
import random
from model import ILViT
from model_gate import ILViT_Gate
from model_gate_sgu import ILViT_Gate_SGU
from model_gate_GFM import ILViT_GFM
from model_gate_SGU_GFM import ILViT_SGU_GFM
from torch.utils.tensorboard import SummaryWriter
import logging
from train import train_model

from thop import profile

if __name__ == '__main__':
    def set_random_seed(seed_value):
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_random_seed(42)
    start_time = time.time()
    # 训练前输出模型的重要设置参数
    print('gpu:',args.gpu)
    print('questionClass:',args.dataset)
    print('model:',args.model)
    print('epoch:',args.epochs)
    print('lr:',args.lr)
    batch_size = args.batchsize # 约占 10G 显存
    device = torch.device('cuda') # 默认使用 GPU
    NUM_EPOCH = args.epochs # 默认迭代次数
    lr = args.lr
    DATA = args.dataset
    net = args.model
    output = args.output
    pair = args.pair

    
    logging.basicConfig(filename=f"/home/mch/our_model/Gate_LIVIT/1_log/{net}-{DATA}-lr{lr}-gamma{args.gamma}-{pair}-{output}.log", level=logging.INFO)
    logger  = logging.getLogger(__name__)
    writer = SummaryWriter(f"/home/mch/our_model/Gate_LIVIT/3_tensorboard_writer/{net}-{DATA}-lr{lr}-gamma{args.gamma}-{pair}-{output}")
    
    # # 设置模型在哪个gpu上运行
    # if args.gpu: torch.cuda.set_device(args.gpu)
    # device = f'cuda:{args.gpu}'
    

    if args.model =='doubule_ILViT_gate_GLU':
        model_ft = ILViT_Gate(
            num_classes = 2,
            dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
            dim = 96,                         # dimension of first layer, doubles every layer
            dim_head = 16,                    # dimension of attention heads, kept at 32 in paper    
            depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
            window_size = 7,                  # window size for block and grids
            mbconv_expansion_rate = 4,        # expansion rate of MBConv
            mbconv_shrinkage_rate = 0.25, 
            stride =1  ,
            dropout = 0.1   ,                  # dropout
            sr_ratio = 1   #注意力机制中没有avgpool
        )
        print("模型为：doubule_ILViT_gate_GLU")
        
    elif args.model == 'double_ILViT_GFM2':
         model_ft = ILViT_GFM(
            num_classes = 2,
            dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
            dim = 96,                         # dimension of first layer, doubles every layer
            dim_head = 16,                    # dimension of attention heads, kept at 32 in paper    
            depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
            window_size = 7,                  # window size for block and grids
            mbconv_expansion_rate = 4,        # expansion rate of MBConv
            mbconv_shrinkage_rate = 0.25, 
            stride =1  ,
            dropout = 0.1   ,                  # dropout
            sr_ratio = 1   #注意力机制中没有avgpool
        )
         print("模型为：'double_ILViT_GFM2'")
         

    else:
        model_ft = ILViT(
            num_classes = 2,
            dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
            dim = 96,                         # dimension of first layer, doubles every layer
            dim_head = 16,                    # dimension of attention heads, kept at 32 in paper    
            depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
            window_size = 7,                  # window size for block and grids
            mbconv_expansion_rate = 4,        # expansion rate of MBConv
            mbconv_shrinkage_rate = 0.25, 
            stride =1  ,
            dropout = 0.1   ,                  # dropout
            sr_ratio = 1   #注意力机制中没有avgpool
        )
        print("模型为：double_ILViT—cat或者add")
    model_ft =  nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()# 定义损失函数
    
    # print(model_ft)
        
    model = train_model(model=model_ft,# 开始训练模型
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.epochs,device=device,logger=logger,writer=writer)    
    
    # speed = test_inference_speed(model_ft,device,logger)
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model_ft, inputs=(input, ))
    logger.info('flops:{}  params:{}'.format(flops,params))
