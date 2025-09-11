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
from torch.utils.tensorboard import SummaryWriter
import logging
from train import train_model
from unit import test_model,get_params_groups,create_lr_scheduler,test_inference_speed
# from datasets import dataloders
from model_series import ILViT_series
from model_cbam import ILViT_cbam
from model_GRN import ILViT_GRN
from model_contrast import ILViT_contrast,ILViT_with_usual_net
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
    device = torch.device(f'cuda:{args.gpu}') # 默认使用 GPU
    NUM_EPOCH = args.epochs # 默认迭代次数
    lr = args.lr
    DATA = args.dataset
    net = args.model

    
    logging.basicConfig(filename=f"/home/mch/our_model/LIVIT_3/1_log/{args.optim}-{net}-{DATA}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-wd{args.weight_decay}-epoch{NUM_EPOCH}.log", level=logging.INFO)
    logger  = logging.getLogger(__name__)
    writer = SummaryWriter(f"/home/mch/our_model/LIVIT_3/3_tensorboard_writer/{args.optim}{net}-{DATA}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-wd{args.weight_decay}-epoch{NUM_EPOCH}")

    # 设置模型在哪个gpu上运行
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = f'cuda:{args.gpu}'
    V0 = ILViT(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 3
    )
    #V1是注意力机制没有avgpool存在对比
    V1 = ILViT(
        num_classes = args.num_classes,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        # depth = (2, 2, 5, 2),  
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1   #注意力机制中没有avgpool
    )
        #DIM维度和maxvit一样
    V2 = ILViT(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96*2,                         # dimension of first layer, doubles every layer
        dim_head = 16*2,                    # dimension of attention heads, kept at 32 in paper
        depth = (1, 2, 3, 1),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        # sr_ratio = 3
        sr_ratio = 1
    )
    V3 = ILViT_series(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        # depth = (2, 2, 5, 2),  
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1   #注意力机制中没有avgpool
    )
    V4 = ILViT_series(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 48,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        # depth = (2, 2, 5, 2),  
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1   #注意力机制中没有avgpool
    )
    V5 =  ILViT_cbam(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        # depth = (2, 2, 5, 2),  
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1   #注意力机制中没有avgpool
    )
    V6 =  ILViT_GRN(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        # depth = (2, 2, 5, 2),  
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1   #注意力机制中没有avgpool
    )
    #g3-b3
    V7 = ILViT_contrast(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1 ,  #注意力机制中没有avgpool
        kernel_size_b =3,
        padding_b =1,
        kernel_size_g =3,
        padding_g =1,
    )
    #g5-b5
    V8 = ILViT_contrast(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1 ,  #注意力机制中没有avgpool
        kernel_size_b =5,
        padding_b =2,
        kernel_size_g =5,
        padding_g =2,
    )
    #B5-G3
    V9 = ILViT_contrast(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1 ,  #注意力机制中没有avgpool
        kernel_size_b =5,
        padding_b =2,
        kernel_size_g =3,
        padding_g =1,
    )
    V10 = ILViT_with_usual_net(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1 ,  #注意力机制中没有avgpool
        kernel_size_b =3,
        padding_b =1,
        kernel_size_g =5,
        padding_g =2,
    )

    
    V = {'ILViT':V0,'ILViT_no_avgppol':V1,'ILViT_dimequal_maxvit':V2,'ILViT_series96':V3,'ILViT_series48':V4,
         'ILViT_cbam':V5,'ILViT_grn':V6,'ILViT_b3_g3':V7,'ILViT_b5_g5':V8,'ILViT_b5_g3':V9,'ILViT_with_usual_net':V10
         }
    # V = {'ILViT':V0,'ILViT_no_avgppol':V1}
    model_ft = V[net]

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()# 定义损失函数
    
    if args.optim == 'sgd':
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)# 定义优化器：进行L2正则化
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)# 定义优化器：进行L2正则化
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=args.stepsize,gamma=args.gamma)# 学习率每5代衰减
    
    model = train_model(model=model_ft,# 开始训练模型
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.epochs,device=device,logger=logger,writer=writer)    
    model = test_model(model_ft, criterion, device,logger)
    speed = test_inference_speed(model_ft,device,logger)
    # input = torch.randn(1, 3, 224, 224).to(device)
    # flops, params = profile(model_ft, inputs=(input, ))
    # logger.info('flops:{}  params:{}'.format(flops,params))