from torchvision import models
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from arguments import args
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import logging
from train import train_model
from unit import test_model,test_inference_speed
from vit_pytorch.deit_model import deit_tiny_patch16_224,deit_small_patch16_224,deit_base_patch16_224
from vit_pytorch.cait import cait_XXS24_224,cait_XS24_224,cait_S24_224
from vit_pytorch.cross_vit import crossvit_tiny_224,crossvit_small_224,crossvit_base_224
from vit_pytorch.t2t import t2t_vit_14,t2t_vit_19,t2t_vit_24
from vit_pytorch.swin_transformer import swin_t_224, swin_s_224,swin_b_224,swin_l_224
from timm.models import create_model
from vit_pytorch.max_vit import maxvit_t,maxvit_s,maxvit_b,maxvit_l
from thop import profile
# from vit_pytorch.max_vit_sknet import MaxViT_skent

# from vit_pytorch.distill import DistillableViT, DistillWrapper,DistillableT2TViT,ViT

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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#并行运算
    device = torch.device(f'cuda:{args.gpu}') # 默认使用 GPU
    NUM_EPOCH = args.epochs # 默认迭代次数
    lr = args.lr
    DATA = args.dataset
    net = args.model

    
    logging.basicConfig(filename=f"/home/mch/vit-pytorch/1_log/{net}-{DATA}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{NUM_EPOCH}.log", level=logging.INFO)
    logger  = logging.getLogger(__name__)
    writer = SummaryWriter(f"/home/mch/vit-pytorch/3_tensorboard_writer/{net}-{DATA}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{NUM_EPOCH}")

    # 设置模型在哪个gpu上运行
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = f'cuda:{args.gpu}'
    deit_t = deit_tiny_patch16_224() 
    deit_s = deit_small_patch16_224()
    deit_b = deit_base_patch16_224()
    crossvit_t = crossvit_tiny_224()
    crossvit_s =  crossvit_small_224()
    crossvit_b = crossvit_base_224()
    cait_XXS24 = cait_XXS24_224()
    cait_XS24 = cait_XS24_224()
    cait_S24 =  cait_S24_224()
    t2t_14 = t2t_vit_14()
    t2t_19 = t2t_vit_19()
    t2t_24 = t2t_vit_24()
    swin_t =  swin_t_224()
    swin_s =  swin_s_224()
    swin_b = swin_b_224()
    swin_l = swin_l_224()
    twins_svt_s = create_model('twins_svt_small',pretrained=False,num_classes=args.num_classes,
                               drop_rate=0.1,drop_path_rate=0.1, proj_drop_rate=0.1,attn_drop_rate=0.1)
    twins_svt_b = create_model('twins_svt_base',pretrained=False,num_classes=args.num_classes,
                               drop_rate=0.1,drop_path_rate=0.1, proj_drop_rate=0.1,attn_drop_rate=0.1)
    twins_svt_l = create_model('twins_svt_large',pretrained=False,num_classes=args.num_classes,
                               drop_rate=0.1,drop_path_rate=0.1, proj_drop_rate=0.1,attn_drop_rate=0.1)
    # from thop import profile
    # import torch
    # import torchvision.models as models

    # # 示例模型：ResNet-50
    # model =twins_svt_s
    # model.to('cuda:1')

    # # 创建一个与训练集输入形状相同的示例输入
    # input = torch.randn(1, 3,448, 448).to('cuda:1')

    # # 计算 FLOPs 和参数量
    # flops, params = profile(model, inputs=(input, ))

    # print(f"FLOPs: {flops}")
    # print(f"参数量: {params}")
 
    V = {'deit_tiny_patch16_224':deit_t ,'deit_small_patch16_224':deit_s,'deit_base_patch16_224':deit_b,
            'crossvit_tiny_224':crossvit_t ,'crossvit_small_224':crossvit_s, 'crossvit_base_224':crossvit_b,
            'cait_XXS24_224':cait_XXS24,'cait_XS24_224':cait_XS24,'cait_S24_224':cait_S24,
            't2t_vit_14':t2t_14,'t2t_vit_19':t2t_19,'t2t_vit_24':t2t_24,
            'swin_t_224':swin_t,'swin_s_224':swin_s,'swin_b_224':swin_b,'swin_l_224':swin_l,
            'twins_svt_small':twins_svt_s,'twins_svt_base':twins_svt_b,'twins_svt_large':twins_svt_l,
            'maxvit_t':maxvit_t,'maxvit_s':maxvit_s,'maxvit_b':maxvit_b,'maxvit_l':maxvit_l
            
         }
    model_ft = V[net]
    # input = torch.randn(1, 3, 224, 224)

    # flops, params = profile(model_ft, inputs=(input, ))
    
    # print(f"FLOPs: {flops}")
    # print(f"参数量: {params}")
    
    # model_ft= nn.DataParallel(model_ft)  #并行运算
    
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()# 定义损失函数
    
    if args.optim == 'sgd':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)# 定义优化器：进行L2正则化
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=args.stepsize,gamma=args.gamma)# 学习率每5代衰减

    if args.optim == 'adam':   
        
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-5)
    # scheduler
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.7)
    
    model_ft = train_model(model=model_ft,# 开始训练模型
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.epochs,device=device,logger=logger,writer=writer)    
    model_ft = test_model(model_ft, criterion, device,logger)
    speed = test_inference_speed(model_ft,device,logger)