# from model import model# 导入模型
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from arguments import args
from maxvit_conv_model import Maxvit_conv
import numpy as np
from train import train_model
from vit_pytorch.max_vit import MaxViT
from vit_pytorch.cross_vit import CrossViT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.max_vit_sknet import MaxViT_skent
from vit_pytorch.t2t_maxvit import T2T_maxvit
from torchvision.models import resnet50
from vit_pytorch.distill import DistillableViT, DistillWrapper,DistillableT2TViT,ViT
from deit_model import  deit_base_distilled_patch16_224
if __name__ == '__main__':
    start_time = time.time()
    # 训练前输出模型的重要设置参数
    print('gpu:',args.gpu)
    print('questionClass:',args.dataset)
    print('model:',args.net)
    print('Gepoch:',args.epochs)
    print('lr:',args.lr)
    # print('k:',args.nsubset)

    # 设置模型在哪个gpu上运行
    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")  
    v=  ViT(
    image_size = 896,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
    v0 = MaxViT(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
        depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
        dropout = 0.1                     # dropout
    )
    v1 = CrossViT(
        image_size = 256,
        num_classes = 2,
        depth = 4,               # number of multi-scale encoding blocks
        sm_dim = 192,            # high res dimension
        sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
        sm_enc_depth = 2,        # high res depth
        sm_enc_heads = 8,        # high res heads
        sm_enc_mlp_dim = 2048,   # high res feedforward dimension
        lg_dim = 384,            # low res dimension
        lg_patch_size = 64,      # low res patch size
        lg_enc_depth = 3,        # low res depth
        lg_enc_heads = 8,        # low res heads
        lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
        cross_attn_depth = 2,    # cross attention rounds
        cross_attn_heads = 8,    # cross attention heads
        dropout = 0.1,
        emb_dropout = 0.1
    )
    

    v2 = T2TViT(
        dim = 512,
        image_size = 896,
        depth = 5,
        heads = 8,
        mlp_dim = 512,
        num_classes = 2,
        t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    )


    teacher = resnet50(pretrained = True)
    num_fits = teacher.fc.in_features  #.in_features是一个属性，用于获取该层的输入特征数。
    teacher.fc = nn.Linear(num_fits, 2)
    v3 = DistillableViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 2,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    v4 = DistillableT2TViT(
        
    dim = 512,
    image_size = 224,
    depth = 5,
    heads = 8,
    mlp_dim = 512,
    num_classes = 2,
    t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module


    )
    distiller = DistillWrapper(
        student = v4,
        teacher = teacher,
        temperature = 3,           # temperature of distillation
        alpha = 0.5,               # trade between main loss and distillation loss
        hard = False               # whether to use soft or hard distillation
    )


    v5= Maxvit_conv
    v6 = T2T_maxvit(
        image_size = 896,
        
    )
    v7 =  deit_base_distilled_patch16_224
    # print (v7)
    v8 = MaxViT_skent(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
        depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
        dropout = 0.1                     # dropout
    )
    v = {'MaxViT_skent':v8,'ViT':v,'Maxvit':v0,'Cross-vit':v1, 'T2T-vit':v2,'Deit':v3,'Deit-t2t':v4,'Maxvit_conv':v5,'T2T_maxvit':v6,'pretrain_deit':v7}
    net = v[f'{args.net}']
    model_ft = net.to(device)
    
    '''
    #if args.gpu: torch.cuda.set_device(args.gpu)
    #model_ft = model.to('cuda')
    '''
    # 双迁移模型设置
    # if args.dtl:
    #     pretrained_file = '/home/mch/YangJie/output/'+'ourmodel_3'+'_'+'LCI'+args.dataset[-2:]+'/best_model_修改网络.pkl'
    #     # model_ft.load_state_dict(torch.load(pretrained_file).module.state_dict())
    #     model_ft.load_state_dict(torch.load(pretrained_file))
      
    # model_ft = torch.nn.DataParallel(model_ft.cuda(),device_ids = [0])程序并行操作
    criterion = nn.CrossEntropyLoss()# 定义损失函数
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)# 定义优化器：进行L2正则化
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=5,gamma=0.7)# 学习率每5代衰减
    model_ft = train_model(model=model_ft,# 开始训练模型
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.epochs,distiller=distiller)
    

  
