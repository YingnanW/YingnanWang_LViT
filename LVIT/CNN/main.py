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
from unit import test_model ,test_inference_speed
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

    logging.basicConfig(filename=f"/home/mch/our_model/CNN/1_log/{net}-{DATA}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{NUM_EPOCH}.log", level=logging.INFO)#保存日志
    logger  = logging.getLogger(__name__)
    writer = SummaryWriter(f"/home/mch/our_model/CNN/3_tesorboard_writer/{net}-{DATA}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{NUM_EPOCH}")
 
        
        
  
    # 设置模型在哪个gpu上运行
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = f'cuda:{args.gpu}'
    
    r1 = models.resnet50( pretrained=False) 
    r1_num_fits = r1.fc.in_features  #.in_features是一个属性，用于获取该层的输入特征数。
    r1.fc = nn.Linear(r1_num_fits, args.num_classes)
    
    r2 = models.resnet101( pretrained=False) 
    r2_num_fits = r2.fc.in_features  #.in_features是一个属性，用于获取该层的输入特征数。
    r2.fc = nn.Linear(r2_num_fits, args.num_classes)
    
    r3 = models.resnet152( pretrained=False) 
    r3_num_fits = r3.fc.in_features  #.in_features是一个属性，用于获取该层的输入特征数。
    r3.fc = nn.Linear(r3_num_fits, args.num_classes)
    
    vgg13 = models.vgg13(pretrained = False)
    vgg13_num_fits = vgg13.classifier[-1].in_features  #.in_features是一个属性，用于获取该层的输入特征数。
    vgg13.classifier[-1] = nn.Linear(vgg13_num_fits, args.num_classes)
    
    vgg16 = models.vgg16(pretrained = False)
    vgg16_num_fits = vgg16.classifier[-1].in_features  #.in_features是一个属性，用于获取该层的输入特征数。
    vgg16.classifier[-1] = nn.Linear(vgg16_num_fits, args.num_classes)
    
    vgg19 = models.vgg19(pretrained = False)
    vgg19_num_fits = vgg19.classifier[-1].in_features  #.in_features是一个属性，用于获取该层的输入特征数。
    vgg19.classifier[-1] = nn.Linear(vgg19_num_fits, args.num_classes)
    
    googlenet = models.googlenet(pretrained = False)
    googlenet_num_fits = googlenet.fc.in_features
    googlenet.fc = nn.Linear(googlenet_num_fits,args.num_classes)
 
   
    
    squeezenet1_1= models.squeezenet1_1(pretrained=True)
    
    squeezenet1_1.classifier[1]= nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    # print(squeezenet1_1)
    densenet121 = models.densenet121(pretrained = False)
    densenet121.classifier = nn.Linear(1024,args.num_classes,bias = True)
    
    densenet169 = models.densenet169(pretrained = False)
    densenet169.classifier = nn.Linear(1024,args.num_classes,bias = True)
    
    densenet201 = models.densenet201(pretrained = False)
    densenet201.classifier = nn.Linear(1024,args.num_classes,bias = True)
    
    mobilenet_v2 = models.mobilenet_v2(pretrained = False)
    mobilenet_v2.classifier[1] = nn.Linear(1280,args.num_classes,bias = True)
    
    efficientnet_b0 = models.efficientnet_b0(pretrained = False)
    efficientnet_b0.classifier[1] = nn.Linear(1280,args.num_classes,bias = True)
    
    convnext_s = models.convnext_small(pretrained = False)
    convnext_s.classifier[2]=nn.Linear(in_features=768, out_features=args.num_classes, bias=True)
   
    from edgenext import EdgeNeXt
    edgenext_s = EdgeNeXt()
   
    from thop import profile
    import torch
    import torchvision.models as models



    
    V = {'resnet50':r1,'resnet101':r2,'resnet152':r3,'vgg13':vgg13,'vgg16':vgg16,'vgg19':vgg19,
         'googlenet':googlenet,'squeezenet1_1':squeezenet1_1,'densenet121':densenet121,'densenet169':densenet169,
         'densenet201':densenet201,'mobilenet_v2':mobilenet_v2,'efficientnet_b0':efficientnet_b0,
         'convnext_s':convnext_s,'edgenext_s':edgenext_s}
    model_ft = V[net]
    input = torch.randn(1, 3, 224, 224)

    flops, params = profile(model_ft, inputs=(input, ))
    
    print(f"FLOPs: {flops}")
    print(f"参数量: {params}")
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()# 定义损失函数
    
    
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)# 定义优化器：进行L2正则化
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=args.stepsize,gamma=args.gamma)# 学习率每5代衰减
    
    
    
    model_ft = train_model(model=model_ft,# 开始训练模型
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.epochs,device=device,logger=logger,writer=writer)    
    model_ft = test_model(model_ft, criterion, device,logger)
 #   speed = test_inference_speed(model_ft,device,logger)
    # logger.info('flops:{}  params:{}'.format(flops,params))
    