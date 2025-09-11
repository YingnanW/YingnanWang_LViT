from datasets import image_datasets,dataloders# 导入数据集
import os
import time
import torch
import numpy as np
from arguments import args


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

dataset_sizes = {'train': len(image_datasets['train']),'val': len(image_datasets['val'])}
print(dataset_sizes)
batch_size = args.batchsize

# 定义模型训练过程
def train_model(model, criterion, optimizer, scheduler, num_epochs, device,logger,writer):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_sens,best_spec,best_ppv,best_npv = 0,0,0,0
    ACC,LOSS = [], []


    for epoch in range(num_epochs):
        begin_time = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        # 区分训练阶段和验证阶段
        for phase in ['train', 'val']:
            count_batch = 0
            if phase == 'train':
                scheduler.step()
                model.train(True)  # 设置为训练模式
            else:model.train(False)  # 设置为验证模式
            running_loss = 0.0
            TP, TN, FP, FN =0.0,0.0,0.0,0.0
            
            for images, labels in dataloders[phase]:               
                images, labels = images.to(device), labels.to(device)
                if phase == 'train':
                    count_batch += 1
                    optimizer.zero_grad()
                    opt = model(images)
                    
                    if type(opt) is dict:                       
                #   A training-time RepVGGplus outputs a dict. The items are:
                    #   'main':     the output of the final layer
                    #   '*aux*':    the output of auxiliary classifiers
                        loss = 0
                        for name, pred in opt.items():
                            if 'aux' in name:
                                loss += 0.1 * criterion(pred, labels)          #  Assume "criterion" is cross-entropy for classification
                            else:
                                loss += criterion(pred,labels)
                    else:
                        loss = criterion(opt,labels)   
                    
                    
                
                    _,preds = torch.max(opt['main'],1)
                    #loss = criterion(opt, labels)  #得到的是一个batch的平均损失
                    loss.backward()
                    optimizer.step()
                  
                if phase == 'val':
                    with torch.no_grad():
                        model.eval()
                        count_batch += 1
                        opt = model(images)
                # if args.model == 'googlenet':
                #     opt1 = opt1.logits
                
                
                        if type(opt) is dict:
                           opt = opt['main']
                        _,preds = torch.max(opt,1)
                        loss = criterion(opt, labels)
                        
                running_loss += (loss.data.item()*batch_size)

                TP += ((preds==0)&(labels.data==0)).cpu().sum()
                TN += ((preds==1)&(labels.data==1)).cpu().sum()
                FN += ((preds==1)&(labels.data==0)).cpu().sum()
                FP += ((preds==0)&(labels.data==1)).cpu().sum()
                # print result every 10 batch
                if count_batch%10 == 0:
                    batch_loss = running_loss / (batch_size*count_batch)
                    batch_acc = (TP+TN)/(TP+TN+FP+FN)
                    print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                          format(phase, epoch+1, count_batch, batch_loss, batch_acc, time.time()-begin_time))
                    begin_time = time.time()

            if dataset_sizes[phase]!= 0:
                epoch_loss = running_loss / dataset_sizes[phase]
            else:
                epoch_loss =0
            if TP+TN+FP+FN != 0:
                epoch_acc=(TP+TN)/(TP+TN+FP+FN)
                sens = TP/(TP+FN)# recall
                spec = TN/(TN+FP)
                ppv = TP/(TP+FP)# Precision
                npv = TN/(TN+FN)
                F1 = 2 * (sens * ppv) / (sens + ppv)
                
            else:
                epoch_acc,sens,spec,ppv,npv = 0, 0, 0, 0, 0
            logger.info('\n【{}】epoch:{}/{} loss:{:.4f} acc:{:.4f}'.format(phase,epoch,num_epochs,epoch_loss,epoch_acc))
          
            print('{} Loss: {:.4f} ACC: {:.4f} SENS: {:.4f} SPEC: {:.4f} PPV: {:.4f} NPV: {:.4f}'.format(phase, epoch_loss, epoch_acc,sens,spec,ppv,npv))
            # save model
          
               
            if phase == 'val' and epoch_acc > best_acc:
                if not os.path.exists(f'/home/mch/our_model/CNN/2_best_model/{args.model}-{args.dataset}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{args.epochs}/'):
                    os.makedirs(f'/home/mch/our_model/CNN/2_best_model/{args.model}-{args.dataset}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{args.epochs}/')
                best_sens,best_spec,best_ppv,best_npv = sens,spec,ppv,npv
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts,f'/home/mch/our_model/CNN/2_best_model/{args.model}-{args.dataset}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{args.epochs}/best_model.pkl')# 存储最好的模型
            if phase == 'train':
                 writer.add_scalar(f'train_loss',epoch_loss, epoch)
                 writer.add_scalar(f'train_acc', epoch_acc, epoch)
                
            if phase == 'val':
                 LOSS.append(epoch_loss)
                 ACC.append(float(epoch_acc))
                 logger.info('\n【Val】:  SENS: {:.4f}  SPEC: {:.4f}  PPV: {:.4f}  NPV: {:.4f} F1:{:.4f}'.format(sens,spec,ppv,npv,F1))
                 logger.info(f"验证的best_acc:{best_acc:.4f}")
                 writer.add_scalar(f'val_loss',epoch_loss, epoch)
                 writer.add_scalar(f'val_acc', epoch_acc, epoch)
                 writer.add_scalar(f'验证的best_acc',best_acc, epoch)
                 writer.add_scalar(f'val_SENS',sens, epoch)
                 writer.add_scalar(f'val_SPEC',spec, epoch)
                 writer.add_scalar(f'val_PPV',ppv, epoch)
                 writer.add_scalar(f'val_NPV',npv, epoch)
        # 终止迭代
        logger.info("-----"*20)
        # if len(ACC)>10:
        #     if len(list(set([i for i in ACC[-5:]])))==1:
        #         break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    logger.info("*****"*20)
    loss = ["{:.4f}".format(val_loss) for val_loss in LOSS]
    acc = ["{:.4f}".format(val_acc) for val_acc in ACC]
    logger.info('Val_LOSS:{}'.format(loss))
    logger.info('Val_ACC:{}'.format(acc))
    logger.info("*****"*20)
    # logger.info('epoch:{} val_best_acc:{:.4f} SENS: {:.4f}  SPEC: {:.4f}  PPV: {:.4f}  NPV: {:.4f}'.format(best_epoch,best_acc,best_sens,best_spec,best_ppv,best_npv))

    return model





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
    
    
    from repvggplus import create_RepVGGplus_by_name   
    repvggplus =  create_RepVGGplus_by_name(net, deploy=False, use_checkpoint=False)
    repvggplus.stage3_first_aux[3] = nn.Linear(640, 2, bias=True)
 

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
   
    from thop import profile
    import torch
    import torchvision.models as models

   
    
    V = {'resnet50':r1,'resnet101':r2,'resnet152':r3,'vgg13':vgg13,'vgg16':vgg16,'vgg19':vgg19,
         'googlenet':googlenet,'squeezenet1_1':squeezenet1_1,'densenet121':densenet121,'densenet169':densenet169,
         'densenet201':densenet201,'mobilenet_v2':mobilenet_v2,'efficientnet_b0':efficientnet_b0,
         'convnext_s':convnext_s,'RepVGGplus-L2pse':repvggplus}
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
#    speed = test_inference_speed(model_ft,device,logger)
    # logger.info('flops:{}  params:{}'.format(flops,params))