
import os
import sys
import json
import pickle
import random
import math
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
random.seed(42)
def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-2):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        # 0.001  ->  1
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha  
        else:
        # 1   ->0.01
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
from arguments import args
# if args.dataset == 'LCI' or args.dataset == 'WLI':
#     from datasets_3class import test_dataloders ,test_dataset_sizes# 导入数据集
# else:
from datasets import test_dataloders ,test_dataset_sizes# 导入数据集



import torch
import time

def test_inference_speed(model, device,logger ):

    model.eval()
    start_time = time.time()
    num_images = 0
    # 遍历测试数据集，进行推理
    with torch.no_grad():
        for images, _ in test_dataloders:
            # 将数据移到指定设备
            images = images.to(device)
            
            # 进行推理
            outputs = model(images)
            
            # 统计处理的图像数量
            num_images += images.size(0)

    # 计算总时间
    total_time = time.time() - start_time

    # 计算每秒处理的图像数量
    images_per_sec = num_images / total_time
    logger.info(f"处理速度:{images_per_sec:.2F}张图像/秒")

    return images_per_sec





# 定义模型训练过程
def test_model(model, criterion, device,logger):
    test_best_acc = 0.0
    num_epochs =1
    test_LOSS,test_ACC = [] ,[]
    model.load_state_dict(torch.load(f'/home/mch/our_model/LIVIT_3/2_best_model/{args.optim}-{args.model}-{args.dataset}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-wd{args.weight_decay}-epoch{args.epochs}/best_model.pkl'))
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        test_count_batch = 0
        model.train(False)  # 设置为验证模式
        test_running_loss = 0.0
        TP2, TN2, FP2, FN2, =0.0,0.0,0.0,0.0,
  
          
        with torch.no_grad():
            model.eval()
            for images2, labels2 in test_dataloders:
                test_count_batch += 1
                images2, labels2 = images2.to(device), labels2.to(device)   
                opt1 = model(images2)
                # if args.model == 'googlenet':
                #     opt1 = opt1.logits
                _,pred1 = torch.max(opt1,1)
                val_loss = criterion(opt1, labels2)
                test_running_loss += (val_loss.data.item()*args.batchsize)
            
                TP2 += ((pred1==0)&(labels2.data==0)).cpu().sum()
                TN2 += ((pred1==1)&(labels2.data==1)).cpu().sum()
                FN2 += ((pred1==1)&(labels2.data==0)).cpu().sum()
                FP2 += ((pred1==0)&(labels2.data==1)).cpu().sum()
                # print result every 10 batch

               
                            
            if test_dataset_sizes!= 0:
                test_epoch_loss = test_running_loss / test_dataset_sizes
            else:
                test_epoch_loss =0
            if TP2+TN2+FP2+FN2 != 0:
                    test_epoch_acc=(TP2+TN2)/(TP2+TN2+FP2+FN2)
                    test_sens = TP2/(TP2+FN2)# recall
                    test_spec = TN2/(TN2+FP2)
                    test_ppv = TP2/(TP2+FP2)# Precision
                    test_npv = TN2/(TN2+FN2)
                    test_F1 = 2 * (test_sens * test_ppv) / (test_sens + test_ppv)
            else:
                test_epoch_acc,test_sens,test_spec,test_ppv,test_npv = 0, 0, 0, 0, 0         
            print('{} Loss: {:.4f} ACC: {:.4f} SENS: {:.4f} SPEC: {:.4f} PPV: {:.4f} NPV: {:.4f} F1:{:.4f}'.format('test', test_epoch_loss, test_epoch_acc,test_sens,test_spec,test_ppv,test_npv,test_F1))
            test_LOSS.append(test_epoch_loss)
            test_ACC.append(float(test_epoch_acc))       
        # save model
        if test_epoch_acc > test_best_acc:
           
            test_best_acc = test_epoch_acc
            
        #输出日志
        logger.info(' epoch{} 测试test_Loss: {:.4f}  test_ACC: {:.4f}  ' .format(epoch,test_epoch_loss,test_epoch_acc))       
        logger.info('SENS: {:.4f} SPEC: {:.4f} PPV: {:.4f} NPV: {:.4f} F1:{:.4f}'.format(test_sens,test_spec,test_ppv,test_npv,test_F1))          
        logger.info(f"test_Best_acc:{test_best_acc:.4F}")
  
    
  

    return model
    
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
