import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
from arguments import args

from datasets import test_dataloders ,test_dataset_sizes

import torch
import time

def test_inference_speed(model, device,logger ):

    model.eval()
    start_time = time.time()
    # num_images = 0
    # 遍历测试数据集，进行推理
    with torch.no_grad():
        for images, _ in test_dataloders:
            # 将数据移到指定设备
            images = images.to(device)
            
            # 进行推理
            outputs = model(images)
            
            # 统计处理的图像数量
            # num_images += images.size(0)

    # 计算总时间
    total_time = time.time() - start_time

    # 计算每秒处理的图像数量
    images_per_sec =test_dataset_sizes / total_time
    logger.info(f"处理速度:{images_per_sec:.2F}张图像/秒")

    return images_per_sec





# 定义模型训练过程
def test_model(model, criterion, device,logger):
    test_best_acc = 0.0
    num_epochs =1
    test_LOSS,test_ACC = [] ,[]
    model.load_state_dict(torch.load(f'/home/mch/our_model/CNN/2_best_model/{args.model}-{args.dataset}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{args.epochs}/best_model.pkl'))
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        test_count_batch = 0
        model.train(False)  # 设置为验证模式
        test_running_loss = 0.0
        TP2, TN2, FP2, FN2 =0.0,0.0,0.0,0.0
  
          
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
  
    
  

#     return model
    
# class WarmUpLR(_LRScheduler):
#     """warmup_training learning rate scheduler
#     Args:
#         optimizer: optimzier(e.g. SGD)
#         total_iters: totoal_iters of warmup phase
#     """
#     def __init__(self, optimizer, total_iters, last_epoch=-1):

#         self.total_iters = total_iters
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         """we will use the first m batches, and set the learning
#         rate to base_lr * m / total_iters
#         """
#         return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
