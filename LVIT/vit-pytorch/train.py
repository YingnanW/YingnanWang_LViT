from datasets import image_datasets,dataloders# 导入数据集
import os
import time
import torch
import numpy as np
from arguments import args



dataset_sizes = {'train': len(image_datasets['train']),'val': len(image_datasets['val'])}
print(dataset_sizes)
batch_size = args.batchsize

# 定义模型训练过程
def train_model(model, criterion, optimizer, scheduler, num_epochs, device,logger,writer):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
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
                    if args.model == 'googlenet':
                        opt = opt.logits
                    _,preds = torch.max(opt,1)
                    loss = criterion(opt, labels)  #得到的是一个batch的平均损失
                    loss.backward()
                    optimizer.step()
                  
                if phase == 'val':
                    with torch.no_grad():
                        model.eval()
                        count_batch += 1
                        opt = model(images)
                # if args.model == 'googlenet':
                #     opt1 = opt1.logits
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
                if not os.path.exists(f'/home/mch/vit-pytorch/2_best_model/{args.model}-{args.dataset}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{args.epochs}/'):
                    os.makedirs(f'/home/mch/vit-pytorch/2_best_model/{args.model}-{args.dataset}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{args.epochs}/')
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts,f'/home/mch/vit-pytorch/2_best_model/{args.model}-{args.dataset}-lr{args.lr}-momentum{args.momentum}-gamma{args.gamma}-epoch{args.epochs}/best_model.pkl')# 存储最好的模型
            if phase == 'train':
                 writer.add_scalar(f'train_loss',epoch_loss, epoch)
                 writer.add_scalar(f'train_acc', epoch_acc, epoch)
                
            if phase == 'val':
                 LOSS.append(epoch_loss)
                 ACC.append(float(epoch_acc))
                 logger.info('\n【Val】:  SENS: {:.4f}  SPEC: {:.4f}  PPV: {:.4f}  NPV: {:.4f} F1:{:.4f} '.format(sens,spec,ppv,npv,F1))
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

    return model