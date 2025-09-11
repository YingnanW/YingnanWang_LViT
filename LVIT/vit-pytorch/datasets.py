import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import InterpolationMode
import os
import glob2# 使用glob2读取文件夹内的数据集地址，方便后续直接用cv2.imread直接进行读取
from torch.utils.data import Dataset
from torchvision import transforms
import random
from torchvision.transforms.functional import InterpolationMode
from arguments import args
#随机种子
def set_random_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

# 读取数据列表

# def measure(list0):
#     return int(len(list0)*0.9)
# # 训练集 &&验证集 && 测试集
# label0 = ['WLI/IM/贲门/','WLI/IM/胃底/','WLI/IM/胃窦/','WLI/IM/胃角/','WLI/IM/胃体/',
#                 'WLI/GA/贲门/','WLI/GA/胃底/','WLI/GA/胃窦/','WLI/GA/胃角/','WLI/GA/胃体/',
#                 'WLI/Normal/贲门/','WLI/Normal/胃底/','WLI/Normal/胃窦/','WLI/Normal/胃角/','WLI/Normal/胃体/']
# label1 = ['LCI/IM/贲门/','LCI/IM/胃底/','LCI/IM/胃窦/','LCI/IM/胃角/','LCI/IM/胃体/',
#                 'LCI/GA/贲门/','LCI/GA/胃底/','LCI/GA/胃窦/','LCI/GA/胃角/','LCI/GA/胃体/',
#                 'LCI/Normal/贲门/','LCI/Normal/胃底/','LCI/Normal/胃窦/','LCI/Normal/胃角/','LCI/Normal/胃体/']
# label_list_all = label0+label1
# data_list_len=[42,14,1183,600,599,
#                77,99,1600,569,1036,
#                534,1074,1095,649,2501,
#                35,17,1247,602,648,
#                94,78,1482,524,1092,
#                545,631,740,448,1564]
    
# # 初始化变量
# data_list_train = []
# data_list_val = []
# data_list_test = []

# for i in range(len(label_list_all)):
#     label_list = label_list_all[i]
#     data_list = glob2.glob('/home/mch/data/huaxi_data/'+label_list+'*')

#     # 打乱数据集
#     random.shuffle(data_list)

#     # 计算各个子集的大小
#     train_size = int(data_list_len[i] * 0.8)
#     val_size = int(data_list_len[i] * 0.1)
#     test_size = data_list_len[i] - train_size - val_size  # 剩余的作为测试集

#     # 划分数据集
#     item_train = data_list[:train_size]
#     item_val = data_list[train_size:train_size + val_size]
#     item_test = data_list[train_size + val_size:train_size + val_size + test_size]

#     data_list_train.append(item_train)
#     data_list_val.append(item_val)
#     data_list_test.append(item_test)
# # 统计数量
# data_list0_train = sum(data_list_train[0:5],[])   #WLI/IM
# data_list1_train = sum(data_list_train[5:10],[])  #WLI/GA
# data_list2_train = sum(data_list_train[10:15],[])  #WLI/Normal
# data_list3_train = sum(data_list_train[15:20],[])
# data_list4_train = sum(data_list_train[20:25],[])
# data_list5_train = sum(data_list_train[25:30],[])

# data_list0_val = sum(data_list_val[0:5],[])
# data_list1_val = sum(data_list_val[5:10],[])
# data_list2_val = sum(data_list_val[10:15],[])
# data_list3_val = sum(data_list_val[15:20],[])
# data_list4_val = sum(data_list_val[20:25],[])
# data_list5_val = sum(data_list_val[25:30],[])

# data_list0_test = sum(data_list_test[0:5],[])
# data_list1_test = sum(data_list_test[5:10],[])
# data_list2_test = sum(data_list_test[10:15],[])
# data_list3_test = sum(data_list_test[15:20],[])
# data_list4_test = sum(data_list_test[20:25],[])
# data_list5_test = sum(data_list_test[25:30],[])

# if args.dataset == 'WLIIM':
#     li = [0,2]
#     print('Question:WLIIM')
#     print('data_number:  患病 {}, 正常 {} ,合计 {}'.format(len(data_list0_train),len(data_list2_train),len(data_list0_train)+len(data_list2_train)))
    
# elif args.dataset == 'WLIGA':
#     li = [1,2]
#     print('Question:WLIGA')
#     # print('data_number:',len(data_list1_train),len(data_list2_train))
#     print('data_number:  患病 {}, 正常 {} ,合计 {}'.format(len(data_list1_train),len(data_list2_train),len(data_list1_train)+len(data_list2_train)))
# elif args.dataset == 'LCIIM':
#     li = [3,5]
#     print('Question:LCIIM')
#     # print('data_number:',len(data_list3_train),len(data_list5_train))
#     print('data_number:  患病 {}, 正常 {} ,合计 {}'.format(len(data_list3_train),len(data_list5_train),len(data_list3_train)+len(data_list5_train)))
# elif args.dataset == 'LCIGA':
#     li = [4,5]
#     print('Question:LCIGA')
#     # print('data_number:',len(data_list4_train),len(data_list5_train))
#     print('data_number:  患病 {}, 正常 {} ,合计 {}'.format(len(data_list4_train),len(data_list5_train),len(data_list4_train)+len(data_list5_train)))
# # WLI&&LCI混合的实验
# elif args.dataset == 'WLI+LCI-IM':
#     data_list3_train = data_list3_train+data_list0_train
#     data_list5_train = data_list5_train+data_list2_train
#     data_list3_val = data_list3_val+data_list0_val
#     data_list5_val = data_list5_val+data_list2_val
#     data_list3_test = data_list3_test+data_list0_test
#     data_list5_test = data_list5_test+data_list2_test
#     li = [3,5]
#     print('Question:WLI+LCI-IM')
#     print('data_number:',len(data_list3_train),len(data_list5_train))
#     # print('data_number:  患病 {}, 正常 {} ,合计 {}',format(len(data_list0_train),len(data_list2_train),len(data_list0_train)+len(data_list2_train)))
# elif args.dataset == 'WLI+LCI-GA':
#     data_list4_train = data_list4_train+data_list1_train
#     data_list5_train = data_list5_train+data_list2_train
#     data_list4_val = data_list4_val+data_list1_val
#     data_list5_val = data_list5_val+data_list2_val
#     data_list4_test = data_list4_test+data_list1_test
#     data_list5_test = data_list5_test+data_list2_test
#     li = [4,5]
#     print('Question:WLI+LCI-GA')
#     print('data_number:',len(data_list4_train),len(data_list5_train))
#     # print('data_number:  患病 {}, 正常 {} ,合计 {}',format(len(data_list0_train),len(data_list2_train),len(data_list0_train)+len(data_list2_train)))
    
    
# # 在每一类中随机抽取制作训练集、验证集、测试集的标签，比例设置为8：：1：1
# # 制作数据集-标签字典、随机打乱、存入txt文档中
# data_name_train = [data_list0_train,data_list1_train,data_list2_train,data_list3_train,data_list4_train,data_list5_train]
# data_name_val = [data_list0_val,data_list1_val,data_list2_val,data_list3_val,data_list4_val,data_list5_val]
# data_name_test = [data_list0_test,data_list1_test,data_list2_test,data_list3_test,data_list4_test,data_list5_test]

# train_list0 = data_name_train[li[0]]                                                                                       
# train_list1 = data_name_train[li[1]]                                                                                        
# val_list0 = data_name_val[li[0]]
# val_list1 = data_name_val[li[1]]
# test_list0 = data_name_test[li[0]]
# test_list1 = data_name_test[li[1]]
# #空字典
# train_list,val_list, test_list={},{},{}
# for item in train_list0:
#     train_list[item]=0
# for item in train_list1:
#     train_list[item]=1
    
# for item in val_list0:
#     val_list[item]=0
# for item in val_list1:
#     val_list[item]=1 

# for item in test_list0:
#     test_list[item]=0
# for item in test_list1:
#     test_list[item]=1 
# random_train,random_val, random_test=[], [], []
# for key, values in train_list.items():
#         random_train.append(key+","+str(values))
# random.shuffle(random_train)
# for key, values in val_list.items():
#         random_val.append(key+","+str(values))
# random.shuffle(random_val)
# for key, values in test_list.items():
#         random_test.append(key+","+str(values))
# random.shuffle(random_test)
# with open(f'/home/mch/{args.dataset}-train.txt', 'w') as f:
#     for i in random_train:
#         f.write(i+"\n")
# with open(f'/home/mch/{args.dataset}-val.txt', 'w') as f:
#     for i in random_val:
#         f.write(i+"\n")
# with open(f'/home/mch/{args.dataset}-test.txt', 'w') as f:
#     for i in random_test:
#         f.write(i+"\n")
def default_loader(path):
        try:   
            img = Image.open(path)
            return img
        except:
            print("Cannot read image: {}".format(path)) 

data_transforms={
            'train':transforms.Compose([
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5)]),             # 随机垂直翻转
                    transforms.RandomRotation(degrees=180),           # 随机旋转，范围为0到180度
                    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), 
                                    scale=None, shear=None, interpolation=InterpolationMode.BILINEAR, fill=0), # 随机水平和垂直平移，平移范围为宽度和高度的20%
                    transforms.Resize(1000)                   # 缩放到1000x800
                                                ]),
                    transforms.Resize((256,256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4931,0.3016,0.2211],[0.3398,0.215,0.1612]) # 各通道颜色的均值和方差,用于归一化
                ]),
            'val':transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.CenterCrop(224),        
                    transforms.ToTensor(),
                    transforms.Normalize([0.4931,0.3016,0.2211],[0.3398,0.215,0.1612])
                ])
        }
 # 定义自己的数据类
class CustomImageLoader(data.Dataset):
            ##自定义类型数据输入
            def __init__(self, img_path, txt_path, data_transforms1=None, loader = default_loader,):

                with open(txt_path) as files:
                    lines = files.readlines() 
                    self.imgs = [os.path.join(img_path,line.strip().split(",")[0]) for line in lines]
                    self.labels = [int(line.strip().split(',')[-1]) for line in lines]
                self.data_tranforms1 = data_transforms1
                self.loader = loader
            def __len__(self):
                return len(self.imgs)
            def __getitem__(self, item):   
                img_name = self.imgs[item]
                label = self.labels[item]
                img = self.loader(img_name)
                img1 = self.data_tranforms1(img)
                return img1, label



# 返回数组形式img和对应的标签，img对应进行了不同的变换
class customData(Dataset):# 最终返回img和label
    def __init__(self, img_path, txt_path, dataset = '', data_trans1=None, data_trans2=None, loader = default_loader):
        with open(txt_path) as input_file:
            # 以','分隔获取对应的图像名和标签
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(',')[0]) for line in lines]
            self.img_label = [int(line.strip().split(',')[-1]) for line in lines]
        self.trans1 = data_trans1
        self.trans2 = data_trans2
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        if self.dataset == 'train':
            img = self.trans1(img)
        if self.dataset == 'val':
            img = self.trans2(img)
        if self.dataset == 'test':
            img = self.trans2(img)

 
        return img, label

image_datasets = {x: customData(img_path='',
                                    txt_path=(f'/home/mch/{args.dataset}'+'-' + x + '.txt'),
                                    data_trans1 = data_transforms['train'],
                                    data_trans2 = data_transforms['val'],
                             
                                    dataset= x ) for x in ['train', 'val']}
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batchsize,
                                                 num_workers=2,
                                                 shuffle = False,drop_last=True) for x in ['train','val']}

test_image_datasets = customData(img_path='',
                                    txt_path=(f'/home/mch/{args.dataset}'+'-' + 'test' + '.txt'),
                                    data_trans1 = data_transforms['train'],
                                    data_trans2 = data_transforms['val'],                            
                                    dataset= 'test')                                                              
test_dataloders = torch.utils.data.DataLoader(test_image_datasets,
                                             batch_size=32,
                                             num_workers=4,
                                             shuffle = True,
                                             drop_last=False)
test_dataset_sizes =  len(test_image_datasets)