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

 
#给配对策略B准备的提前参数   
wli_dir = ['WLI/IM/', 'WLI/GA/', 'WLI/Normal/']
lci_dir = ['LCI/IM/', 'LCI/GA/', 'LCI/Normal/']   
    
#给配对策略A准备的提前参数   
wli_dir_list0 = ['WLI/IM/贲门/','WLI/IM/胃底/','WLI/IM/胃窦/','WLI/IM/胃角/','WLI/IM/胃体/',
                'WLI/GA/贲门/','WLI/GA/胃底/','WLI/GA/胃窦/','WLI/GA/胃角/','WLI/GA/胃体/',
                'WLI/Normal/贲门/','WLI/Normal/胃底/','WLI/Normal/胃窦/','WLI/Normal/胃角/','WLI/Normal/胃体/']
lci_dir_list1 = ['LCI/IM/贲门/','LCI/IM/胃底/','LCI/IM/胃窦/','LCI/IM/胃角/','LCI/IM/胃体/',
                'LCI/GA/贲门/','LCI/GA/胃底/','LCI/GA/胃窦/','LCI/GA/胃角/','LCI/GA/胃体/',
                'LCI/Normal/贲门/','LCI/Normal/胃底/','LCI/Normal/胃窦/','LCI/Normal/胃角/','LCI/Normal/胃体/']

data_list_train = []
data_list_train_B = []  #放策略B配对数据集
data_list_val = []
data_list_len = [35,14,1183,600,599,  #2431
                 77,78,1482,524,1036,  #3197
                 534,631,740,448,1564]  #3917
data_all_len = [2431,3197,3917] 
wli_item_B = []
lci_item_B = []

#策略B数量
pair_count = [int(len_ * 0.2) for len_ in data_all_len]
#策略A          
for i in range(len(wli_dir_list0)):
    wli_dir = wli_dir_list0[i]
    lci_dir = lci_dir_list1[i]
    # 获取并随机打乱数据
    wli_data_list =  glob2.glob('/home/mch/data/huaxi_data/'+ wli_dir+'*')
    lci_data_list =  glob2.glob('/home/mch/data/huaxi_data/'+ lci_dir+'*')
    random.shuffle(wli_data_list)
    random.shuffle(lci_data_list)
    # 按80%和20%比例分割数据
    wli_item = wli_data_list[:int(data_list_len[i]*0.8)]
    wli_item_val =  wli_data_list[-int(data_list_len[i]*0.2):]
   
    lci_item = lci_data_list[:int(data_list_len[i]*0.8)]
    lci_item_val =  lci_data_list[-int(data_list_len[i]*0.2):]
    # 配对并组合数据
    item = [f'{wli}、{lci}' for wli, lci in zip(wli_item, lci_item)]
    item_val = [f'{wli}、{lci}' for wli, lci in zip(wli_item_val, lci_item_val)]
    
    #+策略B
    if args.pair == 'A+B':
        
       wli_item_B += wli_data_list[:int(data_list_len[i])]
       lci_item_B += lci_data_list[:int(data_list_len[i])]
       if (i+1)%5 == 0:
            random.shuffle(wli_item_B)
            random.shuffle(lci_item_B)
            # 20%的数据进行配对
            wli_pair = wli_item_B[:pair_count[i//5]]
            lci_pair = lci_item_B[:pair_count[i//5]]
            item += [f'{wli}、{lci}' for wli, lci in zip(wli_pair, lci_pair)]
            wli_item_B = []  #再次清空
            lci_item_B = []
           
           
    data_list_train.append(item)
    data_list_val.append(item_val)
    
   

# 均匀采样
data_list0_train = sum(data_list_train[0:5],[])
data_list1_train = sum(data_list_train[5:10],[])
data_list2_train = sum(data_list_train[10:15],[])


data_list0_val = sum(data_list_val[0:5],[])
data_list1_val = sum(data_list_val[5:10],[])
data_list2_val = sum(data_list_val[10:15],[])


# 根据不同模型选择对应的训练数据，并均匀采样，输出数据样本总量
# 不混合的实验
if args.dataset == 'IM':
    li = [0,2]
    print('Question:IM')
    print('data_number:',len(data_list0_train),len(data_list2_train))
elif args.dataset == 'GA':
    li = [1,2]
    print('Question:GA')
    print('data_number:',len(data_list1_train),len(data_list2_train))

# 在每一类中随机抽取制作训练集、测试集的标签，比例设置为8：2
# 制作数据集-标签字典、随机打乱、存入txt文档中
data_name_train = [data_list0_train,data_list1_train,data_list2_train]
data_name_val = [data_list0_val,data_list1_val,data_list2_val]

train_list0 = data_name_train[li[0]]
train_list1 = data_name_train[li[1]]
test_list0 = data_name_val[li[0]]
test_list1 = data_name_val[li[1]]
train_list, test_list={},{}
#制作label，有病为0，正常为1
for item in train_list0:
    train_list[item]=0  
for item in train_list1:
    train_list[item]=1
for item in test_list0:
    test_list[item]=0
for item in test_list1:
    test_list[item]=1 
random_train, random_test=[], []
for key, values in train_list.items():
        random_train.append(key+","+str(values))
random.shuffle(random_train)
for key, values in test_list.items():
        random_test.append(key+","+str(values))
random.shuffle(random_test)
if args.pair == 'A+B':
    B = '2'
else:
    B = ''
with open(f'/home/mch/our_model/Gate_LIVIT/{args.dataset}-train{B}.txt', 'w') as f:
    for i in random_train:
        f.write(i+"\n")
with open(f'/home/mch/our_model/Gate_LIVIT/{args.dataset}-val{B}.txt', 'w') as f:
    for i in random_test:
        f.write(i+"\n")

# 使用PIL Image读取图片
def default_loader(path):
    try:   
        img = Image.open(path)
        return img
    except:
        print("Cannot read image: {}".format(path))


# 图像增强处理
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



# 返回数组形式img和对应的标签，img对应进行了不同的变换
class customData(Dataset):# 最终返回img和label
    def __init__(self,  txt_path, dataset = '', data_trans1=None, data_trans2=None, loader = default_loader):
        with open(txt_path) as input_file:
            # 以','分隔获取对应的图像名和标签
            lines = input_file.readlines()
            self.img_name_x1 = [(line.strip().split('、')[0]) for line in lines]
            self.img_name_x2 = [(line.strip().split('、')[1].split(',')[0]) for line in lines]
            
            self.img_label = [int(line.strip().split(',')[-1]) for line in lines]
        self.trans1 = data_trans1
        self.trans2 = data_trans2
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len( self.img_label )

    def __getitem__(self, item):
        label = self.img_label[item]
        img_name_x1 = self.img_name_x1[item]
        img_name_x2 = self.img_name_x2[item]
      
        img1 = self.loader(img_name_x1)
        img2 = self.loader(img_name_x2)
        if self.dataset == 'train':
            img01 = self.trans1(img1)
            img02 = self.trans1(img2)
 
                
        if self.dataset == 'val':
            img01 = self.trans2(img1) 
            img02 = self.trans2(img2)
          
   
        img_ = [img01,img02]
 
        return img_, label

image_datasets = {x: customData(txt_path=(f'/home/mch/our_model/Gate_LIVIT/{args.dataset}'+'-' + x +B + '.txt'),
                                    data_trans1 = data_transforms['train'],
                                    data_trans2 = data_transforms['val'],          
                                    dataset= x ) for x in ['train', 'val']}
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batchsize,
                                                 num_workers=4,
                                                 shuffle = False,drop_last=True) for x in ['train','val']}

