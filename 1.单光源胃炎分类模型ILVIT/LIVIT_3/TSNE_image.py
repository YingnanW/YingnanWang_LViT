import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from datasets import image_datasets,dataloders
from arguments import args
# === 1. 加载模型结构 ===
#from model import ILViT  # 改成你实际的模型定义文件名
from model_contrast import ILViT_contrast
model_ft = ILViT_contrast(
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
device = torch.device(f'cuda:{args.gpu}') 

# model_ft =  nn.DataParallel(model_ft)
model_ft = model_ft.to(device)

# === 3. 加载模型参数 ===
#LCIGA
#checkpoint = torch.load(f'/home/mch/our_model/LIVIT_3/2_best_model/ILViT_no_avgppol-LCIGA-lr0.01-momentum0.9-gamma0.8-epoch150/best_model.pkl', map_location=device)
#WLIGA 
checkpoint = torch.load(f'/home/mch/our_model/LIVIT_3/2_best_model/sgd-ILViT_b3_g3-WLIGA-lr0.01-momentum0.9-gamma0.8-wd0.0005-epoch120/best_model.pkl', map_location=device)
##LCIIM (有点偏差)
#checkpoint = torch.load(f'/home/mch/our_model/LIVIT_3/2_best_model/sgd-ILViT_b3_g3-LCIIM-lr0.01-momentum0.9-gamma0.8-wd0.0005-epoch120/best_model.pkl', map_location=device)
# #WLIIM
#checkpoint = torch.load(f'/home/mch/our_model/LIVIT_3/2_best_model/sgd-ILViT_no_avgppol-WLIIM-lr0.02-momentum0.9-gamma0.8-wd0.0005-epoch120/best_model.pkl', map_location=device)

model_ft.load_state_dict(checkpoint)
# === 2. 替换分类头的后两层为 Identity，提取64维特征 ===

model_ft.mlp_head[3] = nn.Identity()
model_ft.mlp_head[4] = nn.Identity()
model_ft.eval()

# === 4. 遍历测试集，提取特征和标签 ===
all_features = []
all_labels = []



with torch.no_grad():
    for images, labels in dataloders['val']:
        images, labels = images.to(device), labels.to(device)       

        feats = model_ft(images)  # 输出为 [B, 64]
        all_features.append(feats.cpu())
        all_labels.append(labels.cpu())

features = torch.cat(all_features).numpy()
labels = torch.cat(all_labels).numpy()

# === 5. T-SNE 降维并绘图 ===
features_2d = TSNE(n_components=2, random_state=42).fit_transform(features)

plt.figure(figsize=(10, 8))
colors = ['red', 'blue']
label_names = [f'{args.dataset}', 'Normal']

for i, label_name in enumerate(label_names):
    idx = labels == i
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=label_name, alpha=0.6, c=colors[i])


plt.legend(fontsize=18) 
plt.title(f'T-SNE Visualization of ILViT_{args.dataset}_ Features')
plt.savefig(f'/home/mch/our_model/LIVIT_3/tsne_ilvit_{args.dataset}_2.pdf')

