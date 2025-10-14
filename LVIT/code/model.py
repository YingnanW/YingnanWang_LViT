#2600m 输出大小3*224*224-----64*112*112
import torch
from torch import nn
from torchsummary import summary
from thop import profile
from torch import nn, einsum
from functools import partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from arguments import args
import random
random.seed(42)
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class SKConv(nn.Module):
    def __init__(self, features, M=3, G=32, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        return feats_V


class Stem_conv(nn.Module):
    def __init__(self, in_features=3, mid_features=64,out_features=64, M=3, G=32, r=16, L=32,stride =None):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(Stem_conv, self).__init__()
        dwconv = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=7, stride=stride, padding=3, groups=in_features, bias=False),  # Depthwise convolution
            nn.Conv2d(in_features, mid_features, kernel_size=1, stride=1, bias=False)  # Pointwise convolution
        )
        self.conv1 = nn.Sequential(
            dwconv,
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=2, L=L)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
            )
        

        if in_features == out_features: # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x                 #3,224,224
        
        out = self.conv1(x)         #64,224*224
        out = self.conv2_sk(out)    #64,112*112
        out = self.conv3(out)       #64,112*112
        y= self.shortcut(residual)  #64,112*112
        return self.relu(out + y)   #64,112*112


class Residual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
 
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)   #丢弃部分样本

class CSA_Residual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x
    
def CSA(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.,
    kernel_size = None  ,
    padding = None
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1
    assert (dim_in % 16) == 0, 'dimension should be divisible by dimension per head __00'
    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1, groups=dim_in//32 ),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride = stride, padding = padding, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net =  CSA_Residual(net, dropout = dropout) #已经建立在net上面的样本丢弃

    return net



# attention related classes

class LightMSA(nn.Module):
    def __init__(
        self,
        dim,
        # dim_head = 32,
        dim_head = 16,#为了和maxvit做对比有一个相同的多头
        dropout = 0.,
        window_size = 7,
        sr_ratio=None,
        pool = None
    ):
    # """  sr_ratio (float)  : k, v resolution downsample ratio
    # Returns:
    #     x : LMSA attention result, the shape is (B, H, W, C) that is the same as inputs.
    # """
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'
        
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim*2, bias=False)
        # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=1)#输入7*7->5*5
            self.norm = nn.LayerNorm(dim) 
        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)#创建空字典
        #         # 定义嵌入层
        # embedding_layer = nn.Embedding(num_embeddings=10000, embedding_dim=50)

        # # 输入索引序列 (假设批量大小为2，序列长度为3)
        # input_indices = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # # 获取嵌入向量
        # embedded_vectors = embedding_layer(input_indices)

        # print(embedded_vectors.shape)  # 输出: torch.Size([2, 3, 50])

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        # print(grid)
        grid = rearrange(grid, 'c i j -> (i j) c')
        # print(grid)
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')#增加一维
        # print(rel_pos)# 计算每对位置之间的相对位置差
        rel_pos += window_size - 1     
        # print(rel_pos)#将相对位置差调整到正值范围 [0, 2*window_size-2]，即 [0, 12]
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        # print(rel_pos_indices1)
        if sr_ratio > 1:
             rel_pos_indices = rel_pos_indices[:,24:]  #24 = 49-25,需要一个49*25的相对位置
        # print(rel_pos_indices)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device,h = *x.shape, x.device, self.heads#这是一个大等式
        x = self.norm(x)

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
          
        # project for queries, keys, values
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d',h = h) # B,N,H,DIM -> B,H,N,DIM
        # print(h)
        # conv for down sample the x resoution for the k, v
        if self.sr_ratio > 1:
            x = rearrange(x, 'b (w1 w2) d -> b d w1 w2',w1 =window_height)
            x_reduce_resolution = self.sr(x)
            x_kv = rearrange(x_reduce_resolution, 'b d H W -> b (H W) d ')#通过stride=k卷积核后，H和W的值分别缩小变为w1和w2的1/k倍
            x_kv = self.norm(x_kv)
        else:
            x = rearrange(x, 'b (w1 w2) d -> b d w1 w2',w1 =window_height)
            x_kv = rearrange(x, 'b d H W -> b (H W) d ')
        
        kv_emb = rearrange(self.to_kv(x_kv), 'b N (dim h l ) -> l b h N dim', h=h, l=2)         # 2 B H N DIM
        k, v = kv_emb[0], kv_emb[1]   #2, B,H,N,DIM -> B,H,N,DIM
        
        # k, v = self.to_kv(x).chunk(3, dim = -1)



        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)#q与k的点积

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices) #这里加的位置偏置，bias是一个固定不变的数字
   
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  #张量操作的简明方法，可以进行各种矩阵乘法、点积、外积、转置、求和等操作。

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        # return out
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)


    
class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3 , stride = 1):
        super(DWCONV, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,
            stride = stride, padding = 1, groups = in_channels, bias = True
        )

    def forward(self, x):
        result = self.depthwise(x)

        return result
class IRFFN(nn.Module):
    """
    Inverted Residual Feed-forward Network
    """
    def __init__(self, in_channels, R=4):
        super(IRFFN, self).__init__()
        exp_channels = int(in_channels * R)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size = 1),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        result = x + self.conv2(self.dwconv(self.conv1(x)))
        return result



class ILViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,#每层得第一个维度，后面翻倍
        depth,
        # dim_head = 32,#注意力机制得维度
        dim_head = 16,
        dim_conv_stem = None,#卷积模块尺寸维度，默认为第一层卷积S0的输出维度
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        stride = 1,#如果是224就是1，如果是448*448输入，就是2
        channels = 3,
        sr_ratio = 3  #K V中的卷积核大小以及卷积跨步
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

   
        self.conv_stem = Stem_conv(
            in_features=3, mid_features=64,out_features=64, M=3, G=32, r=16, L=32,stride =stride)

        # variables
        
        
        # 假如有如下参数：num_stages = 3
        # dim = 64
        # dim_conv_stem = 32
        
        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        # print(dims)  # 输出: (64, 128, 256)
        dims = (dim_conv_stem, *dims)
        # print(dims)  # 输出: (32, 64, 128, 256)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        # dims[:-1] 将是 (32, 64, 128),dims[1:] 将是 (64, 128, 256),zip(dims[:-1], dims[1:]):
        # zip 函数将这两个元组“压缩”在一起，生成一对对的元素。
        #  具体来说，它将第一个元组的第 i 个元素和第二个元组的第 i 个元素配对。
        # print(dim_pairs)  # 输出: ((32, 64), (64, 128), (128, 256))
        self.layers1 = nn.ModuleList([])
        self.layers2 = nn.ModuleList([])


        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                #这段代码的意思是检查当前的 stage_ind 是否等于 0，并将结果（布尔值 True 或 False）赋值给变量 is_first
                stage_dim_in = layer_dim_in if is_first else layer_dim

                block1 = nn.Sequential(
                    CSA(
                        dim_in = stage_dim_in//2,#输入的维度
                        dim_out = layer_dim//2,#输出的维度
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate,
                        kernel_size = 3, #3-.>padding=1    kernel=5-->padding 2
                        padding=1
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
                    Residual(layer_dim//2, LightMSA(dim = layer_dim//2, dim_head = dim_head, dropout = dropout, window_size = w, sr_ratio = sr_ratio)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
                    Residual(layer_dim//2, IRFFN(in_channels = layer_dim//2)),
                    

                )
                block2 = nn.Sequential(
                    CSA(
                        dim_in = stage_dim_in//2,#输入的维度
                        dim_out = layer_dim//2,#输出的维度
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate,
                        kernel_size = 3, #3-.>padding=1    kernel=5-->padding 2
                        padding=1
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
                    Residual(layer_dim//2, LightMSA(dim = layer_dim//2, dim_head = dim_head, dropout = dropout, window_size = w, sr_ratio = sr_ratio)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
                    Residual(layer_dim//2, IRFFN(in_channels = layer_dim//2)),
                    

                )
                grid1 = nn.Sequential(
                    CSA(
                         dim_in = stage_dim_in//2,#输入的维度
                        dim_out = layer_dim//2,#输出的维度
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate,
                        kernel_size = 5,
                        padding= 2
                    ),
                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
                    Residual(layer_dim//2,  LightMSA(dim = layer_dim//2, dim_head = dim_head, dropout = dropout, window_size = w,sr_ratio = sr_ratio)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
                    Residual(layer_dim//2, IRFFN(in_channels = layer_dim//2)),
                    # Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                )
                grid2 = nn.Sequential(
                    CSA(
                         dim_in = stage_dim_in//2,#输入的维度
                        dim_out = layer_dim//2,#输出的维度
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate,
                        kernel_size = 5,
                        padding= 2
                    ),
                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
                    Residual(layer_dim//2,  LightMSA(dim = layer_dim//2, dim_head = dim_head, dropout = dropout, window_size = w,sr_ratio = sr_ratio)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
                    Residual(layer_dim//2, IRFFN(in_channels = layer_dim//2)),
                    # Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                )

                self.layers1.append(nn.ModuleList([block1,grid1]))
                self.layers2.append(nn.ModuleList([block2,grid2]))

        # mlp head out
       #独立分支1的输出
        self.mlp_head1 = nn.Sequential(
                Reduce('b d h w -> b d', 'mean'),
                nn.LayerNorm(dims[-1]),
                nn.Linear(dims[-1],32),
                nn.Dropout(dropout),
                nn.Linear(32, num_classes)
            )
        #独立分支2的输出
        self.mlp_head2 = nn.Sequential(
                Reduce('b d h w -> b d', 'mean'),
                nn.LayerNorm(dims[-1]),
                nn.Linear(dims[-1],32),
                nn.Dropout(dropout),
                nn.Linear(32, num_classes)
            )
        #合计输出
        self.mlp_head = nn.Sequential(
                Reduce('b d h w -> b d', 'mean'),
                nn.LayerNorm(dims[-1]*2),
                nn.Linear(dims[-1]*2,32*2),
                nn.Dropout(dropout),
                nn.Linear(32*2, num_classes)
            )

    def forward(self, x1,x2):
        x1 = self.conv_stem(x1)
        x2 = self.conv_stem(x2)
        for block1, grid1 in self.layers1:
                b, c, h, w = x1.shape
                x1a, x1b = x1[:, :c//2, :, :], x1[:, c//2:, :, :]
                x1a = block1(x1a)
                x1b = grid1(x1b)
                x1 = torch.cat([x1b, x1a], dim=1)#交换顺序 
        for block2, grid2 in self.layers2:
                b, c, h, w = x2.shape
                x2a, x2b = x2[:, :c//2, :, :], x2[:, c//2:, :, :]
                x2a = block2(x2a)
                x2b = grid2(x2b)
                x2 = torch.cat([x2b, x2a], dim=1)#交换顺序 


        y1 = self.mlp_head1(x1)
        y2 = self.mlp_head1(x2)
        x = torch.cat([x1,x2],dim =1)
        y =  self.mlp_head(x)
        return  y,y1,y2


V1 = ILViT(
        num_classes = 2,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
        # depth = (2, 2, 5, 2),  
        depth = (1, 2, 3, 1),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25, 
        stride =1  ,# 如果是448*448的输入，那么就是2
        dropout = 0.1   ,                  # dropout
        sr_ratio = 1   #注意力机制中没有avgpool
    )


# from torch.utils.tensorboard import SummaryWriter
# # 创建 SummaryWriter
# writer = SummaryWriter('runs/my_model_double_Ilvit')

# # 创建两个虚拟输入张量
# input1 = torch.randn(1, 3, 224, 224)
# input2 = torch.randn(1, 3, 224, 224)

# # 使用 add_graph 方法记录模型的计算图
# writer.add_graph(V1, (input1, input2))

# # 关闭 SummaryWriter
# writer.close()




