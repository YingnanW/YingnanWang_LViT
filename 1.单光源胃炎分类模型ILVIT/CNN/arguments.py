import argparse
def args_parser():
    parser = argparse.ArgumentParser(description='模型参数设置')
    #模型和数据选择
    parser.add_argument('--dataset', type=str, default='LCIIM', choices=['LCIIM','LCIGA','WLIIM','WLIGA'], help="name of dataset")
    parser.add_argument('--model', type=str, default='edgenext_s', help='model name',
                        choices=['resnet50','resnet101','resnet152','vgg13','vgg16','vgg19',
                                 'googlenet','densenet121','densenet169','densenet201',
                                 'mobilenet_v2','efficientnet_b0', 'convnext_s','RepVGGplus-L2pse','edgenext_s'])
    #ILViT   ILViT_dimequal_maxvit ILViT_no_avgppol  MaxViT_parallnel_attention   MaxViT  MaxViT_skent
    #优化器相关选择

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.7, help='SGD momentum (0.5~0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--milestones', type=float, default=[100], help='Attenuation steps in SGD')
    parser.add_argument('--gamma', type=float, default=0.8, help='Attenuation factor in SGD')
    # parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--stepsize', type=int, default=10)
    
   
    #训练其他配置
    parser.add_argument('--epochs', type=int, default=150, help='the number of epochs')
    parser.add_argument('--batchsize', type=int, default=32, help='the size of batch')
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--gpu', default=0, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--ngpu', type=bool, default=False, help='use n gpu')
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args
args = args_parser()