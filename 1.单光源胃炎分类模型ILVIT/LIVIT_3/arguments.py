import argparse
def args_parser():
    parser = argparse.ArgumentParser(description='模型参数设置')
    #模型和数据选择
    parser.add_argument('--dataset', type=str, default='WLIGA', choices=['LCIIM','LCIGA','WLIIM','WLIGA'], help="name of dataset")
    parser.add_argument('--model', type=str, default='ILViT', help='model name',
                        choices=['ILViT','ILViT_series96','ILViT_series48','ILViT_cbam','ILViT_grn',
                                 'ILViT_b3_g3','ILViT_b5_g5','ILViT_b5_g3','ILViT_with_usual_net'])
    #ILViT   ILViT_dimequal_maxvit ILViT_no_avgppol  MaxViT_parallnel_attention   MaxViT  MaxViT_skent
    #优化器相关选择
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (0.5~0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--milestones', type=float, default=[100], help='Attenuation steps in SGD')
    parser.add_argument('--gamma', type=float, default=0.8, help='Attenuation factor in SGD')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--stepsize', type=int, default=10)
    
   
    #训练其他配置
    parser.add_argument('--epochs', type=int, default=120, help='the number of epochs')
    parser.add_argument('--batchsize', type=int, default=32, help='the size of batch')
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--gpu', type=int,default=1, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--ngpu', type=bool, default=False, help='use n gpu')
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args
args = args_parser()