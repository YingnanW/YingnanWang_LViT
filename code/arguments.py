import argparse
def args_parser():
    parser = argparse.ArgumentParser(description='模型参数设置')

    parser.add_argument('--dataset', type=str, default='IM', choices=['IM','GA'], help="name of dataset")
    parser.add_argument('--model', type=str, default='double_ILViT_GFM2', help='model name')
    
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (0.5~0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--milestones', type=float, default=[100], help='Attenuation steps in SGD')
    parser.add_argument('--gamma', type=float, default=0.8, help='Attenuation factor in SGD')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--stepsize', type=int, default=10)
    
    parser.add_argument('--b', type=float, default=0.3) 
    parser.add_argument('--pair', type=str, default='A' ,choices=['A','A+B'])
    parser.add_argument('--output', type=str, default='more' ,choices=['main','more'])
    parser.add_argument('--fushion', type=str, default='mix',choices=['mix','finall'])
    parser.add_argument('--epochs', type=int, default=150, help='the number of epochs')
    parser.add_argument('--batchsize', type=int, default=24, help='the size of batch')
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--gpu', type=int,default=1, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--ngpu', type=bool, default=False, help='use n gpu')
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args
args = args_parser()
