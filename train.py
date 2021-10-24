import argparse
from pprint import pprint

import torch


from experiments.classification import ClassificationExperiment
from experiments.utils import init_Seed


def main():
    init_Seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vgg19', choices=['alexnet', 'resnet', 'resnet9', 'mobilenetv3', 'vgg13', 'vgg19'],
                        help='architecture (default: mobilenetv3)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='training epochs (default: 200)')

    parser.add_argument('--dataset', default='cifar100', choices=['cifar10',
                                                                 'cifar100',
                                                                 'caltech-101',
                                                                 'caltech-256',
                                                                 'imagenet1000'],
                        help='training dataset (default: cifar10)')
    parser.add_argument('--norm-type', default='bn', choices=['bn', 'gn', 'in', 'none'],
                        help='norm type (default: bn)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')

    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
    parser.add_argument('--pretrained-path',
                        help='load pretrained path')

    # Lookahead setting
    parser.add_argument('--LA', action='store_true', default=False,
                        help='enables lookahead')
    parser.add_argument('--LA_config', default='LA_config',
                        help='LA config python file')
    

    # SWA setting
    parser.add_argument('--SWA', action='store_true', default=False,
                        help='enables SWA')
    parser.add_argument('--SWA_config', default='SWA_config',
                        help='SWA config python file')
    
    # SGD
    parser.add_argument('--lr_config', default='SGD_config',
                        help='lr config python file')
    # paths

            
    # misc
    parser.add_argument('--save-interval', type=int, default=0,
                        help='save model interval')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='for evaluation')
    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')
    parser.add_argument('--tag',
                        help='tag')


    args = parser.parse_args()

    pprint(vars(args))

    exp = ClassificationExperiment(vars(args))


    exp.training()

    print('Training done at', exp.logdir)


if __name__ == '__main__':
    main()
