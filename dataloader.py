import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms
import numpy as np

# def prep_dataloader(args):


#     normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

#     train_transform = transforms.Compose([])

#     train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
#     train_transform.transforms.append(transforms.RandomHorizontalFlip())
    
#     train_transform.transforms.append(transforms.ToTensor())
#     train_transform.transforms.append(normalize)

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize])

#     if args['dataset'] == 'cifar10':
#         num_classes = 10
#         train_dataset = datasets.CIFAR10(root='data/cifar10',
#                                         train=True,
#                                         transform=train_transform,
#                                         download=True)

#         test_dataset = datasets.CIFAR10(root='data/cifar10',
#                                         train=False,
#                                         transform=test_transform,
#                                         download=True)
#     elif args['dataset'] == 'cifar100':
#         num_classes = 100
#         train_dataset = datasets.CIFAR100(root='data/cifar100',
#                                         train=True,
#                                         transform=train_transform,
#                                         download=True)

#         test_dataset = datasets.CIFAR100(root='data/cifar100',
#                                         train=False,
#                                         transform=test_transform,
#                                         download=True)

#     # Data Loader (Input Pipeline)
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                             batch_size=args['batch_size'],
#                                             shuffle=True,
#                                             pin_memory=True,
#                                             num_workers=2)

#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                             batch_size=args['batch_size'],
#                                             shuffle=False,
#                                             pin_memory=True,
#                                             num_workers=2)
    
#     return train_loader, test_loader


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os


def prep_dataloader(args, resize=None):
    if resize == None:
        resize = 32

    if args['dataset'] == "cifar100":
        transform_train = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(resize, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data/cifar100', train=True, download=True, transform=transform_train),
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=0
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data/cifar100', train=False, transform=transform_test),
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=0
        )
    elif args['dataset'] == "cifar10":
        transform_train = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(resize, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform_train),
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=0
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=False, transform=transform_test),
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=0
        )

    # elif args['dataset'] == "IMAGENET":
    #     traindir = os.path.join(args.data, 'train')
    #     valdir = os.path.join(args.data, 'val')

    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])

    #     train_dataset = datasets.ImageFolder(
    #         traindir,
    #         transforms.Compose([
    #             transforms.RandomResizedCrop(resize),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #         ]))

    #     # Check class labels
    #     # print(train_dataset.classes)

    #     if args.distributed:
    #         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     else:
    #         train_sampler = None

    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=(train_sampler is None),
    #         num_workers=args.workers,
    #         pin_memory=True,
    #         sampler=train_sampler
    #     )

    #     test_loader = torch.utils.data.DataLoader(
    #         datasets.ImageFolder(valdir, transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])),
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         num_workers=args.workers,
    #         pin_memory=True
    #     )

    return train_loader, test_loader