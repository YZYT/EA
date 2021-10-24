import os
from pprint import pprint

import torch
import torch.optim as optim
from torch import nn

from dataset import prepare_dataset, prepare_wm
from dataloader import prep_dataloader
from experiments.base import Experiment
from experiments.trainer import Trainer, Tester
from experiments.trainer_private import TesterPrivate
from experiments.utils import construct_passport_kwargs, load_passport_model_to_normal_model, \
    load_normal_model_to_passport_model, load_normal_model_to_normal_model
from models.alexnet_normal import AlexNetNormal
from models.alexnet_passport import AlexNetPassport
# from models.resnet_normal import ResNet18, ResNet9
from models.resnet import ResNet18
# from models.mobilenetv3 import mobilenetv3
from models.VGG import VGG13, VGG19
from models.resnet_passport import ResNet18Passport, ResNet9Passport
from optimizers.SWA import SWA
from optimizers.Lookahead import Lookahead

from configs import lr_configs


class ClassificationExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)

        self.in_channels = 1 if self.dataset == 'mnist' else 3
        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech-101': 101,
            'caltech-256': 256,
            'imagenet1000': 1000
        }[self.dataset]

        """
        use mine data augumentation
        """
        # self.train_data, self.valid_data = prepare_dataset(self.args)
        self.train_data, self.valid_data = prep_dataloader(self.args)

        self.construct_model()
        
        self.construct_optimizer()

        if len(self.lr_config['scheduler']) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **self.lr_config['scheduler'])
        else:
            scheduler = None

        self.trainer = Trainer(self.model, self.optimizer, scheduler, self.device)

        self.makedirs_or_load()


    def construct_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), **self.lr_config['optimizer'])
        if self.args['SWA']:
            print('SWA training ...')
            SWA_config = getattr(lr_configs, self.args['SWA_config'])
            steps_per_epoch = int(len(self.train_data.dataset) / self.args['batch_size'])
            print("update per epoch:", steps_per_epoch)
            self.swa_start = self.args['epochs'] * SWA_config['SWA_ratio']
            self.optimizer = SWA(self.optimizer, swa_start=self.swa_start * steps_per_epoch,
                            swa_freq=steps_per_epoch, swa_lr=SWA_config['SWA_lr'])
            print(self.optimizer)
        
        elif self.args['LA']:
            print('Lookahead training ...')
            LA_config = getattr(lr_configs, self.args['LA_config'])
            self.optimizer = Lookahead(self.optimizer, **LA_config)
            print(self.optimizer)

    def construct_model(self):
        print('Construct Model ...')

        def load_pretrained():
            if self.pretrained_path is not None:
                sd = torch.load(self.pretrained_path)
                model.load_state_dict(sd)


        self.is_baseline = True

        if self.arch == 'alexnet':
            model = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
        elif self.arch == 'mobilenetv3':
            model = mobilenetv3(n_class=100)
        elif self.arch == 'vgg13':
            model = VGG13(num_class=100)
        elif self.arch == 'vgg19':
            model = VGG19(num_class=100)
        else:
            ResNetClass = ResNet18 if self.arch == 'resnet' else ResNet9
            model = ResNetClass(num_classes=self.num_classes, norm_type=self.norm_type)

        load_pretrained()
        self.model = model.to(self.device)

        pprint(self.model)


    def adjust_learning_rate(self, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def schedule(self, epoch):
        swa_start = 250 * 0.75

        # t = (epoch) / (swa_start if self.args[SWA else args.epochs)
        # lr_ratio = args.SWA_lr / args.learning_rate if args.SWA else 0.01

        t = (epoch) / swa_start
        lr_ratio = 0.05 / 0.1
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return 0.1 * factor



    def training(self):
        best_acc = float('-inf')

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True

        if self.save_interval > 0:
            self.save_model('epoch-0.pth')

        print('Start training ...')

        for ep in range(1, self.epochs + 1):

            # self.adjust_learning_rate(ep)
            # if self.args['SWA']:
            #     self.adjust_learning_rate(self.schedule(ep))

            train_metrics = self.trainer.train(ep, self.train_data)
            
            if self.args['SWA'] and ep >= self.swa_start:
                # Batchnorm update
                self.optimizer.swap_swa_sgd()
                self.optimizer.bn_update(self.train_data, self.model, device='cuda')
                valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')
                self.optimizer.swap_swa_sgd()
            
            elif self.args['LA']:
                self.optimizer._backup_and_load_cache()
                valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')
                self.optimizer._clear_and_load_backup()
            else:
                valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')

           
            metrics = {'epoch': ep}
            for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]

            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')

            if best_acc < metrics['valid_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_acc']
                self.save_model('best.pth')

            self.save_last_model()

    def evaluate(self):
        self.trainer.test(self.valid_data)
