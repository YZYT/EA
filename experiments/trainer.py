import time

import torch
import torch.nn.functional as F

from models.losses.sign_loss import SignLoss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


class Tester(object):
    def __init__(self, model, device, verbose=True):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        self.model = model
        self.device = device
        self.verbose = verbose

    def test(self, dataloader, msg='Testing Result', compare=[]):
        self.model.eval()
        loss_meter = 0
        acc_meter = 0
        runcount = 0

        start_time = time.time()
        with torch.no_grad():
            for i, load in enumerate(dataloader):
                data, target = load[:2]
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                pred = self.model(data)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item()  # sum up batch loss
                pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                compare.append((pred, target))
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0)
                if self.verbose:
                    print(f'{msg} [{i + 1}/{len(dataloader)}]: '
                          f'Loss: {loss_meter / runcount:6.4f} '
                          f'Acc: {100 * acc_meter / runcount:6.2f} ({time.time() - start_time:.2f}s)', end='\r')

        loss_meter /= runcount
        acc_meter = 100 * acc_meter / runcount

        if self.verbose:
            print(f'{msg}: '
                  f'Loss: {loss_meter:6.4f} '
                  f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
            print()

        return {'loss': loss_meter, 'acc': acc_meter, 'time': time.time() - start_time}


class Trainer(object):
    def __init__(self, model, optimizer, scheduler, device):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, e, dataloader):
        self.model.train()
        loss_meter = 0
        acc_meter = 0

        start_time = time.time()
        for i, (data, target) in enumerate(dataloader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            pred = self.model(data)
            loss = F.cross_entropy(pred, target)

            loss.backward()
            self.optimizer.step()

            loss_meter += loss.item()
            acc_meter += accuracy(pred, target)[0].item()

            print(f'Epoch {e:3d} [{i:4d}/{len(dataloader):4d}] '
                  f'Loss: {loss_meter / (i + 1):6.4f} '
                  f'Acc: {acc_meter / (i + 1):.4f} ({time.time() - start_time:.2f}s)', end='\r')

        print()

        loss_meter /= len(dataloader)
        acc_meter /= len(dataloader)

        if self.scheduler is not None:
            self.scheduler.step()

        return {'loss': loss_meter,
                'acc': acc_meter,
                'time': time.time() - start_time}


    def test(self, dataloader, msg='Testing Result'):
        self.model.eval()
        loss_meter = 0
        acc_meter = 0
        runcount = 0

        start_time = time.time()
        with torch.no_grad():
            for i, load in enumerate(dataloader):
                data, target = load[:2]
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                pred = self.model(data)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
                pred = pred.max(1, keepdim=True)[1]

                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0)
                print(f'{msg} [{i + 1}/{len(dataloader)}]: '
                      f'Loss: {loss_meter / runcount:6.4f} '
                      f'Acc: {acc_meter / runcount:6.2f} ({time.time() - start_time:.2f}s)', end='\r')

        loss_meter /= runcount
        acc_meter = 100 * acc_meter / runcount
        print(f'{msg}: '
              f'Loss: {loss_meter:6.4f} '
              f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
        print()

        return {'loss': loss_meter,
                'acc': acc_meter,
                'time': time.time() - start_time}