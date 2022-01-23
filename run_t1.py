import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models

from resnet18_torchvision import resnet18
from cub_first100 import CUB
import tqdm
save_path = './results/cub_res18torch_inc_t1_fsll_ss/'

os.makedirs(save_path,exist_ok=True)

def apply_2d_rotation(input_tensor1, rotation):

    assert input_tensor1.dim() >= 2
    input_tensor = input_tensor1.clone()

    height_dim = input_tensor.dim() - 2
    width_dim = height_dim + 1

    flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
    flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
    spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)

    if rotation == 0:  # 0 degrees rotation
        return input_tensor
    elif rotation == 90:  # 90 degrees rotation
        return flip_upside_down(spatial_transpose(input_tensor))
    elif rotation == 180:  # 90 degrees rotation
        return flip_left_right(flip_upside_down(input_tensor))
    elif rotation == 270:  # 270 degrees rotation / or -90
        return spatial_transpose(flip_upside_down(input_tensor))
    else:
        raise ValueError(
            "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
        )





def create_4rotations_images(images, stack_dim=None):
    """Rotates each image in the batch by 0, 90, 180, and 270 degrees."""
    images_4rot = []
    for r in range(4):
        images_4rot.append(apply_2d_rotation(images, rotation=r * 90))

    if stack_dim is None:
        images_4rot = torch.cat(images_4rot, dim=0)
    else:
        images_4rot = torch.stack(images_4rot, dim=stack_dim)

    return images_4rot



def create_rotations_labels(batch_size, device):
    """Creates the rotation labels."""
    labels_rot = torch.arange(4, device=device).view(4, 1)

    labels_rot = labels_rot.repeat(1, batch_size).view(-1)
    return labels_rot


parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--gpu', dest='gpu', default='0', type=str)
best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_pre = models.resnet18(pretrained=True)
    model = resnet18()
    model.load_state_dict(model_pre.state_dict())

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)

    model.rotm = nn.Sequential(nn.Conv2d(512,512,3,1,1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),nn.Conv2d(512,512,3,1,1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),nn.Conv2d(512,512,3,1,1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),nn.Conv2d(512,512,3,1,1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1))

    model.rotm_avgpool = nn.AdaptiveAvgPool2d(1)
    model.rotm_class = nn.Linear(512,4)


    model = model.cuda()


    
    for p in model.parameters():
        p.requires_grad = True  

    cudnn.benchmark = True
    

    trainset_t1 = CUB(train=True)
    testset_t1 = CUB(train=False)


    train_loader = torch.utils.data.DataLoader(
        trainset_t1,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        testset_t1,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()   

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)



    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30,40], gamma=0.1, last_epoch=args.start_epoch - 1)

 
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    with torch.no_grad():
        best = validate(val_loader, model, criterion)

    for epoch in tqdm.tqdm(range(args.start_epoch, args.epochs)):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        with torch.no_grad():
            prec1 = validate(val_loader, model, criterion)


        fname = 'epoch'+str(epoch)+'_cub_res18_t1_'+str(prec1)+'.pth'
        torch.save(model.state_dict(), save_path+fname)



        


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_rot = AverageMeter()
    top1 = AverageMeter()
    top1_rot = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in tqdm.tqdm(enumerate(train_loader)):
        origtarget = target.cuda()
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        rot_img = create_4rotations_images(input)  
        labels_rotation = create_rotations_labels(len(input),input.device)
        labels_rotation = labels_rotation.cuda()
        target = target.repeat(4)    
        target = target.cuda()

        ltrot,logits_fromrot = model(rot_img, is_feat=True)
        logits_rot=model.rotation(ltrot[3])

        output = model(input)

        cls_loss = criterion(output, origtarget)
        rot_loss = criterion(logits_rot,labels_rotation)   
        loss = 1*cls_loss + 1*rot_loss    



        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


        loss = loss.float()
        cls_loss = cls_loss.float()
        rot_loss = rot_loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, origtarget)
        prec1_rot = accuracy(logits_rot.data, labels_rotation)


        losses.update(loss.item(), rot_img.size(0))
        losses_cls.update(cls_loss.item(), input.size(0))
        losses_rot.update(rot_loss.item(), rot_img.size(0))
        top1.update(prec1[0], input.size(0))
        top1_rot.update(prec1_rot[0], rot_img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_CLS {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Loss_ROT {rot_loss.val:.4f} ({rot_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@1 Rot {top1_rot.val:.3f} ({top1_rot.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses,cls_loss=losses_cls,rot_loss=losses_rot, top1=top1, top1_rot=top1_rot))



def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

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
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
