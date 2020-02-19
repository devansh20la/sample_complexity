import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import random
from torch.utils.tensorboard import SummaryWriter
import utils.util_funcs as uf
import time
import logging
import numpy as np
from args import get_args
import train_funcs as tf
import logging
import glob
import os
from models import bottleneck


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


class bottle_model(nn.Module):

    def __init__(self, args):
        super(bottle_model, self).__init__()
        self.model = tf.get_model(args)

        state = torch.load(args.load_cp + "best_model.pth.tar", map_location='cpu')
        self.model.load_state_dict(state['model'])

        if 'vgg' in args.mtype:
            classifier = list(self.model.classifier)
            classifier.pop()
            classifier.pop()
            self.model.classifier = nn.Sequential(*classifier)

        elif 'udacity' in args.mtype:
            regressor = list(self.model.regressor)
            regressor.pop()
            self.model.regressor = nn.Sequential(*regressor)

        self.bottleneck = bottleneck(args.feature_dim, args.dim, args.feature_dim, args.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.bottleneck(x)

        return x

def main(args):
    logger = logging.getLogger('my_log')

    dset_loaders = tf.get_loader(args, training=True)

    model = bottle_model(args)

    if args.use_cuda:
        model = model.cuda()

    if "udacity" in args.dtype:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if args.use_cuda:
        torch.backends.cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.m, weight_decay=args.wd)


    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.milestones,
                                               gamma=0.1)

    writer = SummaryWriter(log_dir=args.cp_dir)


    start_epoch = 0
    best_err = float('inf')

    for epoch in range(start_epoch, args.ep):

        logger.info('Epoch: [%d | %d]' % (epoch, args.ep))

        if "udacity" in args.dtype:
            trainloss, trainerr1, trainerr5 = tf.reg_model_run('train', dset_loaders,
                                         model, criterion, optimizer,
                                         args)
        else:
            trainloss, trainerr1, trainerr5 = tf.class_model_run('train', dset_loaders,
                                                     model, criterion, optimizer,
                                                     args)

        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(trainloss, trainerr1))
        writer.add_scalar('Train/Train_Loss', trainloss, epoch)
        writer.add_scalar('Train/Train_Err1', trainerr1, epoch)
        writer.add_scalar('Train/Train_Err5', trainerr5, epoch)

        if "udacity" in args.dtype:
            valloss, valerr1, valerr5 = tf.reg_model_run('val', dset_loaders, model,
                                     criterion, optimizer, args)
        else:
            valloss, valerr1, valerr5 = tf.class_model_run('val', dset_loaders, model,
                                                criterion, optimizer, args)

        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(valloss, valerr1))
        writer.add_scalar('Val/Val_Loss', valloss, epoch)
        writer.add_scalar('Val/Val_Err1', valerr1, epoch)
        writer.add_scalar('Val/Val_Err5', valerr5, epoch)

        scheduler.step()

        is_best = valerr1 < best_err

        if epoch % 50 == 0:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_err': best_err
            }

            torch.save(state,
                       os.path.join(args.cp_dir,
                                    'train_model_ep{}.pth.tar'.format(epoch)))
        if is_best:
            best_err = min(valerr1, best_err)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_err': best_err
            }

            logger.info("New Best Model Found")
            logger.info("Best Loss:{0:2f}".format(best_err))

            torch.save(state,
                       os.path.join(args.cp_dir, 'best_model.pth.tar'))
            is_best = False


if __name__ == '__main__':
    args = get_args()

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)

    # Intialize directory and create path
    list_of_files = sorted(glob.glob1(args.cp_dir, '*run*'))

    if len(list_of_files) == 0:
        list_of_files = 0
    else:
        list_of_files = list_of_files[-1]
        list_of_files = int(list_of_files[3:]) + 1

    args.cp_dir = os.path.join(args.cp_dir,
                                       'run{0}'.format(list_of_files))
    create_path(args.cp_dir)

    # Logging tools
    logger = logging.getLogger('my_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(
        args.cp_dir, time.strftime("%Y%m%d-%H%M%S") + '.log'))
    logger.addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info(args)

    if "resnet18" in args.mtype:
        args.feature_dim = 512

    elif "vgg16" in args.mtype:
        args.feature_dim = 4096

    elif "lenet" in args.mtype:
        args.feature_dim = 84
        
    elif "resnet50" in args.mtype:
        args.feature_dim = 2048
    
    elif "udacity_cnn" in args.mtype:
        args.feature_dim = 1024
    else:
        quit()

    main(args)
