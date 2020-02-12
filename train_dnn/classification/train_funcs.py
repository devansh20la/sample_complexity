import torch
from torchvision import transforms
from models import ResNet18, resnet50, vgg16, udacity_cnn, fcnet, LeNet
import utils.util_funcs as uf
import torchvision
from utils import *
import logging
import time
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
import PIL 

def get_model(args):
    """ Function to load model based on the args
    """
    if 'cifar' in args.mtype:
        if args.mtype == 'cifar_resnet18':
            model = ResNet18(num_classes=args.num_classes, filters=args.filters)

        elif args.mtype == 'cifar_resnet50':
            model = ResNet50(num_classes=args.num_classes)

        elif args.mtype == 'cifar_vgg16':
            model = vgg16(filters=args.filters, num_classes=args.num_classes)

    elif 'imagenet' in args.mtype:
        if args.mtype == 'imagenet_resnet50':
            model = resnet50(num_classes=args.num_classes,
                             pretrained=False)

    elif args.mtype == 'udacity':
        model = udacity_cnn()

    elif args.mtype == 'fcnet':
        if args.model_depth == 1:
            logger.info("Model depth is 1, width has no purpose here")
        model = fcnet(args)

    elif args.mtype == 'lenet':
        if len(args.filters) > 2:
            print("length of filters in long")
        model = LeNet(args.filters)


    return model

def get_loader(args, training):
    """ function to get data loader specific to different datasets
    """
    if args.dtype == 'cifar10':
        dsets = cifar10_dsets(args, training)

    elif args.dtype == 'cifar100':
        dsets = cifar100_dsets(args, training)

    elif args.dtype == 'imagenet':
        dsets = imagenet_dsets(args, training)
    
    elif args.dtype == 'mnist':
        dsets = mnist_dsets(args, training)

    elif args.dtype == "udacity":
        dsets = udacity_dsets(args, training)

    if training is True:
        dset_loaders = {
            'train': torch.utils.data.DataLoader(dsets['train'], batch_size=args.bs,
                                                 shuffle=True, pin_memory=True,
                                                 num_workers=24),
            'val': torch.utils.data.DataLoader(dsets['val'], batch_size=128,
                                               shuffle=False, pin_memory=True,
                                               num_workers=24)
        }

    else:
        dset_loaders = {
            'test': torch.utils.data.DataLoader(dsets['test'], batch_size=128,
                                                shuffle=False, pin_memory=True,
                                                num_workers=24)
        }

    return dset_loaders

def class_model_run(phase, loader, model, criterion, optimizer, args):
    """
        Function to forward pass through classification problem
    """
    logger = logging.getLogger('my_log')

    if phase == 'train':
        model.train()
    else:
        model.eval()

    loss = uf.AverageMeter()
    err1 = uf.AverageMeter()
    err5 = uf.AverageMeter()
    t = time.time()

    for batch_idx, inp_data in enumerate(loader[phase],1):

        inputs = inp_data['img']
        targets = inp_data['target']

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if phase == 'train':
            with torch.set_grad_enabled(True):
                # compute output
                outputs = model(inputs)
                # print(outputs.size(), targets.size())
                batch_loss = criterion(outputs, targets)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        elif phase == 'val':
            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
        else:
            logger.info('Define correct phase')
            quit()

        loss.update(batch_loss.item(), inputs.size(0))
        batch_err = uf.accuracy(outputs, targets, topk=(1,5))
        err1.update(float(100.0 - batch_err[0]), inputs.size(0))
        err5.update(float(100.0 - batch_err[1]), inputs.size(0))

        if batch_idx % args.print_freq == 0:
            logger.info("Phase:{0} -- Batch_idx:{1}/{2} -- {3:.2f} samples/sec"
                        "-- Loss:{4:.2f} -- Error1:{5:.2f}".format(
                          phase, batch_idx, len(loader[phase]),
                          err1.count / (time.time() - t), loss.avg, err1.avg))

    return loss.avg, err1.avg, err5.avg

def reg_model_run(phase, loader, model, criterion, optimizer, args):

    logger = logging.getLogger('my_log')

    if phase == 'train':
        model.train()
    else:
        model.eval()

    loss = uf.AverageMeter()
    err = uf.AverageMeter()
    t = time.time()

    for batch_idx, inp_data in enumerate(loader[phase], 1):

        inputs = inp_data['img']
        targets = inp_data['target']

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if phase == 'train':
            with torch.set_grad_enabled(True):
                # compute output
                outputs = model(inputs)
                # print(outputs.size(), targets.size())
                batch_loss = criterion(outputs, targets)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        elif phase == 'val':
            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
        else:
            logger.info('Define correct phase')
            quit()

        loss.update(batch_loss.item(), inputs.size(0))
        batch_err = torch.abs(outputs - targets)
        batch_err = torch.sum(batch_err > (np.pi / 8)).type(torch.FloatTensor) /inputs.size(0)
        err.update(float(batch_err), inputs.size(0))

        if batch_idx % args.print_freq == 0:
            logger.info("Phase:{0} -- Batch_idx:{1}/{2} -- {3:.2f} samples/sec"
                        "-- Loss:{4:.2f} -- Error:{5:.2f}".format(
                            phase, batch_idx, len(loader[phase]),
                            err.count / (time.time() - t), loss.avg, err.avg))

    return loss.avg, err.avg

def mnist_dsets(args, training):
    """ Function to load mnist data
    """
    if training is True:
        dsets = {
            'train': MNIST(args.data_dir, train=True, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))]),
                           lp=args.train_size),
            'val': MNIST(args.data_dir, train=False, download=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))]),
                         lp=1.0)
        }
    else:
        dsets = {
            'test': MNIST(args.data_dir, train=False, download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))]))
        }

    return dsets

def cifar10_dsets(args, training):
    """ Function to load cifar10 data
    """
    transform = {
        'train': transforms.Compose([transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))]),
        'val': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))])
    }
    if training is True:
        dsets = {
            'train': CIFAR10(root=args.data_dir,
                             load_percent=args.train_size, train=True,
                             download=False, transform=transform['train']),
            'val': CIFAR10(root=args.data_dir, load_percent=1, train=False,
                           download=False, transform=transform['val'])
        }
    else:
        dsets = {
        'test': CIFAR10(root=args.data_dir, load_percent=1.0,
                        train=False, download=False,
                        transform=transform['val'])
        }

    return dsets

def cifar100_dsets(args, training):
    """ Function to load cifar100 data
    """
    transform = {
        'train': transforms.Compose([transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))]),
        'val': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))])
    }
    if training is True:
        dsets = {
            'train': CIFAR100(root=args.data_dir,
                              load_percent=args.train_size, train=True,
                              download=False, transform=transform['train']),
            'val': CIFAR100(root=args.data_dir,
                            load_percent=1.0, train=False, download=False,
                            transform=transform['val'])
        }
    else:
        dsets = {
            'test': CIFAR100(root=args.data_dir, load_percent=1.0,
                             train=False, download=False,
                             transform=transform['val'])
        }
        
    return dsets

def imagenet_dsets(args, training):
    """ Function to load imagenet data
    """
    transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])])
    }

    if training is True:
        dsets = {
            'train': ImageNet(root=args.data_dir, split='train',
                              images_per_class=args.img_per_class,
                              download=False,
                              transform=transform['train']),
            'val': ImageNet(root=args.data_dir, split='val',
                            download=False, transform=transform['val'])
        }

    else:
        dsets = {
            'test': ImageNet(root=args.data_dir, split='val',
                             download=False, transform=transform['val'])
        }

    return dsets

def udacity_dsets(args, training):
    transform = {
        'train': [transforms.Resize((128,128), interpolation=PIL.Image.ANTIALIAS),
                  uf.RandomHorizontalFlip(),
                  transforms.ToTensor()
        ],
        
        'val': [transforms.Resize((128,128), interpolation=PIL.Image.ANTIALIAS),
            transforms.ToTensor(),
        ]
    }

    if training is True:
        dsets = {
            'train': udacity(args.data_dir, args.train_size, 'train', transform=transform['train']),
            'val': udacity(args.data_dir, 1.0, 'val', transform=transform['val'])
        }
    else:
        dsets = {
            'test': udacity(args.data_dir, 1.0, 'test', transform=transform['val'])
        }

    return dsets

# elif 'nvidia' in args.dtype:
#         transform = {
#             'train': transforms.Compose([transforms.Resize((32, 32),
#                                          interpolation=PIL.Image.ANTIALIAS),
#                                          transforms.ToTensor()]),
#             'val': transforms.Compose([transforms.Resize((32, 32),
#                                        interpolation=PIL.Image.ANTIALIAS),
#                                        transforms.ToTensor()])
#         }

#         samples, class_vector = make_dataset(root=args.data_dir)
#         samples_train, samples_test, class_vector_train, \
#             class_vector_test = train_test_split(samples, class_vector,
#                                                  test_size=0.20,
#                                                  random_state=12,
#                                                  stratify=class_vector)

#         if training is True:
#             samples_train, samples_val, class_vector_train, \
#                 class_vector_val = train_test_split(samples_train,
#                                                     class_vector_train,
#                                                     test_size=0.20,
#                                                     random_state=12,
#                                                     stratify=class_vector_train)

#             if args.data_size < 1.0:
#                 samples_train, _, class_vector_train, _ \
#                     = train_test_split(samples_train, class_vector_train,
#                                        test_size=1 - args.data_size,
#                                        random_state=12,
#                                        stratify=class_vector_train)
#                 dsets = {
#                     'train': SyntheticDataset(samples=samples_train,
#                                               transform=transform['train']),
#                     'val': SyntheticDataset(samples=samples_val,
#                                             transform=transform['val'])
#                 }
#                 sampler = {
#                     'train': StratifiedSampler(torch.Tensor(class_vector_train), args.bs),
#                     'val': StratifiedSampler(torch.Tensor(class_vector_val), 64)
#                 }
#         else:
#             dsets = SyntheticDataset(samples=samples_test, transform=transform['val'])

#     elif args.dtype == 'synthetic':
#         normalize = transforms.Normalize(mean=[0.5], std=[0.5])

#         transform = {
#             'train': transforms.Compose([normalize]),
#             'val': transforms.Compose([normalize])
#         }
#         data = torch.load(args.data_dir)
#         X = data['X']
#         Y = data['Y']

#         X_train, X_test, \
#             Y_train, Y_test = train_test_split(np.array(X), np.array(Y),
#                                                test_size=0.20, random_state=12)

#         if args.train_size < 1.0:
#             X_train, _, \
#                 Y_train, _ = train_test_split(X_train, Y_train,
#                                               test_size=1.0 - args.train_size,
#                                               random_state=12)

#         if training is True:
#             dsets = {
#                 'train': SyntheticDataset(samples=(X_train, Y_train),
#                                           transform=None),

#                 'val': SyntheticDataset(samples=(X_test, Y_test),
#                                         transform=None)
#             }
#         else:
#             dsets = {
#                 'test': SyntheticDataset(samples=(X_test, Y_test),
#                                          transform=None)
#             }
