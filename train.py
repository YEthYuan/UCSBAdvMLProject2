import argparse
import logging
import os

import torch

from data_util import cifar10_dataloader
from model_util import ResNet18
from attack_util import *

logger = logging.getLogger(__name__)

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=160, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument(
        "--eps", type=int, default=8, help="Attack budget: epsilon / 255"
    )
    parser.add_argument(
        "--alpha", type=float, default=2, help="PGD attack step size: alpha / 255"
    )
    parser.add_argument(
        "--attack_step", type=int, default=10, help="Number of PGD iterations"
    )
    parser.add_argument(
        "--confidence", type=float, default=0., help="Confidence tau in C&W loss"
    )
    parser.add_argument(
        '--norm', type=str, default='Linf', choices=['Linf', 'L2', 'L1'], help='Norm to use for attack'
    )
    parser.add_argument(
        "--train_method", type=str, default="at", choices=['at', 'fat'],
        help="Adversarial Training or Fast Adversarial Training"
    )
    parser.add_argument(
        '--data_dir', default='./data/', type=str, help="Folder to store downloaded dataset"
    )
    parser.add_argument(
        '--save_dir', default='./out/at/', help='Filepath to the trained model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024, help='Batch size for attack'
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logfile = os.path.join(args.save_dir, 'train.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    logger.info(args)

    train_loader, val_loader, test_loader, dataset_normalization = cifar10_dataloader(args.batch_size, args.data_dir)
    mean, std = dataset_normalization.get_params()

    epsilon = (args.eps / 255.) / std
    alpha = (args.alpha / 255.) / std

    model = ResNet18()
    # be lazy...
    opt = torch.optim.SGD(model.parameters(), args.lr, args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    prev_robust_acc = 0.
    best_pgd_acc = 0
    test_acc_best_pgd = 0
    start_epoch = 0



