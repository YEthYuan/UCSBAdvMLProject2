import argparse
import logging
import os
import time

import torch
from tqdm import tqdm

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
    upper_limit = ((1 - mean) / std)
    lower_limit = ((0 - mean) / std)
    delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    delta.requires_grad = True

    model = ResNet18()
    # be lazy...
    opt = torch.optim.SGD(model.parameters(), args.lr, args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    prev_robust_acc = 0.
    best_pgd_acc = 0
    test_acc_best_pgd = 0

    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "adv_val_loss": [],
        "adv_val_acc": []
    }

    logger.info("Start training now!")
    start_train_time = time.time()

    for epoch in range(0, args.epochs):
        logger.info(f"Epoch {epoch} starts ...")
        logger.info("Training ...")
        model.train()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(tqdm(train_loader)):
            X = X.cuda()
            y = y.cuda()
            for j in range(len(epsilon)):
                delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
            delta.data = torch.clamp(delta, lower_limit - X, upper_limit - X)

            if args.train_method == 'fat':
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = torch.clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)

            elif args.train_method == 'at':
                for _ in range(args.attack_iters):
                    output = model(X + delta)
                    loss = criterion(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = torch.clamp(delta, lower_limit - X, upper_limit - X)
                    delta.grad.zero_()

            delta = delta.detach()

            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()

            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

        results['train_loss'].append(train_loss / train_n)
        results['train_acc'].append(train_acc / train_n)
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Train Loss: {train_loss / train_n}")
        logger.info(f"Train Acc: {train_acc / train_n}")

        logger.info("Evaluating the standard accuracy ...")
        val_loss = 0
        val_acc = 0
        val_n = 0
        model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(tqdm(val_loader)):
                X, y = X.cuda(), y.cuda()
                output = model(X)
                loss = F.cross_entropy(output, y)
                val_loss += loss.item() * y.size(0)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_n += y.size(0)
