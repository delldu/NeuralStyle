#! /usr/bin/env python
# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, Thu Sep 20 21:42:14 CST 2018
# ***
# ************************************************************************************/


import argparse
import os

import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np

import model


def sample_int(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class LoopSampler(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(sample_int(self.num_samples))

    def __len__(self):
        return 2**31


class FolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FolderDataset'


def update_learning_rate(optimizer, step):
    lr = args.lr / (1.0 + args.lr_decay * step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_steps(epochs):
    n = int((epochs + 1) / 10)
    if n < 10:
        n = 10

    # round to 10x times
    n = 10 * int((n + 9) / 10)
    return n


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument(
    '-content',
    type=str,
    required=True,
    help='Directory path to a batch of content images')
parser.add_argument(
    '-style',
    type=str,
    required=True,
    help='Directory path to a batch of style images')
# Model options
parser.add_argument(
    '-encoder',
    type=str,
    default='models/encoder.pth',
    help='Pre-trained encoder model, default: models/encoder.pth')
parser.add_argument(
    '-decoder',
    type=str,
    default='models/decoder.pth',
    help='Pre-trained decoder model, default: models/decoder.pth')

# Training options
parser.add_argument(
    '-save_dir',
    default='logs',
    help='Directory to save the model, default: logs')
parser.add_argument(
    '-log_dir',
    default='logs',
    help='Directory to save the log, default: logs')
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-lr_decay', type=float, default=5e-5)
parser.add_argument(
    '-epochs',
    type=int,
    default=1000,
    help='epochs for training. default: 1000')
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-style_weight', type=float, default=10.0)
parser.add_argument('-content_weight', type=float, default=1.0)
parser.add_argument('-n_threads', type=int, default=4)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    # writer = SummaryWriter(log_dir=args.log_dir)

    encoder = model.encoder_load(args.encoder)
    decoder = model.decoder_load(args.decoder)

    network = model.StyleNet(encoder, decoder)
    network.train()
    network.to(device)

    T = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])
    c_dataset = FolderDataset(args.content, T)
    s_dataset = FolderDataset(args.style, T)

    c_iter = iter(
        data.DataLoader(
            c_dataset,
            batch_size=args.batch_size,
            sampler=LoopSampler(c_dataset),
            num_workers=args.n_threads))
    s_iter = iter(
        data.DataLoader(
            s_dataset,
            batch_size=args.batch_size,
            sampler=LoopSampler(s_dataset),
            num_workers=args.n_threads))

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    save_interval = save_steps(args.epochs)

    for i in tqdm(range(args.epochs)):
        update_learning_rate(optimizer, i)
        c_images = next(c_iter).to(device)
        s_images = next(s_iter).to(device)
        loss_c, loss_s = network(c_images, s_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # writer.add_scalar('loss_content', loss_c.item(), i + 1)
        # writer.add_scalar('loss_style', loss_s.item(), i + 1)

        if (i + 1) % save_interval == 0 or (i + 1) == args.epochs:
            state_dict = model.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, '{:s}/decoder_iter_{:d}.pth.tar'.format(
                args.save_dir, i + 1))
