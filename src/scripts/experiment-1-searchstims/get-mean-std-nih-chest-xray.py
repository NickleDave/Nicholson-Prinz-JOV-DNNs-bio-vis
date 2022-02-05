#!/usr/bin/env python
# coding: utf-8
"""compute channel-wise mean and standard deviation of 
NIH Chest X-ray dataset 
(Kaggle version, converted to .png files)
https://www.kaggle.com/nih-chest-xrays/data
"""
import json
import os

import pyprojroot
import torch
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm


data_dir = '~/Documents/data/nih-chest-xray/'
traindir = os.path.join(data_dir, 'train')

transform = transforms.ToTensor()

dataset = datasets.ImageFolder(root=traindir, transform=transform)
dataloader = data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

mean = torch.zeros(3)
std = torch.zeros(3)

for i, data in enumerate(tqdm(dataloader)):
    data = data[0].squeeze(0)
    if (i == 0): size = data.size(1) * data.size(2)
    mean += data.sum((1, 2)) / size

mean /= len(dataloader)
mean = mean.unsqueeze(1).unsqueeze(2)

for i, data in enumerate(tqdm(dataloader)):
    data = data[0].squeeze(0)
    std += ((data - mean) ** 2).sum((1, 2)) / size

std /= len(dataloader)
std = std.sqrt()


mean_std = {
    'means': ','.join([str(v) for v in mean.cpu().numpy().flatten().tolist()]),
    'stds': ','.join([str(v) for v in std.cpu().numpy().tolist()]),
}
with open(pyprojroot.here() / 'data' / 'nih-chest-xray-mean-std.json', 'w') as fp:
    json.dump(mean_std, fp)
