"""Tester class"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from . import nets
from .utils.dataset import VisSearchDataset

NUM_WORKERS = 4

# for preprocessing, normalize using values used when training these models on ImageNet for torchvision
# see https://github.com/pytorch/examples/blob/632d385444ae16afe3e4003c94864f9f97dc8541/imagenet/main.py#L197-L198
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Tester:
    """class for measuring accuracy of CNNs on test set after training for visual search task"""
    def __init__(self,
                 net_name,
                 csv_file,
                 restore_path,
                 num_classes=2,
                 batch_size=64,
                 device='cuda',
                 num_workers=NUM_WORKERS,
                 ):
        """create new Tester instance

        Parameters
        ----------
        net_name
        new_learn_rate_layers
        csv_file
        restore_path
        num_classes
        batch_size
        device
        num_workers
        """
        self.net_name = net_name
        if net_name == 'alexnet':
            model = nets.alexnet.build(pretrained=False, progress=False, num_classes=num_classes)
        elif net_name == 'VGG16':
            model = nets.vgg16.build(pretrained=False, progress=False, num_classes=num_classes)

        self.restore_path = restore_path
        model_file = str(restore_path) + '-model.pt'
        model.load_state_dict(
            torch.load(model_file)
        )
        model.to(device)
        self.model = model
        self.device = device

        normalize = transforms.Normalize(mean=MEAN,
                                         std=STD)

        self.testset = VisSearchDataset(csv_file=csv_file,
                                        split='test',
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(), normalize]
                                        ))
        self.test_loader = DataLoader(self.testset, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers,
                                      pin_memory=True)

        self.csv_file = csv_file
        self.dataset_df = pd.read_csv(csv_file)

        self.batch_size = batch_size

    def test(self):
        """method to test trained model

        Returns
        -------
        acc : float
            accuracy on test set
        pred : numpy.ndarray
            predictions for test set
        """
        self.model.eval()

        total = int(np.ceil(len(self.testset) / self.batch_size))
        pbar = tqdm(self.test_loader)
        acc = []
        pred = []
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(pbar):
                pbar.set_description(f'batch {i} of {total}')
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                output = self.model(x_batch)
                # below, _ because torch.max returns (values, indices)
                _, pred_batch = torch.max(output.data, 1)
                acc_batch = (pred_batch == y_batch).sum().item() / y_batch.size(0)
                acc.append(acc_batch)

                pred_batch = pred_batch.cpu().numpy()
                pred.append(pred_batch)

        acc = np.asarray(acc).mean()
        pred = np.concatenate(pred)

        return acc, pred