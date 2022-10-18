import os, glob
import numpy as np 
import pandas as pd

import torch 
from torch import nn 
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize


class NPF(Dataset):
    def __init__(self, files):
        # self.df = pd.read_csv(fp, parse_dates=[0], index_col=0)
        self.files = files
        # self.days = sorted(np.unique(self.df.index.date.astype(str)).tolist())
        self.lenx = len(self.files)
        # self.vmax = vmax

    def __getitem__(self, index):
        file = self.files[index]

        vals = pd.read_csv(file, parse_dates=[0], index_col=0).values
        vals = vals / np.max(vals)
        # values = df.values / self.vmax

        return torch.from_numpy(vals).unsqueeze(0).float()

    def __len__(self):
        return self.lenx


def build_dataset(args):
   
    files = sorted(glob.glob(f'{args.dataroot}/{args.station}/*.csv'))
    NPF_set = NPF(files)
    NPF_loader = DataLoader(NPF_set, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, drop_last=True, pin_memory=True)
    return NPF_loader


if __name__ == '__main__':
    # build_dataset()
    import argparse
    # for X in loader:
    #     print(X)
    parser = argparse.ArgumentParser('gNPF')

    # datasets
    parser.add_argument('--dataroot', type=str, default='datasets', help='folder that stores the datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=64, help='mini batch size')
    parser.add_argument('--station', type=str, default='hyy', help='station name [var | hyy | kum]')
    # parser.add_argument('--vmax', type=float, default=1e6, help='maximum value for normalization')
    args = parser.parse_args()

    loader = build_dataset(args)
    for X in loader:
        print(X.shape, X.min(), X.max())