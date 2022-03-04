import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset, DataLoader

from utils import SequentialDistributedSampler


class ISICDataset(Dataset):
    def __init__(self, root_path, transform, mode, runtime=1, ratio=100):
        super(ISICDataset, self).__init__()
        self.root = root_path
        self.mode = mode
        if mode == "test":
            label_list = pd.read_csv(os.path.join(root_path, "testing.csv"))
        else:
            label_list = pd.read_csv(
                os.path.join(root_path, f"isic2018_label{ratio}_{runtime}.csv")
            )
            unlabel_list = pd.read_csv(
                os.path.join(root_path, f"isic2018_unlabel{ratio}_{runtime}.csv")
            )
        # label
        self.root = root_path
        self.labeled_imgs = label_list["image"].values
        self.labeled_gr = label_list.iloc[:, 1:-1].values.astype(int)
        self.transform = transform

        # unlabel
        if mode != "test":
            self.unlabeled_imgs = unlabel_list["image"].values
            self.unlabeled_gr = unlabel_list.iloc[:, 1:-1].values.astype(int)

    def x_add_pl(self, pl, idxs):
        self.labeled_imgs = np.concatenate(
            (self.labeled_imgs, self.unlabeled_imgs[idxs])
        )
        self.labeled_gr = np.concatenate((self.labeled_gr, pl))

    def x_update_pl(self, idxs, args):
        self.labeled_imgs = self.labeled_imgs[idxs]
        self.labeled_gr = self.labeled_gr[idxs]

    def u_update_pl(self, idxs):
        self.unlabeled_imgs = np.delete(self.unlabeled_imgs, idxs)
        self.unlabeled_gr = np.delete(self.unlabeled_gr, idxs)

    def __getitem__(self, item):
        if self.mode == "test" or self.mode == "labeled" or self.mode == "anchor":
            img_path = self.labeled_imgs[item]
            img_path = os.path.join(self.root, "train", img_path + ".jpg")
            target = self.labeled_gr[item]
        else:
            img_path = self.unlabeled_imgs[item]
            img_path = os.path.join(self.root, "train", img_path + ".jpg")
            target = self.unlabeled_gr[item]

        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img = self.transform(img)
        return (img, target, item)

    def __len__(self):
        if self.mode == "labeled" or self.mode == "test" or self.mode == "anchor":
            return self.labeled_imgs.shape[0]
        elif self.mode == "unlabeled":
            return self.unlabeled_imgs.shape[0]


class ISICDataloader:
    def __init__(self, batch_size=128, num_workers=9, img_resize=224, root_dir=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize
        self.root_dir = root_dir

    def run(self, mode, dataset=None, transform=None, ratio=100, runtime=1):
        if dataset:
            all_dataset = dataset
        else:
            all_dataset = ISICDataset(
                root_path=self.root_dir,
                transform=transform,
                mode=mode,
                ratio=ratio,
                runtime=runtime,
            )
        sampler = (
            torch.utils.data.distributed.DistributedSampler(all_dataset)
            if mode == "labeled"
            else SequentialDistributedSampler(all_dataset)
        )
        loader = DataLoader(
            dataset=all_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if mode == "labeled" else False,
        )
        return loader, all_dataset, sampler
