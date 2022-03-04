import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader

from utils import SequentialDistributedSampler

# from utils.gcloud import download_chestxray_unzip

Labels = {
    "No Finding": 14,
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
}
mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


class ChestDataset(Dataset):
    def __init__(self, root_dir, transform, mode, runtime=1, ratio=100) -> None:
        self.transform = transform
        self.root_dir = root_dir
        self.mode = mode

        gr_path = os.path.join(root_dir, "Data_Entry_2017.csv")
        gr = pd.read_csv(gr_path, index_col=0)
        gr = gr.to_dict()["Finding Labels"]

        img_list = os.path.join(
            root_dir,
            "test_list.txt"
            if mode == "test"
            else "train_val_list_{}_{}.txt".format(ratio, runtime),
        )
        with open(img_list) as f:
            names = f.read().splitlines()
        self.labeled_imgs = np.asarray([x for x in names])

        all_img_list = os.path.join(root_dir, "train_val_list.txt")
        with open(all_img_list) as f:
            all_names = f.read().splitlines()

        labeled_gr = np.asarray([gr[i] for i in self.labeled_imgs])

        self.labeled_gr = np.zeros((labeled_gr.shape[0], 15))
        for idx, i in enumerate(labeled_gr):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
            self.labeled_gr[idx] = binary_result
        self.all_imgs = np.asarray([x for x in all_names])
        self.unlabeled_imgs = np.setdiff1d(self.all_imgs, self.labeled_imgs)
        unlabeled_gr = np.asarray([gr[i] for i in self.unlabeled_imgs])
        self.unlabeled_gr = np.zeros((unlabeled_gr.shape[0], 15))
        for idx, i in enumerate(unlabeled_gr):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
            self.unlabeled_gr[idx] = binary_result

    def x_add_pl(self, pl, idxs):
        self.labeled_imgs = np.concatenate(
            (self.labeled_imgs, self.unlabeled_imgs[idxs])
        )
        self.labeled_gr = np.concatenate((self.labeled_gr, pl))

    def x_update_pl(self, idxs, args):
        print(self.labeled_imgs.shape)
        self.labeled_imgs = self.labeled_imgs[idxs]
        self.labeled_gr = self.labeled_gr[idxs]
        # mask = self.labeled_gr > (
        #     np.amax(self.labeled_gr, axis=1) - args.max_interval
        # ).reshape(-1, 1)
        # self.labeled_gr[mask] = 1
        # self.labeled_gr[~mask] = 0
        print(self.labeled_imgs.shape)

    def u_update_pl(self, idxs):
        self.unlabeled_imgs = np.delete(self.unlabeled_imgs, idxs)
        self.unlabeled_gr = np.delete(self.unlabeled_gr, idxs, axis=0)

    def __getitem__(self, item):

        if self.mode == "labeled" or self.mode == "anchor":
            img_path = os.path.join(self.root_dir, "data", self.labeled_imgs[item])
            input_path = float(self.labeled_imgs[item].split(".")[0].replace("_", "."))
            target = self.labeled_gr[item]
        elif self.mode == "unlabeled":
            img_path = os.path.join(self.root_dir, "data", self.unlabeled_imgs[item])
            input_path = float(
                self.unlabeled_imgs[item].split(".")[0].replace("_", ".")
            )
            target = self.unlabeled_gr[item]
        elif self.mode == "test":
            img_path = os.path.join(self.root_dir, "data", self.labeled_imgs[item])
            input_path = float(self.labeled_imgs[item].split(".")[0].replace("_", "."))
            target = self.labeled_gr[item]

        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img_w = self.transform(img)
        return (img_w, target, item, input_path)

    def __len__(self):
        if self.mode == "labeled" or self.mode == "test" or self.mode == "anchor":
            return self.labeled_imgs.shape[0]
        elif self.mode == "unlabeled":
            return self.unlabeled_imgs.shape[0]


class ChestDataloader:
    def __init__(
        self,
        batch_size=128,
        num_workers=8,
        img_resize=512,
        root_dir=None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize

        self.root_dir = root_dir

    def run(self, mode, dataset=None, transform=None, exp=None, ratio=100, runtime=1):
        use_transform = transform

        if dataset:
            all_dataset = dataset
        else:
            all_dataset = ChestDataset(
                root_dir=self.root_dir,
                transform=use_transform,
                mode=mode,
                ratio=ratio,
                runtime=runtime,
            )
        batch_size = (
            (self.batch_size * 2)
            if mode == "test"
            else (self.batch_size * 3)
            if mode == "unlabeled" or mode == "anchor"
            else self.batch_size
        )
        sampler = (
            torch.utils.data.distributed.DistributedSampler(all_dataset)
            if mode == "labeled"
            else SequentialDistributedSampler(all_dataset)
        )
        loader = DataLoader(
            dataset=all_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if mode == "labeled" else False,
        )

        return loader, all_dataset, sampler

    # def update(self,pl):
