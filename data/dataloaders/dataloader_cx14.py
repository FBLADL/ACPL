import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import ImageFilter
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from utils import DistributedEvalSampler

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
        if len(transform) > 1:
            self.strong_aug, self.weak_aug = transform
        else:
            self.weak_aug = transform[0]
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
        # gr = np.asarray([gr[i] for i in all_names])

        labeled_gr = np.asarray([gr[i] for i in self.labeled_imgs])
        # self.unlabeled_gr = np.asarray([gr[i] for i in self.unlabeled_imgs])

        self.labeled_gr = np.zeros((labeled_gr.shape[0], 15))
        for idx, i in enumerate(labeled_gr):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
            self.labeled_gr[idx] = binary_result
        # if mode == "unlabeled":
        self.all_imgs = np.asarray([x for x in all_names])
        self.unlabeled_imgs = np.setdiff1d(self.all_imgs, self.labeled_imgs)
        unlabeled_gr = np.asarray([gr[i] for i in self.unlabeled_imgs])
        self.unlabeled_gr = np.zeros((unlabeled_gr.shape[0], 15))
        for idx, i in enumerate(unlabeled_gr):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
            self.unlabeled_gr[idx] = binary_result

    def update(self, pl, idxs=None, loop=1):
        if loop == 1:
            self.labeled_imgs = np.concatenate(
                (self.labeled_imgs, self.unlabeled_imgs[idxs])
            )
            # self.labeled_imgs.append(self.unlabeled_imgs[idxs])
            self.labeled_gr = np.concatenate((self.labeled_gr, pl))
        else:
            self.labeled_imgs[-idxs.shape[0] :] = self.unlabeled_imgs[idxs]
            self.labeled_gr[-pl.shape[0] :, :] = pl
        # self.unlabeled_imgs.delete(idxs)
        # print(self.labeled_imgs.shape[0])
        # print(self.labeled_gr.shape[0])

    def __getitem__(self, item):
        if self.mode == "labeled":
            img = io.imread(
                os.path.join(self.root_dir, "data", self.labeled_imgs[item])
            )
            img = Image.fromarray(img).convert("RGB")
            img_s = self.strong_aug(img)
            img_w = self.weak_aug(img)
            # img2 = self.transform(img)
            target = self.labeled_gr[item]
            return img_s, img_w, target, item
        elif self.mode == "unlabeled":
            img = io.imread(
                os.path.join(self.root_dir, "data", self.unlabeled_imgs[item])
            )
            img = Image.fromarray(img).convert("RGB")
            img_s = self.strong_aug(img)
            img_w = self.weak_aug(img)
            target = self.unlabeled_gr[item]
            # img2 = self.transform(img)
            return img_s, img_w, target, item
        elif self.mode == "test":
            img = io.imread(
                os.path.join(self.root_dir, "data", self.labeled_imgs[item])
            )
            img = Image.fromarray(img).convert("RGB")
            img_w = self.weak_aug(img)
            target = self.labeled_gr[item]
            return img_w, target, item

    def __len__(self):
        if self.mode == "labeled" or self.mode == "test":
            return self.labeled_imgs.shape[0]
        elif self.mode == "unlabeled":
            return self.unlabeled_imgs.shape[0]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ChestDataloader:
    def __init__(
        self,
        batch_size=128,
        num_workers=8,
        img_resize=512,
        root_dir=None,
        gc_cloud=False,
        imagenet=False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize
        if imagenet:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.508135, 0.508135, 0.508135]
            std = [0.252920, 0.252920, 0.252920]

        # print("Use GCloud: {} ".format(gc_cloud))
        if gc_cloud:
            download_chestxray_unzip(root_dir)

        self.root_dir = root_dir

        self.strong_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_resize, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8
                ),
                transforms.RandomRotation(45),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                # transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                # transforms.RandomErasing(inplace=True)
            ]
        )

        self.weak_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_resize, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Resize(img_resize + 32 * 2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def run(self, mode, dataset=None, exp=None, ratio=100, runtime=1):
        use_transform = (
            [self.strong_aug, self.weak_aug]
            if mode == "labeled" or mode == "unlabeled"
            else [self.transform_test]
        )

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
            if mode == "unlabeled"
            else self.batch_size
        )
        sampler = (
            torch.utils.data.distributed.DistributedSampler(all_dataset)
            if mode == "labeled"
            else DistributedEvalSampler(all_dataset, shuffle=False)
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
