from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from skimage import io
from PIL import Image
from torchvision.transforms import transforms
import random
from PIL import ImageFilter
import torch
from utils.gcloud import download_chexpert_unzip


class ChexpertDataset(Dataset):
    def __init__(self, root_path, img_filepath, transform, policy='ones'):
        img_filepath = os.path.join(root_path, img_filepath)

        # filter lateral
        # all_tokens = all_tokens[all_tokens[keys[3]] != 'Lateral']
        # all_tokens = all_tokens.reset_index()
        # fill nan with 0
        # all_tokens = all_tokens.fillna(1 if policy == 'ones' else 0)
        # all_tokens = all_tokens.fillna(0)
        # fill -1 with 0
        # all_tokens = all_tokens.replace(-1,0)
        # all_tokens = all_tokens.replace(-1, 1 if policy == 'ones' else 0)

        self.img_paths = []
        self.labels = []
        file_d = open(img_filepath, 'r')
        line = True
        while line:
            line = file_d.readline()
            if line:
                lineItems = line.split()
                img_path = os.path.basename(lineItems[0]).split('_')
                img_path = os.path.join(img_path[0], img_path[1], img_path[2] + '_' + img_path[3])
                img_path = os.path.join(root_path, 'train', img_path)

                label = lineItems[1:]
                label = np.asarray([int(float(i)) for i in label])
                if np.count_nonzero(label) != 0:
                    self.img_paths.append(img_path)
                    self.labels.append(label)

        file_d.close()

        self.root_path = root_path
        self.transform = transform

    def __getitem__(self, item):
        img = os.path.join(self.img_paths[item])
        # print(img)
        img = io.imread(img)
        img = Image.fromarray(img).convert('RGB')
        img_trans1 = self.transform(img)

        target = torch.FloatTensor(self.labels[item])
        # if np.count_nonzero(target) == 0:
        #     target = np.append(target,1)
        # else:
        #     target = np.append(target,0)

        imgs = [img_trans1]
        for i in range(5):
            imgs.append(self.transform(img))
        return imgs, target, item

    def __len__(self):
        return len(self.img_paths)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ChexpertLoader:
    def __init__(self, root_path, batch_size, img_resize=128, gcloud=True):
        self.batch_size = batch_size
        self.root_path = root_path
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # if gcloud:
        #     download_chexpert_unzip(root_path)
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop((img_resize, img_resize), scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.RandomAffine(10)
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)  # not strengthened
            ], p=0.8),
            transforms.RandomApply(
                [GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # transforms.RandomErasing(p=0.5, inplace=True),
        ])
        self.train_transform = transforms.Compose([
            # transforms.Resize(img_resize),
            # transforms.CenterCrop(img_resize),
            transforms.RandomResizedCrop((img_resize, img_resize), scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8),
            transforms.RandomRotation(10),
            # transforms.RandomApply(
            #     [GaussianBlur([.1, 2.])], p=0.5),
            # transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((img_resize, img_resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def run(self, mode='train', ratio=100):
        if mode == 'label':
            file_path = f'train_{ratio}.txt'
            transform = self.train_transform
        if mode == 'unlabel':
            file_path = f'train_500_unlab.txt'
            transform = self.train_transform
        elif mode == 'test':
            file_path = f'train_500_test_10000.txt'
            transform = self.val_transform
        elif mode == 'moco_train':
            file_path = 'train.csv'
            transform = self.moco_transform
        chexpert_dataset = ChexpertDataset(root_path=self.root_path, img_filepath=file_path,
                                           transform=transform)
        if mode == 'moco_train':
            sampler = torch.utils.data.distributed.DistributedSampler(
                chexpert_dataset)
        else:
            sampler = None
        loader = DataLoader(dataset=chexpert_dataset, batch_size=self.batch_size,
                            shuffle=True if mode !='test'  else False, pin_memory=True, sampler=sampler,
                            drop_last=True if 'label' in mode else False, num_workers=8)
        return loader, sampler
