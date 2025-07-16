from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
import torch
import tools

# from torchnet.meter import AUCMeter
import torchvision
from tqdm import tqdm


def unpickle(file):
    import _pickle as cPickle

    with open(file, "rb") as fo:
        dict = cPickle.load(fo, encoding="latin1")
    return dict


class animal_dataset(Dataset):
    def __init__(
        self,
        dataset,
        transform,
        mode,
        pred=[],
        saved=False,
    ):

        self.transform = transform
        self.mode = mode
        saved = saved
        if self.mode == "test":

            data = torchvision.datasets.ImageFolder(root=dataset + "/testing", transform=None)
            self.test_label = []
            self.test_data = []
            if not saved:
                for i in tqdm(range(data.__len__())):
                    image, label = data.__getitem__(i)
                    self.test_label.append(label)
                    self.test_data.append(image)
                torch.save(self.test_label, dataset + "/test_label.pt")
                torch.save(self.test_data, dataset + "/test_data.pt")
                print("data saved")
            else:
                self.test_data = torch.load(dataset + "/test_data.pt")
                self.test_label = torch.load(dataset + "/test_label.pt")
                self.test_data = self.test_data
                self.test_label = self.test_label
        else:
            train_data = []
            train_label = []
            data = torchvision.datasets.ImageFolder(root=dataset + "/training", transform=None)
            if not saved:
                for i in tqdm(range(data.__len__())):
                    image, label = data.__getitem__(i)
                    train_label.append(label)
                    train_data.append(image)
                noise_label = train_label
                torch.save(train_label, dataset + "/train_label.pt")
                torch.save(train_data, dataset + "/train_data.pt")
                print("data saved")
            else:
                train_data = torch.load(dataset + "/train_data.pt")
                train_label = torch.load(dataset + "/train_label.pt")
                train_data = train_data
                train_label = train_label
                noise_label = train_label
            if self.mode == "all":
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    clean = np.array(noise_label) == np.array(train_label)
                elif self.mode == "unlabeled":
                    pred_idx = (pred).nonzero()[0]
                self.train_data = torch.utils.data.Subset(train_data, pred_idx)
                self.noise_label = torch.utils.data.Subset(noise_label, pred_idx)

    def __getitem__(self, index):
        if self.mode == "labeled":
            img, target = self.train_data[index], self.noise_label[index]
            # img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target
        elif self.mode == "unlabeled":
            img = self.train_data[index]
            # img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode == "all":
            img, target = self.train_data[index], self.noise_label[index]
            # img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == "test":
            img, target = self.test_data[index], self.test_label[index]
            # img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != "test":
            return len(self.train_data)
        else:
            return len(self.test_data)


class animal_dataloader:
    def __init__(self, dataset, batch_size, num_workers, saved=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.saved = saved

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def run(self, mode, pred=[]):
        if mode == "warmup":
            all_dataset = animal_dataset(
                dataset=self.dataset,
                transform=self.transform_train,
                mode="all",
                saved=self.saved,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False
            )
            return trainloader

        elif mode == "labeled":
            labeled_dataset = animal_dataset(
                dataset=self.dataset,
                transform=self.transform_train,
                mode="labeled",
                pred=pred,
                saved=self.saved,
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
            )
            return labeled_trainloader
        elif mode == "unlabeled":
            unlabeled_dataset = animal_dataset(
                dataset=self.dataset,
                transform=self.transform_train,
                mode="unlabeled",
                pred=pred,
                saved=self.saved,
            )
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
            )
            return unlabeled_trainloader

        elif mode == "test":
            test_dataset = animal_dataset(
                dataset=self.dataset,
                transform=self.transform_test,
                mode="test",
                saved=self.saved,
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False
            )
            return test_loader

        elif mode == "eval_train":
            eval_dataset = animal_dataset(
                dataset=self.dataset,
                transform=self.transform_test,
                mode="all",
                saved=self.saved,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False
            )
            return eval_loader
