from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch

# from torchnet.meter import AUCMeter
import tools

# from autoaugment import CIFAR10Policy


def unpickle(file):
    import _pickle as cPickle

    with open(file, "rb") as fo:
        dict = cPickle.load(fo, encoding="latin1")
    return dict


class cifar_dataset(Dataset):
    def __init__(
        self,
        dataset,
        r,
        noise_mode,
        root_dir,
        transform,
        mode,
        noise_file="",
        pred=[],
        log="",
    ):
        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {
            0: 0,
            2: 0,
            4: 7,
            7: 7,
            1: 1,
            9: 1,
            3: 5,
            5: 3,
            6: 6,
            8: 8,
        }  # class transition for asymmetric noise

        if self.mode == "test":
            if dataset == "cifar10":
                test_dic = unpickle("%s/test_batch" % root_dir)
                self.test_data = test_dic["data"]
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic["labels"]
                self.num_classes = 10
            elif dataset == "cifar100":
                test_dic = unpickle("%s/test" % root_dir)
                self.test_data = test_dic["data"]
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic["fine_labels"]
                self.num_classes = 100
        else:
            train_data = []
            train_label = []
            if dataset == "cifar10":
                for n in range(1, 6):
                    dpath = "%s/data_batch_%d" % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic["data"])
                    train_label = train_label + data_dic["labels"]
                    self.num_classes = 10
                train_data = np.concatenate(train_data)
            elif dataset == "cifar100":
                train_dic = unpickle("%s/train" % root_dir)
                train_data = train_dic["data"]
                train_label = train_dic["fine_labels"]
                self.num_classes = 100
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
            elif noise_mode == "instance":
                data_ = torch.from_numpy(train_data).float().cuda()
                targets_ = torch.IntTensor(train_label).cuda()
                dataset_ = zip(data_, targets_)
                train_label = torch.FloatTensor(train_label).cuda()
                noise_label = tools.get_instance_noisy_label(
                    self.r,
                    dataset_,
                    train_label,
                    num_classes=self.num_classes,
                    feature_size=32 * 32 * 3,
                    norm_std=0.1,
                    seed=123,
                )
                train_label = train_label.cpu()
                # noise_label = noise_label.tolist()
                # print("save noisy labels to %s ..." % noise_file)
                # torch.save(noise_label, noise_file)
                # json.dump(noise_label.tolist(), open(noise_file, "w"))
            elif noise_mode == "real":
                if dataset == "cifar10":
                    noise_file = torch.load("./CIFAR-10_human.pt")
                    clean_label = noise_file["clean_label"]
                    worst_label = noise_file["worse_label"]
                    aggre_label = noise_file["aggre_label"]
                    random_label1 = noise_file["random_label1"]
                    random_label2 = noise_file["random_label2"]
                    random_label3 = noise_file["random_label3"]
                elif dataset == "cifar100":
                    noise_file = torch.load("./CIFAR-100_human.pt")
                    clean_label = noise_file["clean_label"]
                    noisy_label = noise_file["noisy_label"]
            else:  # inject noise
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r * 50000)
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode == "sym":
                            noiselabel = random.randint(0, self.num_classes - 1)
                            noise_label.append(noiselabel)
                        elif noise_mode == "asym":
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])
                # print("save noisy labels to %s ..." % noise_file)
                # json.dump(noise_label, open(noise_file, "w"))

            if self.mode == "all":
                self.train_data = train_data
                self.noise_label = noise_label
                self.clean_label = train_label
            else:
                if self.mode == "labeled" or self.mode == "unlabeled" or self.mode == "partial":
                    pred_idx = pred.nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                self.clean_label = [train_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode == "all":
            img, target, clean = self.train_data[index], self.noise_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            return img1, target, clean, index
        elif self.mode == "test":
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != "test":
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader:
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        if self.dataset == "cifar10":
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            self.transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        elif self.dataset == "cifar100":
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomRotation(15),
                    # CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )
            self.transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

    def run(self, mode):
        if mode == "train":
            all_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_train,
                mode="all",
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True,
            )
            eval_loader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True,
            )
            return trainloader, eval_loader

        elif mode == "test":
            test_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="test",
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader
