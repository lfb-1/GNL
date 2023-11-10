from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def unpickle(file):
    import _pickle as cPickle

    with open(file, "rb") as fo:
        dict = cPickle.load(fo, encoding="latin1")
    return dict


class cifar_dataset(Dataset):
    def __init__(self, mode, root_dir, dataset, transform, target=None) -> None:
        super(cifar_dataset, self).__init__()
        self.mode = mode
        self.transform = transform
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
            # train_label = []
            if dataset == "cifar10":
                for n in range(1, 6):
                    dpath = "%s/data_batch_%d" % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic["data"])
                    # train_label = train_label + data_dic["labels"]
                    self.num_classes = 10
                train_data = np.concatenate(train_data)
            elif dataset == "cifar100":
                train_dic = unpickle("%s/train" % root_dir)
                train_data = train_dic["data"]
                # train_label = train_dic["fine_labels"]
                self.num_classes = 100
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            if dataset == "cifar10":
                assert target in [
                    "worse_label",
                    "aggre_label",
                    "random_label1",
                    "random_label2",
                    "random_label3",
                ]
                noise_file = torch.load("./CIFAR-10_human.pt")
                clean_label = noise_file["clean_label"]
                noise_label = noise_file[target]
            elif dataset == "cifar100":
                assert target == "noisy_label"
                noise_file = torch.load("./CIFAR-100_human.pt")
                clean_label = noise_file["clean_label"]
                noise_label = noise_file[target]

            self.train_data = train_data
            self.noise_label = noise_label
            self.clean_label = torch.tensor(clean_label)

    def __getitem__(self, index):
        if self.mode == "all":
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            return img1, target, index
        elif self.mode == "test":
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != "test":
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]


class cifar_dataloader:
    def __init__(self, dataset, batch_size, num_workers, root_dir, log=None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        # self.log = log
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

    def run(self, mode, target=None):
        if mode == "warmup":
            all_dataset = cifar_dataset(
                mode="all",
                dataset=self.dataset,
                root_dir=self.root_dir,
                target=target,
                transform=self.transform_train,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False,
            )
            return trainloader
        elif mode == "test":
            test_dataset = cifar_dataset(
                mode="test",
                dataset=self.dataset,
                root_dir=self.root_dir,
                transform=self.transform_test,
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
            )
            return test_loader

        elif mode == "eval_train":
            eval_dataset = cifar_dataset(
                dataset=self.dataset,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="all",
                target=target,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
            )
            return eval_loader
