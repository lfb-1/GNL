import torch
from VGG import vgg19_bn
import torch.optim as optim
import torch.nn as nn

# from dataloader_cifarN import cifar_dataloader
import dataloader_animal10n as dataloader
import wandb
import pandas as pd
from helper import AverageMeter
import torchmetrics as tm
from tqdm import tqdm
import torch.nn.functional as F
from dynamic_partial import DynamicPartial, sample_neg, prior_loss, pxy_kl, pyx_kl
from easydict import EasyDict
from sklearn.mixture import GaussianMixture
import torch.distributions as dist

# from torchsort import soft_rank, soft_sort
import numpy as np


class ANIMAL_Trainer:
    def __init__(self, config, name: str):
        self.warmup_epochs = config.warmup_epochs
        self.total_epochs = config.total_epochs
        self.num_classes = config.num_classes
        self.num_pri = config.num_prior
        self.beta = config.beta
        self.reg_kl = pxy_kl if config.optim_goal == "pxy" else pyx_kl

        # self.net = resnet_cifar34(self.num_classes).cuda()
        self.net, self.optim, self.scheduler = self.net_optim(config)

        if config.cot == 2:
            self.net2, self.optim2, self.scheduler2 = self.net_optim(config)

        self.criterion = nn.CrossEntropyLoss(reduction="none").cuda()
        # loader = cifar_dataloader(
        #     config.dataset,
        #     batch_size=config.batch_size,
        #     num_workers=config.num_workers,
        #     root_dir=config.root_dir,
        # )
        loader = dataloader.animal_dataloader(
            dataset=config.root_dir,
            batch_size=config.batch_size,
            num_workers=5,
            saved=True,
        )
        self.train_loader = loader.run("warmup")
        # self.train_loader2 = loader.run("warmup")
        self.eval_loader = loader.run("eval_train")
        self.test_loader = loader.run("test")
        self.latent = DynamicPartial(
            len(self.eval_loader.dataset.train_data), config.beta, config.num_classes
        )
        if config.cot == 2:
            self.latent2 = DynamicPartial(
                len(self.eval_loader.dataset.train_data),
                config.beta,
                config.num_classes,
            )
        if config.wandb:
            self.use_wandb = True
            wandb.login()
            wandb.init(project="GNL", config=config, name=name)
        else:
            self.use_wandb = False
        self.logger = pd.DataFrame(
            # columns=["train acc", "train cov", "train ineff", "test acc", "test cov", "test ineff"]
            columns=[
                "train acc",
                "test acc",
                "clean_cov",
                "noisy_cov",
                "clean_unc",
                "clean_unc",
            ]
        )
        self.train_acc = AverageMeter()
        self.m_cov = AverageMeter()
        self.m_unc_clean = AverageMeter()
        self.m_unc_noisy = AverageMeter()
        self.l_ce = AverageMeter()
        self.l_pri = AverageMeter()
        self.l_kl = AverageMeter()
        self.test_acc = AverageMeter()
        self.test_acc2 = AverageMeter()
        self.calc_acc = tm.Accuracy(
            task="multiclass", num_classes=config.num_classes
        ).cuda()

    def net_optim(self, config):
        net = vgg19_bn()
        net.classifier2 = nn.Linear(4096, self.num_classes)
        net.transition = nn.Linear(4096 + self.num_classes, self.num_classes)
        net = net.cuda()

        optimizer = optim.SGD(
            net.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.wd,
            nesterov=config.nesterov,
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.lr_decay, gamma=0.1
        )
        return net, optimizer, scheduler

    def pipeline(self, train_func):
        for epoch in range(self.total_epochs):
            if epoch < self.warmup_epochs:
                self.train(epoch, self.net, self.optim, self.latent, self.latent2)
                if self.net2 is not None:
                    self.train(
                        epoch, self.net2, self.optim2, self.latent2, self.latent
                    )
            else:
                probs = self.eval_train(self.net)
                if self.net2 is not None:
                    probs2 = self.eval_train(self.net2)
                    self.train(
                        epoch,
                        self.net,
                        self.optim,
                        self.latent,
                        self.latent2,
                        probs2,
                    )
                    self.train(
                        epoch,
                        self.net2,
                        self.optim2,
                        self.latent2,
                        self.latent,
                        probs,
                    )

            self.test(self.net, net2=self.net2)
            self.wandb_update(epoch)
            self.scheduler.step()
            self.scheduler2.step()

    def train(self, epoch, net, optimizer, mov1, mov2, probs=None):
        net.train()
        for batch_idx, (inputs, targets, idx) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch: {epoch}")
        ):
            inputs, targets = (
                inputs.cuda(),
                targets.cuda(),
                # self.train_loader.dataset.clean_label[idx].cuda().to(torch.int64),
            )
            onehot_labels = F.one_hot(targets, self.num_classes).cuda()

            optimizer.zero_grad()
            outputs, tildey, _ = net(inputs)

            pred = F.one_hot(mov2.sample_latent(idx).sample(), self.num_classes).float()
            pred2 = F.one_hot(mov1.sample_latent(idx).sample(),self.num_classes).float()
            prior_cov = (pred + onehot_labels + pred2).clamp(max=1.0)
            # prior_cov = [torch.logical_or(pred[i], onehot_labels).float() for i in range(self.num_pri)]

            prior_unc = [
                sample_neg(
                    prior_cov,
                    self.num_classes,
                    probs[idx] if probs is not None else None,
                )
                for i in range(self.num_pri)
            ]
            prior = [
                (prior_cov + prior_unc[i]).clamp(max=1.0) for i in range(self.num_pri)
            ]
            prior = [
                prior[i] / prior[i].sum(1, keepdim=True) for i in range(self.num_pri)
            ]

            mov1.update_hist(outputs.softmax(1), idx)
            log_outputs = outputs.log_softmax(1)
            log_prior = [prior[i].clamp(1e-9).log() for i in range(self.num_pri)]
            # log_tildey = tildey.log_softmax(1)
            ce = self.criterion(tildey, targets).mean()
            pri = (
                sum(
                    [prior_loss(log_outputs, log_prior[i]) for i in range(self.num_pri)]
                )
                / self.num_pri
            )
            reg_kl = (
                sum(
                    [
                        self.reg_kl(log_outputs, tildey, log_prior[i])
                        for i in range(self.num_pri)
                    ]
                )
                / self.num_pri
            )
            l = ce + pri + reg_kl

            l.backward()
            optimizer.step()

            self.metrics_update(inputs, targets, prior[0], ce, pri, reg_kl)
            # self.train_acc.update(self.calc_acc(outputs, clean.int()).item() * 100.0)

    @torch.no_grad()
    def eval_train(self, net: nn.Module, num_classes=100):
        net.eval()
        losses = torch.zeros(50000)
        for batch_idx, (inputs, targets, index) in enumerate(self.eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, tildey, _ = net(inputs)
            loss = F.cross_entropy(outputs, targets, reduction="none")
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
        losses = ((losses - losses.min()) / (losses.max() - losses.min())).unsqueeze(1)
        input_loss = losses.reshape(-1, 1)
        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return 1 - torch.from_numpy(prob).cuda()

    @torch.no_grad()
    def test(self, net, net2=None):
        net.eval()
        net2.eval()
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _, _ = net(inputs)
            if net2 is not None:
                outputs2, _, _ = net2(inputs)
                outputs = outputs1 + outputs2
            else:
                outputs = outputs1
            # self.test_acc2.update(self.calc_acc(outputs2, targets.int()).item() * 100.0)
            self.test_acc.update(self.calc_acc(outputs, targets.int()).item() * 100.0)

    def metrics_update(self, inputs, targets, prior, ce, pri, reg_kl):
        # self.m_cov.update(
        #     torch.logical_and(prior * prior_cov, F.one_hot(clean, self.num_classes))
        #     .sum()
        #     .item(),
        #     inputs.shape[0],
        # )
        # clean_index = targets == clean
        # noisy_index = targets != clean
        # self.m_unc_clean.update(((prior[clean_index] > 0).sum(1).float().mean().item()))
        # self.m_unc_noisy.update(((prior[noisy_index] > 0).sum(1).float().mean().item()))

        # self.m_unc.update((prior > 0).sum(1).float().mean().item())
        self.l_ce.update(ce.item())
        self.l_pri.update(pri.item())
        self.l_kl.update(reg_kl.item())

    def wandb_update(self, epoch):
        stats = {
            "L_ce": self.l_ce.avg,
            "L_pri": self.l_pri.avg,
            "L_kl": self.l_kl.avg,
            "Coverage": self.m_cov.avg,
            # "Uncertainty": self.m_unc.avg,
            "Clean Uncertainty": self.m_unc_clean.avg,
            "Noisy Uncertainty": self.m_unc_noisy.avg,
            "epoch": epoch,
            "train acc": self.train_acc.avg,
            "test acc": self.test_acc.avg,
        }
        wandb.log(stats)
        print(f"Train acc: {self.train_acc.avg} Test acc: {self.test_acc.avg}\n")
        [
            i.reset()
            for i in [
                self.l_ce,
                self.l_pri,
                self.l_kl,
                self.m_cov,
                self.m_unc_noisy,
                self.m_unc_clean,
                self.train_acc,
                self.test_acc,
                # self.test_acc2
            ]
        ]
