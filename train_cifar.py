import torch
from ResNet import resnet_cifar34
import torch.optim as optim
import torch.nn as nn
from dataloader_cifar import cifar_dataloader
import wandb
import pandas as pd
from helper import AverageMeter
import torchmetrics as tm
from tqdm import tqdm
import torch.nn.functional as F
from dynamic_partial import DynamicPartial, sample_neg, prior_loss, regkl_loss
from easydict import EasyDict
from sklearn.mixture import GaussianMixture
import torch.distributions as dist

# from torchsort import soft_rank, soft_sort
import numpy as np


class CIFAR_Trainer:
    def __init__(self, config, name: str):
        self.warmup_epochs = config.warmup_epochs
        self.total_epochs = config.total_epochs
        self.num_classes = config.num_classes
        self.beta = config.beta

        self.net = resnet_cifar34(self.num_classes).cuda()
        self.optim = optim.SGD(
            self.net.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.wd,
            nesterov=config.nesterov,
        )
        self.latent = DynamicPartial(50000, config.beta, config.num_classes)

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=config.lr_decay, gamma=0.1
        )

        self.criterion = nn.CrossEntropyLoss(reduction="none").cuda()
        loader = cifar_dataloader(
            config.dataset,
            config.r,
            config.noise_mode,
            config.batch_size,
            config.num_workers,
            config.root_dir,
        )

        self.train_loader, self.eval_loader = loader.run("train")
        self.test_loader = loader.run("test")
        if config.wandb:
            self.use_wandb = True
            wandb.login()
            wandb.init(project="GNL", config=config, name=name)
        else:
            self.use_wandb = False

        self.logger = pd.DataFrame(
            # columns=["train acc", "train cov", "train ineff", "test acc", "test cov", "test ineff"]
            columns=["train acc", "test acc", "clean_cov", "noisy_cov", "clean_unc", "clean_unc"]
        )

        self.train_acc = AverageMeter()
        self.m_cov = AverageMeter()
        self.m_unc = AverageMeter()
        self.l_ce = AverageMeter()
        self.l_pri = AverageMeter()
        self.l_kl = AverageMeter()
        self.test_acc = AverageMeter()
        self.calc_acc = tm.Accuracy(task="multiclass", num_classes=config.num_classes).cuda()

    def pipeline(self, train_func):
        for epoch in range(self.total_epochs):
            if epoch < self.warmup_epochs:
                self.train(epoch, self.net, self.optim, self.latent)
            else:
                probs = self.eval_train(self.net)
                self.train(epoch, self.net, self.optim, self.latent, probs)

            self.test(self.net)
            self.wandb_update(epoch)
            self.scheduler.step()

    def train(self, epoch: int, net: nn.Module, optimizer: optim.SGD, mov: DynamicPartial, probs=None):
        net.train()
        for batch_idx, (inputs, targets, clean, idx) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch: {epoch}")
        ):
            inputs, targets, clean = inputs.cuda(), targets.cuda(), clean.cuda().to(torch.int64)
            onehot_labels = F.one_hot(targets, self.num_classes).cuda()

            optimizer.zero_grad()
            outputs, tildey, _ = net(inputs)

            pred = F.one_hot(mov.sample_latent(idx).sample(), self.num_classes).float()
            prior_cov = torch.logical_or(pred, onehot_labels).float()

            prior_unc = sample_neg(
                prior_cov, self.num_classes, probs[idx] if probs is not None else None
            )
            prior = (prior_cov + prior_unc).clamp(max=1.0)
            prior = prior / prior.sum(1, keepdim=True)

            mov.update_hist(outputs.softmax(1), idx)

            log_outputs = outputs.log_softmax(1)
            log_prior = prior.clamp(1e-9).log()
            log_tildey = tildey.log_softmax(1)
            ce = self.criterion(tildey, targets).mean()
            pri = prior_loss(log_outputs, log_prior)
            reg_kl = regkl_loss(log_outputs, log_tildey, log_prior)
            l = ce + pri + reg_kl

            l.backward()
            optimizer.step()

            self.metrics_update(inputs, clean, prior, ce, pri, reg_kl)
            self.train_acc.update(self.calc_acc(outputs, clean.int()).item() * 100.0)

    @torch.no_grad()
    def eval_train(self, net: nn.Module):
        net.eval()
        losses = torch.zeros(50000)
        for batch_idx, (inputs, targets, clean, index) in enumerate(self.eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, tildey, _ = net(inputs)
            loss = F.cross_entropy(tildey, targets, reduction="none")
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
        losses = ((losses - losses.min()) / (losses.max() - losses.min())).unsqueeze(1)
        input_loss = losses.reshape(-1, 1)
        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return 1 - torch.from_numpy(prob).cuda()

    @torch.no_grad()
    def test(self, net):
        net.eval()
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _, _ = net(inputs)
            self.test_acc.update(self.calc_acc(outputs, targets.int()).item() * 100.0)

    def metrics_update(self, inputs, clean, prior, ce, pri, reg_kl):
        self.m_cov.update(
            torch.logical_and(prior, F.one_hot(clean, self.num_classes)).sum().item(),
            inputs.shape[0],
        )
        self.m_unc.update((prior > 0).sum(1).float().mean().item())
        self.l_ce.update(ce.item())
        self.l_pri.update(pri.item())
        self.l_kl.update(reg_kl.item())

    def wandb_update(self, epoch):
        stats = {
            "L_ce": self.l_ce.avg,
            "L_pri": self.l_pri.avg,
            "L_kl": self.l_kl.avg,
            "Coverage": self.m_cov.avg,
            "Uncertainty": self.m_unc.avg,
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
                self.m_unc,
                self.train_acc,
                self.test_acc,
            ]
        ]
