import torch
from ResNet import resnet_cifar34
import torch.optim as optim
import torch.nn as nn
from dataloader_cifar import cifar_dataloader
import wandb
import pandas as pd
from helper import AverageMeter, LossWeightEstimator
import torchmetrics as tm
from tqdm import tqdm
import torch.nn.functional as F
from dynamic_partial import DynamicPartial, sample_neg, prior_loss, pxy_kl, pyx_kl
from easydict import EasyDict
import torch.distributions as dist

# from torchsort import soft_rank, soft_sort
import math


class CIFAR_Trainer:
    def __init__(self, config, name: str):
        self.warmup_epochs = config.warmup_epochs
        self.total_epochs = config.total_epochs
        self.num_classes = config.num_classes
        self.num_pri = config.num_prior
        self.beta = config.beta
        self.reg_kl = pxy_kl if config.optim_goal == "pxy" else pyx_kl

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
        self.num_samples = len(self.train_loader.dataset)
        self.clean_estimator = LossWeightEstimator(
            self.num_samples,
            momentum=getattr(config, "gmm_momentum", 0.9),
            temperature=1.0,
        )
        self.gmm_target_temperature = getattr(config, "gmm_temperature", 1.0)
        self.gmm_transition_epochs = getattr(config, "gmm_transition_epochs", 5)
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
        self.calc_acc = tm.Accuracy(
            task="multiclass", num_classes=config.num_classes
        ).cuda()

    def pipeline(self, train_func):
        for epoch in range(self.total_epochs):
            if epoch < self.warmup_epochs:
                self.train(epoch, self.net, self.optim, self.latent)
            else:
                alpha = self._transition_alpha(epoch)
                self._update_temperature(alpha)
                probs = self.eval_train(self.net)
                probs = self._blend_probs(probs, alpha)
                self.train(epoch, self.net, self.optim, self.latent, probs)

            self.test(self.net)
            self.wandb_update(epoch)
            self.scheduler.step()

    def _transition_alpha(self, epoch: int) -> float:
        if self.gmm_transition_epochs <= 0:
            return 1.0
        progress = epoch - self.warmup_epochs + 1
        if progress <= 0:
            return 0.0
        return float(min(1.0, progress / self.gmm_transition_epochs))

    def _update_temperature(self, alpha: float) -> None:
        current = 1.0 + alpha * (self.gmm_target_temperature - 1.0)
        self.clean_estimator.temperature = max(1e-3, current)

    def _blend_probs(self, probs: torch.Tensor, alpha: float) -> torch.Tensor:
        if alpha >= 1.0:
            return probs
        return probs * alpha + torch.full_like(probs, 0.5) * (1.0 - alpha)

    def train(
        self,
        epoch: int,
        net: nn.Module,
        optimizer: optim.SGD,
        mov: DynamicPartial,
        probs=None,
    ):
        net.train()
        for batch_idx, (inputs, targets, clean, idx) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch: {epoch}")
        ):
            inputs, targets, clean = (
                inputs.cuda(),
                targets.cuda(),
                clean.cuda().to(torch.int64),
            )
            onehot_labels = F.one_hot(targets, self.num_classes).cuda()

            optimizer.zero_grad()
            outputs, tildey, _ = net(inputs)

            #! Conf training
            val_split = int(0.5 * outputs.shape[0])
            cal_logits = outputs[:val_split]
            cal_labels = targets[:val_split]
            test_logits = outputs[val_split:]
            test_labels = targets[val_split:]

            scores = 1 - cal_logits.log_softmax(1)[torch.arange(cal_logits.shape[0]).cuda(), cal_labels].detach()
            quantile_value= (
                math.ceil(outputs.shape[0] + 1) * (1 - 0.05) / outputs.shape[0]
            )
            qhat = torch.quantile(scores, quantile_value,dim=0)
            test_scores = 1 - test_logits.log_softmax(1)
            pred_sets = torch.sigmoid(qhat - test_scores)

            loss = (
                0.01
                * torch.log(
                    torch.sum(pred_sets, dim=1)
                ).mean()
            )

            log_outputs = outputs.log_softmax(1)
            log_prior = prior.clamp(1e-9).log()
            ce = self.criterion(tildey, targets).mean()
            pri = prior_loss(log_outputs, log_prior)
            reg_kl = self.reg_kl(log_outputs, tildey, log_prior)
            l = ce + pri + reg_kl

            l.backward()
            optimizer.step()

            self.metrics_update(inputs, clean, targets,
                                prior, prior, ce, pri, reg_kl)
            self.train_acc.update(self.calc_acc(
                outputs, clean.int()).item() * 100.0)

    @torch.no_grad()
    def eval_train(self, net: nn.Module, num_classes=100):
        net.eval()
        losses = torch.zeros(self.num_samples)
        confidences = torch.zeros(self.num_samples)
        for batch_idx, (inputs, targets, clean, index) in enumerate(self.eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _, _ = net(inputs)
            loss = F.cross_entropy(outputs, targets, reduction="none")
            conf = outputs.softmax(1).gather(1, targets.unsqueeze(1)).squeeze(1)
            losses[index] = loss.detach().cpu()
            confidences[index] = conf.detach().cpu()

        self.clean_estimator.update(losses, confidences)
        clean_prob = self.clean_estimator.predict_clean_probability()
        noisy_prob = (1.0 - clean_prob).clamp(1e-4, 1.0 - 1e-4)
        return noisy_prob.cuda()

    @torch.no_grad()
    def test(self, net):
        net.eval()
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _, _ = net(inputs)
            self.test_acc.update(self.calc_acc(
                outputs, targets.int()).item() * 100.0)

    def metrics_update(self, inputs, clean, targets, prior, prior_cov, ce, pri, reg_kl):
        self.m_cov.update(
            torch.logical_and(prior * prior_cov,
                              F.one_hot(clean, self.num_classes))
            .sum()
            .item(),
            inputs.shape[0],
        )
        clean_index = targets == clean
        noisy_index = targets != clean
        self.m_unc_clean.update(
            ((prior[clean_index] > 0).sum(1).float().mean().item()))
        self.m_unc_noisy.update(
            ((prior[noisy_index] > 0).sum(1).float().mean().item()))

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
        print(
            f"Train acc: {self.train_acc.avg} Test acc: {self.test_acc.avg}\n")
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
            ]
        ]
