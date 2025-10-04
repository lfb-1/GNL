import torch
from ResNet import resnet_cifar34
import torch.optim as optim
import torch.nn as nn

from dataloader_cifarN import cifar_dataloader
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
from collections import deque


class CIFARN_Trainer:
    def __init__(self, config, name: str):
        self.warmup_epochs = config.warmup_epochs
        self.total_epochs = config.total_epochs
        self.num_classes = config.num_classes
        self.num_pri = config.num_prior
        self.beta = config.beta
        self.reg_kl = pxy_kl if config.optim_goal == "pxy" else pyx_kl

        self.net, self.optim, self.latent, self.scheduler = self.net_optim_sch_mov(config)
        self.memory_queue_len = config.memory_queue_len
        # self.net2, self.optim2, self.latent2, self.scheduler2 = self.net_optim_sch_mov(
        #     config
        # )
        self.criterion = nn.CrossEntropyLoss(reduction="none").cuda()
        loader = cifar_dataloader(
            config.dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            root_dir=config.root_dir,
        )
        self.train_loader = loader.run("warmup", target=config.target)
        self.eval_loader = loader.run("eval_train", target=config.target)
        self.test_loader = loader.run("test")
        self.num_samples = len(self.eval_loader.dataset)
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
            wandb.init(project="GNL_new", config=config, name=name)
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
        self.calc_acc = tm.Accuracy(task="multiclass", num_classes=config.num_classes).cuda()

    def net_optim_sch_mov(self, config):
        net = resnet_cifar34(self.num_classes).cuda()
        optimizer = optim.SGD(
            net.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.wd,
            nesterov=config.nesterov,
        )
        latent = DynamicPartial(50000, config.beta, config.num_classes)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_decay, gamma=0.1)
        return net, optimizer, latent, scheduler

    def pipeline(self, train_func):
        if self.memory_queue_len > 0:
            memory_queue = deque(maxlen=self.memory_queue_len)
        else:
            memory_queue = None
        for epoch in range(self.total_epochs):
            if epoch < self.warmup_epochs:
                self.train(epoch, self.net, self.optim, self.latent)
                # self.train(epoch, self.net2, self.optim2, self.latent2)
            else:
                alpha = self._transition_alpha(epoch)
                self._update_temperature(alpha)
                probs = self.eval_train(self.net)
                probs = self._blend_probs(probs, alpha)
                self.train(epoch, self.net, self.optim, self.latent, probs, memory_queue=memory_queue)
                # self.train(epoch, self.net2, self.optim2, self.latent2,self.latent, probs)

            self.test(self.net)
            self.wandb_update(epoch)
            self.scheduler.step()
            # self.scheduler2.step()

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

    def train(self, epoch, net, optimizer, mov, probs=None, memory_queue=None):
        net.train()
        for batch_idx, (inputs, targets, idx) in enumerate(tqdm(self.train_loader, desc=f"Epoch: {epoch}")):
            inputs, targets, clean = (
                inputs.cuda(),
                targets.cuda(),
                self.train_loader.dataset.clean_label[idx].cuda().to(torch.int64),
            )
            onehot_labels = F.one_hot(targets, self.num_classes).cuda()

            optimizer.zero_grad()

            # l = np.random.beta(0.5, 0.5)
            # l = max(l, 1 - l)
            # l = torch.ones_like(torch.from_numpy(l)) * 0.5
            # l = 0.5

            # mix_idx = torch.randperm(inputs.shape[0])
            # mix_inputs = l * inputs + (1 - l) * inputs[mix_idx]
            # mix_targets = l * onehot_labels + (1 - l) * onehot_labels[mix_idx]

            outputs, tildey, _ = net(inputs)

            pred = F.one_hot(mov.sample_latent(idx).sample(), self.num_classes).float()
            prior_cov = (pred + onehot_labels).clamp(max=1.0)
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
                torch.clamp(p, max=1.0) / torch.clamp(p.sum(1, keepdim=True), min=1e-8)
                for p in prior_unc
            ]
            # mix_prior = [
            #     l * prior[i] + (1 - l) * prior[i][mix_idx] for i in range(self.num_pri)
            # ]
            # mix_prior = [
            #     mix_prior[i] / mix_prior[i].sum(1, keepdim=True)
            #     for i in range(self.num_pri)
            # ]

            mov.update_hist(outputs.softmax(1), idx)
            log_outputs = outputs.log_softmax(1)
            # Add numerical stability to log_prior computation
            log_prior = [torch.clamp(prior[i], min=1e-9, max=1.0).log() for i in range(self.num_pri)]

            if memory_queue is None:
                extended_log_outputs = log_outputs
            else:
                if len(memory_queue) == memory_queue.maxlen:
                    # Queue is full, pop the oldest
                    prev_log_outputs = memory_queue.popleft()
                    # Add numerical stability for concatenation
                    extended_log_outputs = torch.cat([log_outputs, prev_log_outputs], dim=0)
                elif len(memory_queue) == 0:
                    extended_log_outputs = log_outputs
                else:
                    # Queue is not full, use all previous batches
                    all_prev_outputs = torch.cat(list(memory_queue), dim=0)
                    extended_log_outputs = torch.cat([log_outputs, all_prev_outputs], dim=0)

                # Apply stability clipping to the extended outputs
                extended_log_outputs = torch.clamp(extended_log_outputs, min=-50.0, max=50.0)
                memory_queue.append(torch.clamp(log_outputs.detach(), min=-50.0, max=50.0))

            # mix_log_prior = [mix_prior[i].clamp(1e-9).log() for i in range(self.num_pri)]
            # log_tildey = tildey.log_softmax(1)
            ce = self.criterion(tildey, targets).mean()
            # ce = -torch.mean(
            #     torch.sum(F.log_softmax(tildey, dim=1) * mix_targets, dim=1)
            # )
            pri = sum([prior_loss([log_outputs, extended_log_outputs], log_prior[i]) for i in range(self.num_pri)]) / self.num_pri
            reg_kl = (
                sum(
                    [
                        self.reg_kl([log_outputs, extended_log_outputs], tildey, log_prior[i], 0.5 if probs is None else 1.0 - probs[idx])
                        for i in range(self.num_pri)
                    ]
                )
                / self.num_pri
            )
            l = ce + pri + reg_kl

            l.backward()

            # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            self.metrics_update(inputs, clean, targets, prior[0], prior_cov, ce, pri, reg_kl)
            self.train_acc.update(self.calc_acc(outputs, clean.int()).item() * 100.0)

    @torch.no_grad()
    def eval_train(self, net: nn.Module, num_classes=100):
        net.eval()
        losses = torch.zeros(self.num_samples)
        confidences = torch.zeros(self.num_samples)
        for batch_idx, (inputs, targets, index) in enumerate(self.eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net.forward_test(inputs)
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
        # net2.eval()
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net.forward_test(inputs)
            # outputs2, _, _ = net2(inputs)
            # outputs = outputs
            self.test_acc.update(self.calc_acc(outputs, targets.int()).item() * 100.0)

    def metrics_update(self, inputs, clean, targets, prior, prior_cov, ce, pri, reg_kl):
        self.m_cov.update(
            torch.logical_and(prior * prior_cov, F.one_hot(clean, self.num_classes)).sum().item(),
            inputs.shape[0],
        )
        clean_index = targets == clean
        noisy_index = targets != clean
        self.m_unc_clean.update(((prior[clean_index] > 0).sum(1).float().mean().item()))
        self.m_unc_noisy.update(((prior[noisy_index] > 0).sum(1).float().mean().item()))

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
            ]
        ]
