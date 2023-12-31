import torch.nn as nn
import torch
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np


class DynamicPartial(nn.Module):
    def __init__(self, num_samples, beta=0.9, num_classes=10, T=0.5):
        super(DynamicPartial, self).__init__()
        self.latent = (torch.ones(num_samples, num_classes) / num_classes).cuda()
        self.beta = beta
        self.T = T

    def update_hist(self, probs, index):
        probs = torch.clamp(probs, 1e-4, 1.0 - 1e-4).detach()
        probs /= probs.sum(1, keepdim=True)

        self.latent[index] = self.beta * self.latent[index] + (1 - self.beta) * probs

    def sample_latent(self, index=None):
        latent_distribution = (
            self.latent[index] ** (1 / self.T) if index is not None else self.latent
        )
        norm_ld = latent_distribution / latent_distribution.sum(1, keepdim=True)
        return dist.Categorical(norm_ld)


class DynamicPartial_Mix(nn.Module):
    def __init__(self, num_samples, beta=0.9, num_classes=10, T=0.5) -> None:
        super(DynamicPartial_Mix, self).__init__()
        self.latent = (torch.ones(num_samples, num_classes) / num_classes).cuda()
        self.beta = beta
        self.T = T

    def update_hist(self, probs, index):
        probs = torch.clamp(probs, 1e-4, 1.0 - 1e-4).detach()
        probs /= probs.sum(1, keepdim=True)
        self.latent[index] = self.beta * self.latent[index] + (1 - self.beta) * probs
        # self.q = (
        #     mixup_l * self.latent[index] + (1 - mixup_l) * self.latent[index][mix_index]
        # )

    def sample_latent(self, index, mix_index, mixup_l):
        # latent_distribution = self.q ** (1 / self.T)
        # q = mixup_l * self.latent[index] + (1 - mixup_l) * self.latent[index][mix_index]
        latent_distribution = (
            self.latent[index] ** (1 / self.T) if index is not None else self.latent
        )
        norm_ld = latent_distribution / latent_distribution.sum(1, keepdim=True)
        return dist.Categorical(norm_ld)


def sample_neg(prior_cov, num_classes, num=None):
    probs = prior_cov.detach().cpu().numpy().astype("float64")
    probs = (1 - probs) / (1 - probs).sum(1, keepdims=True)

    neg = torch.vstack(
        [
            F.one_hot(
                torch.tensor(
                    np.random.choice(
                        num_classes,
                        # int(
                        #     torch.round(num[i] * num_classes - (probs[i] == 0).sum())
                        #     .clamp(min=0.0, max=num_classes)
                        #     .item()
                        # )
                        int(
                            torch.round(num[i] * ((probs[i] > 0).sum() - 1))
                            .clamp(min=0.0, max=num_classes)
                            .item()
                        )
                        if num is not None
                        else np.random.randint(0, num_classes - 2, dtype=np.uint8),
                        replace=False,
                        p=probs[i],
                    )
                ),
                num_classes,
            ).sum(0)
            for i in range(probs.shape[0])
        ]
    ).cuda()
    neg[neg > 1] = 1

    return neg


#! Two approaches for Eq. 12
#! Option 1: log_outputs.softmax(0)
#! Option 2: log_outputs / log_outputs.sum(0,keepdim=True), logsumexp is used for computing in log space
def prior_loss(log_outputs, log_prior):
    return F.kl_div(
        log_outputs,
        (log_prior + log_outputs.log_softmax(0)).log_softmax(1),
        # (log_prior + (log_outputs - torch.logsumexp(log_outputs, dim=0, keepdim=True))).log_softmax(1),
        reduction="batchmean",
        log_target=True,
    )


def pxy_kl(log_outputs, tildey, log_prior):
    return F.kl_div(
        (tildey.log_softmax(1) + log_prior).log_softmax(1),
        log_outputs.detach(),
        reduction="batchmean",
        log_target=True,
    )


def pyx_kl(log_outputs, tildey, log_prior):
    return F.kl_div(
        (
            tildey.log_softmax(1)
            + torch.logsumexp(
                log_outputs.log_softmax(0) + log_prior, dim=1, keepdim=True
            )
            # + torch.logsumexp(
            #     (log_outputs - torch.logsumexp(log_outputs, dim=0, keepdim=True)) + log_prior,
            #     dim=1,
            #     keepdim=True,
            # )
        ).log_softmax(1),
        log_outputs.detach(),
        reduction="batchmean",
        log_target=True,
    )
