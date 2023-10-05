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
        latent_distribution = self.latent[index] ** (1 / self.T) if index is not None else self.latent
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
                        int(
                            torch.round(num[i] * (probs[i] > 0).sum())
                            .clamp(min=0.0, max=num_classes)
                            .item()
                        )
                        # int(num[i].item())
                        if num is not None else np.random.randint(0, num_classes - 1, dtype=np.uint8),
                        replace=False,
                        p=probs[i],
                    )
                ),
                num_classes,
            ).sum(0)
            for i in range(probs.shape[0])
        ]
    ).cuda()

    return neg


def prior_loss(log_outputs, log_prior):
    return F.kl_div(
        log_outputs,
        (log_prior + log_outputs.log_softmax(0)).log_softmax(1),
        reduction="batchmean",
        log_target=True,
    )


def regkl_loss(log_outputs, log_tildey, log_prior):
    return F.kl_div(
        # (log_tildey.log_softmax(1) + log_prior).log_softmax(1),
        log_tildey.log_softmax(1),
        log_outputs.detach(),
        reduction="batchmean",
        log_target=True,
    )
