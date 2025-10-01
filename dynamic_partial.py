import torch.nn as nn
import torch
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np


class DynamicPartial(nn.Module):
    def __init__(self, num_samples, beta=0.9, num_classes=10, T=0.5, memory_size=1000, momentum=0.999):
        super(DynamicPartial, self).__init__()
        self.latent = (torch.ones(num_samples, num_classes) / num_classes).cuda()
        self.beta = beta
        self.T = T

    def update_hist(self, probs, index):
        probs = torch.clamp(probs, 1e-8, 1.0 - 1e-8).detach()

        # Additional safety check for NaN/Inf values
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("[WARNING] DynamicPartial: NaN/Inf detected in probs, skipping update")
            return

        probs_sum = probs.sum(1, keepdim=True)
        probs_sum = torch.clamp(probs_sum, min=1e-8)  # Prevent division by zero
        probs = probs / probs_sum

        self.latent[index] = self.beta * self.latent[index] + (1 - self.beta) * probs

    def sample_latent(self, index=None):
        latent_distribution = self.latent[index] if index is not None else self.latent

        # Apply temperature scaling with numerical stability
        latent_scaled = latent_distribution ** (1 / self.T)

        # Add safety checks for numerical stability
        if torch.isnan(latent_scaled).any() or torch.isinf(latent_scaled).any():
            print("[WARNING] DynamicPartial: NaN/Inf in latent_scaled, using uniform distribution")
            if index is not None:
                uniform_dist = torch.ones_like(latent_distribution) / latent_distribution.shape[-1]
                return dist.Categorical(uniform_dist)
            else:
                uniform_dist = torch.ones_like(self.latent) / self.latent.shape[-1]
                return dist.Categorical(uniform_dist)

        # Normalize with numerical stability
        norm_ld_sum = latent_scaled.sum(1, keepdim=True)
        norm_ld_sum = torch.clamp(norm_ld_sum, min=1e-8)
        norm_ld = latent_scaled / norm_ld_sum

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
                        (
                            int(torch.round(num[i] * (probs[i] > 0).sum()).clamp(min=0.0, max=num_classes).item())
                            # int(num[i].item())
                            if num is not None
                            else np.random.randint(0, num_classes - 1, dtype=np.uint8)
                        ),
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


#! Two approaches for Eq. 12
#! Option 1: log_outputs.softmax(0)
#! Option 2: log_outputs / log_outputs.sum(0,keepdim=True), logsumexp is used for computing in log space


def prior_loss(log_outputs, log_prior):
    # Add numerical stability constants
    max_clip = 50.0

    # Clamp inputs to prevent extreme values
    log_outputs_0_clamped = torch.clamp(log_outputs[0], min=-max_clip, max=max_clip)
    log_outputs_1_clamped = torch.clamp(log_outputs[1], min=-max_clip, max=max_clip)
    log_prior_clamped = torch.clamp(log_prior, min=-max_clip, max=max_clip)

    log_outputs_normalized = log_outputs_0_clamped - torch.logsumexp(log_outputs_1_clamped, dim=0, keepdim=True)
    log_outputs_normalized = torch.clamp(log_outputs_normalized, min=-max_clip, max=max_clip)

    # More stable computation of target distribution
    pre_softmax = log_prior_clamped + log_outputs_normalized
    pre_softmax = torch.clamp(pre_softmax, min=-max_clip, max=max_clip)
    target_dist = F.log_softmax(pre_softmax, dim=1)

    # Clamp target_dist as well
    target_dist = torch.clamp(target_dist, min=-max_clip, max=max_clip)

    kl_result = F.kl_div(
        log_outputs_0_clamped,
        target_dist,
        reduction="batchmean",
        log_target=True,
    )

    # Handle NaN/Inf with graceful fallback
    if torch.isnan(kl_result) or torch.isinf(kl_result):
        print("[WARNING] prior_loss: Loss is NaN or Inf, using fallback!")
        kl_result = torch.tensor(0.0, device=kl_result.device, requires_grad=True)

    return kl_result


def pxy_kl(log_outputs, tildey, log_prior, w_i=0.5):
    # Add numerical stability constants
    max_clip = 50.0

    # Clamp inputs to prevent extreme values
    tildey_log_softmax = F.log_softmax(tildey, dim=1)
    tildey_log_softmax = torch.clamp(tildey_log_softmax, min=-max_clip, max=max_clip)
    log_prior_clamped = torch.clamp(log_prior, min=-max_clip, max=max_clip)

    # More stable computation of input distribution
    pre_softmax = tildey_log_softmax + log_prior_clamped
    pre_softmax = torch.clamp(pre_softmax, min=-max_clip, max=max_clip)
    input_dist = F.log_softmax(pre_softmax, dim=1)

    # Clamp the final distribution
    input_dist = torch.clamp(input_dist, min=-max_clip, max=max_clip)
    log_outputs_0_clamped = torch.clamp(log_outputs[0].detach(), min=-max_clip, max=max_clip)

    kl = F.kl_div(
        input_dist,
        log_outputs_0_clamped,
        reduction="none",
        log_target=True,
    )

    # Handle potential NaN/Inf in KL computation
    kl = torch.clamp(kl, min=-100.0, max=100.0)
    kl_masked = torch.where(torch.isnan(kl) | torch.isinf(kl), torch.zeros_like(kl), kl)

    result = ((1.0 - 2 * w_i) * kl_masked.sum(1)).mean()

    # Additional safety check
    if torch.isnan(result) or torch.isinf(result):
        print("[WARNING] pxy_kl: Loss is NaN or Inf, using fallback!")
        result = torch.tensor(0.0, device=result.device, requires_grad=True)

    return result


def pyx_kl(log_outputs, tildey, log_prior, w_i=0.5):
    # Add numerical stability constants
    eps = 1e-8
    max_clip = 50.0  # Prevent extremely large values

    log_outputs_normalized = log_outputs[0] - torch.logsumexp(log_outputs[1], dim=0, keepdim=True)

    # Clamp log_prior to prevent extreme values
    log_prior_clamped = torch.clamp(log_prior, min=-max_clip, max=max_clip)
    log_outputs_normalized_clamped = torch.clamp(log_outputs_normalized, min=-max_clip, max=max_clip)

    # Add numerical stability for logsumexp computation with clamping
    logsumexp_input = log_outputs_normalized_clamped + log_prior_clamped
    logsumexp_term = torch.logsumexp(logsumexp_input, dim=1, keepdim=True)
    logsumexp_term = torch.clamp(logsumexp_term, min=-max_clip, max=max_clip)

    # Compute tildey log_softmax with stability
    tildey_log_softmax = F.log_softmax(tildey, dim=1)
    tildey_log_softmax = torch.clamp(tildey_log_softmax, min=-max_clip, max=max_clip)

    # More stable computation of input_dist
    pre_softmax = tildey_log_softmax + logsumexp_term
    pre_softmax = torch.clamp(pre_softmax, min=-max_clip, max=max_clip)
    input_dist = F.log_softmax(pre_softmax, dim=1)

    # Clamp the KL divergence inputs to prevent extreme values
    input_dist_clamped = torch.clamp(input_dist, min=-max_clip, max=max_clip)
    log_outputs_0_clamped = torch.clamp(log_outputs[0].detach(), min=-max_clip, max=max_clip)

    kl = F.kl_div(
        input_dist_clamped,
        log_outputs_0_clamped,
        reduction="none",
        log_target=True,
    )

    # Clamp KL values and check for problematic values
    kl = torch.clamp(kl, min=-100.0, max=100.0)  # More conservative clamping for KL
    kl_masked = torch.where(torch.isnan(kl) | torch.isinf(kl), torch.zeros_like(kl), kl)

    result = ((1.0 - 2 * w_i) * kl_masked.sum(1)).mean()

    # Additional safety check
    if torch.isnan(result) or torch.isinf(result):
        print("[WARNING] pyx_kl: Loss is still NaN or Inf after fixes!")
        result = torch.tensor(0.0, device=result.device, requires_grad=True)

    return result
