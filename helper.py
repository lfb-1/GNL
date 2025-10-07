import numpy as np
import torch
from sklearn.mixture import GaussianMixture


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class LossWeightEstimator:
    """Tracks per-sample loss statistics and fits a robust two-component GMM."""

    def __init__(self, num_samples, momentum=0.9, temperature=1.0):
        self.num_samples = num_samples
        self.momentum = momentum
        self.temperature = temperature
        self._stats = torch.zeros(num_samples, 2)
        self._initialised = False

    def update(self, losses: torch.Tensor, confidences: torch.Tensor):
        """Update the running statistics with new loss/confidence measurements."""

        if losses.device.type != "cpu":
            losses = losses.cpu()
        if confidences.device.type != "cpu":
            confidences = confidences.cpu()

        losses = torch.log1p(losses.clamp_min(0.0))
        features = torch.stack([losses, (1.0 - confidences).clamp_min(0.0)], dim=1)

        if not self._initialised:
            self._stats = features.clone()
            self._initialised = True
        else:
            self._stats.mul_(self.momentum).add_((1.0 - self.momentum) * features)

    def predict_clean_probability(self):
        """Return per-sample clean probabilities estimated by the GMM."""

        if not self._initialised:
            return torch.full((self.num_samples,), 0.5)

        stats = torch.nan_to_num(self._stats, nan=0.0, posinf=0.0, neginf=0.0)

        mean = stats.mean(dim=0, keepdim=True)
        std = stats.std(dim=0, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        norm_stats = (stats - mean) / std

        data = norm_stats.cpu().numpy().astype(np.float64)

        try:
            gmm = GaussianMixture(
                n_components=2,
                covariance_type="full",
                reg_covar=5e-4,
                max_iter=50,
                tol=1e-3,
                random_state=0,
            )
            gmm.fit(data)
            prob = gmm.predict_proba(data)
            
            # CRITICAL: Robust component selection
            # Clean samples should have lower loss (feature 0) AND higher confidence (lower feature 1)
            # Use combined criteria to identify clean component
            component_0_mean_loss = gmm.means_[0, 0]
            component_1_mean_loss = gmm.means_[1, 0]
            
            # The component with lower loss is more likely to be clean
            clean_component = np.argmin(gmm.means_[:, 0])
            
            # Sanity check: if means are too close, revert to uniform
            mean_diff = abs(component_0_mean_loss - component_1_mean_loss)
            if mean_diff < 0.1:  # Threshold for ambiguous separation
                print(f"[WARNING] GMM components poorly separated (diff={mean_diff:.4f}), using uniform probs")
                clean_prob = np.full(self.num_samples, 0.5, dtype=np.float64)
            else:
                clean_prob = prob[:, clean_component]
                print(f"[INFO] GMM: clean_component={clean_component}, mean_diff={mean_diff:.4f}, clean_prob range=[{clean_prob.min():.3f}, {clean_prob.max():.3f}]")
        except Exception as e:
            print(f"[WARNING] GMM fitting failed: {e}, using uniform probs")
            clean_prob = np.full(self.num_samples, 0.5, dtype=np.float64)

        clean_prob = np.clip(clean_prob, 1e-5, 1.0 - 1e-5)

        if self.temperature != 1.0:
            logits = np.log(clean_prob / (1.0 - clean_prob))
            logits *= self.temperature
            clean_prob = 1.0 / (1.0 + np.exp(-logits))

        return torch.from_numpy(clean_prob.astype(np.float32))


class LogFT(object):
    def __init__(self, log_file) -> None:
        self.log_file = log_file

    def __call__(self, output) -> None:
        self.log_file.write(output)
        self.log_file.flush()
        print(output)
