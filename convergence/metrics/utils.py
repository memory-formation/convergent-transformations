"""
Metrics implementation from the paper 
"Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). 
The Platonic Representation Hypothesis. ICLR."
Code source: https://github.com/minyoungg/platonic-rep/
"""

import torch
import numpy as np
from typing import Callable
import logging

try:
    import pymp
    pymp_available = True
except ImportError:
    pymp_available = False

__all__ = [
    "hsic_unbiased",
    "hsic_biased",
    "compute_knn_accuracy",
    "compute_nearest_neighbors",
    "longest_ordinal_sequence",
    "compute_distance",
    "remove_outliers",
]


def hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Compute the biased HSIC (the original CKA)"""
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def compute_knn_accuracy(knn: torch.Tensor) -> float:
    """
    Compute the accuracy of the nearest neighbors. Assumes index is the gt label.
    Args:
        knn: a torch tensor of shape N x topk
    Returns:
        acc: a float representing the accuracy
    """
    n = knn.shape[0]
    acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
    acc = acc.float().view(n, -1).max(dim=1).values.mean()
    return acc


def compute_nearest_neighbors(feats: torch.Tensor, topk: int = 1) -> torch.Tensor:
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn


def longest_ordinal_sequence(
    X: torch.Tensor, Y: torch.Tensor, num_threads: int = 4
) -> torch.Tensor:
    """For each pair in X and Y, compute the length of the longest sub-sequence (LCS)"""

    def lcs_length(x, y):
        """
        Compute the length of the longest common subsequence between two sequences.
        This is a classic dynamic programming implementation.
        """
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs = compute_distance(X, Y, lcs_length, num_threads=num_threads)
    return lcs


def compute_distance(
    X: torch.Tensor, Y: torch.Tensor, dist_fn: Callable, num_threads: int = 4
) -> torch.Tensor:
    """compute distance in parallel"""
    B, N = X.shape
    distances = np.zeros(B)
    X, Y = X.cpu().numpy(), Y.cpu().numpy()

    if pymp_available:
        with pymp.Parallel(num_threads) as p:
            for i in p.range(B):
                distances[i] = dist_fn(X[i], Y[i])
    else:
        logging.warning(
            "Please install the pymp library using `pip install pymp-pypi` "
            "to speed up non-batched metrics"
        )
        for i in range(B):
            distances[i] = dist_fn(X[i], Y[i])
    return torch.tensor(distances)


def remove_outliers(feats: torch.Tensor, q: float, exact=False, max_threshold=None):
    if q == 1:
        return feats

    if exact:
        # sorts the whole tensor and gets the q-th percentile
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        # quantile for element in the tensor and take the average
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()

    if max_threshold is not None:
        max_threshold = max(max_threshold, q_val)

    return feats.clamp(-q_val, q_val)
