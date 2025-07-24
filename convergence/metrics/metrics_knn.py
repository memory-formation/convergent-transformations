"""
Metrics implementation from the paper 
"Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). 
The Platonic Representation Hypothesis. ICLR."
Code source: https://github.com/minyoungg/platonic-rep/
"""

import torch

from .utils import (
    compute_nearest_neighbors,
    compute_knn_accuracy,
    longest_ordinal_sequence,
    compute_distance,
)
from .metrics import metric

__all__ = ["cycle_knn", "mutual_knn", "lcs_knn", "edit_distance_knn"]


@metric
def cycle_knn(feats_A: torch.Tensor, feats_B: torch.Tensor, topk: int = 10) -> float:
    """
    Computes the cycle KNN accuracy between two sets of features.

    In this metric, the nearest neighbors are computed for each set of features,
    and then the accuracy of these neighbors matching in both sets is calculated.

    Parameters
    ----------
    feats_A : torch.Tensor
        A 2D tensor of shape (n_samples, feat_dim) representing the first set of features.
    feats_B : torch.Tensor
        A 2D tensor of shape (n_samples, feat_dim) representing the second set of features.
    topk : int, optional
        The number of nearest neighbors to consider for each sample. Default is 10.

    Returns
    -------
    float
        A float representing the cycle KNN accuracy.

    References
    ----------
    Huh, M., Cheung, B., Wang, T., & Isola, P. (2024).
    The Platonic Representation Hypothesis. ICLR.
    """
    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)
    return compute_knn_accuracy(knn_A[knn_B]).item()


@metric
def mutual_knn(feats_A: torch.Tensor, feats_B: torch.Tensor, topk: int = 10) -> float:
    """
    Computes the mutual KNN accuracy between two sets of features.

    Mutual KNN accuracy measures how often the nearest neighbors of one set of features
    are also the nearest neighbors in the other set.

    Parameters
    ----------
    feats_A : torch.Tensor
        A 2D tensor of shape (n_samples, feat_dim) representing the first set of features.
    feats_B : torch.Tensor
        A 2D tensor of shape (n_samples, feat_dim) representing the second set of features.
    topk : int, optional
        The number of nearest neighbors to consider for each sample. Default is 10.

    Returns
    -------
    float
        A float representing the mutual KNN accuracy.

    References
    ----------
    Huh, M., Cheung, B., Wang, T., & Isola, P. (2024).
    The Platonic Representation Hypothesis. ICLR.
    """
    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)

    n = knn_A.shape[0]
    topk = knn_A.shape[1]

    # Create a range tensor for indexing
    range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

    # Create binary masks for knn_A and knn_B
    lvm_mask = torch.zeros(n, n, device=knn_A.device)
    llm_mask = torch.zeros(n, n, device=knn_A.device)

    lvm_mask[range_tensor, knn_A] = 1.0
    llm_mask[range_tensor, knn_B] = 1.0

    acc = (lvm_mask * llm_mask).sum(dim=1) / topk

    return acc.mean().item()


@metric
def lcs_knn(feats_A: torch.Tensor, feats_B: torch.Tensor, topk: int = 10) -> float:
    """
    Computes the Longest Common Subsequence (LCS) KNN similarity between two sets of features.

    This metric calculates the similarity between the nearest neighbors in two sets
    by finding the longest common subsequence of their nearest neighbors.

    Parameters
    ----------
    feats_A : torch.Tensor
        A 2D tensor of shape (n_samples, feat_dim) representing the first set of features.
    feats_B : torch.Tensor
        A 2D tensor of shape (n_samples, feat_dim) representing the second set of features.
    topk : int, optional
        The number of nearest neighbors to consider for each sample. Default is 10.

    Returns
    -------
    float
        A float representing the LCS KNN similarity score.

    References
    ----------
    Huh, M., Cheung, B., Wang, T., & Isola, P. (2024).
    The Platonic Representation Hypothesis. ICLR.
    """
    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)
    score = longest_ordinal_sequence(knn_A, knn_B).float().mean()
    return score.item()


@metric
def edit_distance_knn(
    feats_A: torch.Tensor, feats_B: torch.Tensor, topk: int = 10
) -> float:
    """
    Computes the edit distance between the nearest neighbors of two sets of features.

    This metric calculates the dissimilarity between the nearest neighbors in two sets of features
    by computing the edit distance between their neighbor sequences.

    Parameters
    ----------
    feats_A : torch.Tensor
        A 2D tensor of shape (n_samples, feat_dim) representing the first set of features.
    feats_B : torch.Tensor
        A 2D tensor of shape (n_samples, feat_dim) representing the second set of features.
    topk : int, optional
        The number of nearest neighbors to consider for each sample. Default is 10.

    Returns
    -------
    float
        A float representing the normalized edit distance between the KNN sequences of the two sets of features.

    References
    ----------
    Huh, M., Cheung, B., Wang, T., & Isola, P. (2024).
    The Platonic Representation Hypothesis. ICLR.
    """
    # Delay import
    import torchaudio.functional as TAF

    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)

    # given N x topk with integer entries, compute edit distance
    # n = knn_A.shape[0]
    topk = knn_A.shape[1]

    edit_distance = compute_distance(knn_A, knn_B, TAF.edit_distance)
    return (1 - torch.mean(edit_distance) / topk).item()
