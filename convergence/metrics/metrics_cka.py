"""
Metrics implementation from the paper 
"Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). 
The Platonic Representation Hypothesis. ICLR."
Code source: https://github.com/minyoungg/platonic-rep/
"""

from typing import Literal, Optional

import torch

from .utils import hsic_unbiased, hsic_biased
from .metrics import metric


KernelMetric = Literal["ip", "rbf"]

__all__ = ["cka", "unbiased_cka", "cknna"]


@metric
def cka(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    kernel_metric: KernelMetric = "ip",
    rbf_sigma: float = 1.0,
    unbiased: bool = False,
) -> float:
    """
    Computes the (un)biased Centered Kernel Alignment (CKA) between two sets of features.

    Parameters
    ----------
    feats_A : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_A) representing the first set of features.
    feats_B : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_B) representing the second set of features.
    kernel_metric : {"ip", "rbf"}, optional
        The kernel metric to use. "ip" for inner product (linear kernel) and "rbf" for
        Radial Basis Function (RBF) kernel. Default is "ip".
    rbf_sigma : float, optional
        The bandwidth parameter for the RBF kernel. Only used if kernel_metric is "rbf".
        Default is 1.0.
    unbiased : bool, optional
        If True, computes the unbiased CKA. If False, computes the biased CKA. Default is False.

    Returns
    -------
    float
        The computed CKA value between 0 and 1.

    Raises
    ------
    ValueError
        If an invalid kernel metric is provided.

    References
    ----------
    Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
    Similarity of neural network representations revisited. (ICML 2019).
    """

    if kernel_metric == "ip":
        # Compute kernel matrices for the linear case
        K = torch.mm(feats_A, feats_A.T)
        L = torch.mm(feats_B, feats_B.T)
    elif kernel_metric == "rbf":
        # COMPUTES RBF KERNEL
        K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma**2))
        L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma**2))
    elif kernel_metric == "cosine":
        feats_A = feats_A / torch.norm(feats_A, dim=1, keepdim=True)
        feats_B = feats_B / torch.norm(feats_B, dim=1, keepdim=True)
        K = torch.mm(feats_A, feats_A.T)
        L = torch.mm(feats_B, feats_B.T)
    else:
        raise ValueError(f"Invalid kernel metric {kernel_metric}")

    # Compute HSIC values
    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kk = hsic_fn(K, K)
    hsic_ll = hsic_fn(L, L)
    hsic_kl = hsic_fn(K, L)

    cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    return cka_value.item()


@metric
def unbiased_cka(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    kernel_metric: KernelMetric = "ip",
    rbf_sigma: float = 1.0,
) -> float:
    """
    Computes the unbiased Centered Kernel Alignment (CKA) between two sets of features.

    Parameters
    ----------
    feats_A : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_A) representing the first set of features.
    feats_B : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_B) representing the second set of features.
    kernel_metric : {"ip", "rbf"}, optional
        The kernel metric to use. "ip" for inner product (linear kernel) and "rbf" for
        Radial Basis Function (RBF) kernel. Default is "ip".
    rbf_sigma : float, optional
        The bandwidth parameter for the RBF kernel. Only used if kernel_metric is "rbf".
        Default is 1.0.

    Returns
    -------
    float
        The computed unbiased CKA value between 0 and 1.

    References
    ----------
    Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
    Similarity of neural network representations revisited. (ICML 2019).
    """
    return cka(
        feats_A,
        feats_B,
        kernel_metric=kernel_metric,
        rbf_sigma=rbf_sigma,
        unbiased=True,
    )


@metric
def cknna(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    topk: Optional[int] = None,
    distance_agnostic: bool = False,
    unbiased: bool = True,
) -> float:
    """
    Computes the Centered Kernel Nearest Neighbor Alignment (CKNNA) between two sets of features.

    Parameters
    ----------
    feats_A : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_A) representing the first set of features.
    feats_B : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_B) representing the second set of features.
    topk : int, optional
        The number of nearest neighbors to consider for each sample. If None, uses all other samples.
        Must be greater than or equal to 2. Default is None.
    distance_agnostic : bool, optional
        If True, treats all nearest neighbors equally regardless of distance. Default is False.
    unbiased : bool, optional
        If True, computes the unbiased similarity between nearest neighbors. Default is True.

    Returns
    -------
    float
        The computed CKNNA value between 0 and 1.

    Raises
    ------
    ValueError
        If `topk` is less than 2.

    References
    ----------
    Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
    Similarity of neural network representations revisited. (ICML 2019).
    """
    n = feats_A.shape[0]

    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")

    if topk is None:
        topk = feats_A.shape[0] - 1

    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    def similarity(K, L, topk):
        if unbiased:
            K_hat = K.clone().fill_diagonal_(float("-inf"))
            L_hat = L.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K, L

        # get topk indices for each row
        # if unbiased we cannot attend to the diagonal unless full topk
        # else we can attend to the diagonal
        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)

        # create masks for nearest neighbors
        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)

        # intersection of nearest neighbors
        mask = mask_K * mask_L

        if distance_agnostic:
            sim = mask * 1.0
        else:
            if unbiased:
                sim = hsic_unbiased(mask * K, mask * L)
            else:
                sim = hsic_biased(mask * K, mask * L)
        return sim

    sim_kl = similarity(K, L, topk)
    sim_kk = similarity(K, K, topk)
    sim_ll = similarity(L, L, topk)

    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()
