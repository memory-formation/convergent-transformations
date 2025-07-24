"""
Metrics implementation from the paper 
"Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). 
The Platonic Representation Hypothesis. ICLR."
Code source: https://github.com/minyoungg/platonic-rep/
"""

import torch
import numpy as np
from sklearn.cross_decomposition import CCA


from .metrics import metric

__all__ = ["svcca"]


@metric
def svcca(feats_A: torch.Tensor, feats_B: torch.Tensor, cca_dim: int = 10) -> float:
    """
    Computes the Singular Vector Canonical Correlation Analysis (SVCCA) similarity between
    two sets of features.

    SVCCA is used to measure the similarity between two sets of neural network activations
    or features by combining Singular Value Decomposition (SVD) and Canonical Correlation Analysis (CCA).

    Parameters
    ----------
    feats_A : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_A) representing the first set of features.
    feats_B : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_B) representing the second set of features.
    cca_dim : int, optional
        The number of dimensions to retain in the Canonical Correlation Analysis. Default is 10.

    Returns
    -------
    float
        The SVCCA similarity value between 0 and 1, where 1 indicates perfect alignment of the
        singular vectors between the two feature sets.

    Raises
    ------
    ValueError
        If an invalid dimension is provided for CCA.

    References
    ----------
    Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017).
    SVCCA: Singular Vector Canonical Correlation Analysis for
    Deep Learning Dynamics and Interpretability.
    In Advances in Neural Information Processing Systems (NeurIPS 2017).
    """

    # Center and scale the activations
    def preprocess_activations(act):
        act = act - torch.mean(act, axis=0)
        act = act / (torch.std(act, axis=0) + 1e-8)
        return act

    feats_A = preprocess_activations(feats_A)
    feats_B = preprocess_activations(feats_B)

    # Compute SVD
    U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
    U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)

    U1 = U1.cpu().detach().numpy()
    U2 = U2.cpu().detach().numpy()

    # Compute CCA
    cca = CCA(n_components=cca_dim)
    cca.fit(U1, U2)
    U1_c, U2_c = cca.transform(U1, U2)

    # sometimes it goes to nan, this is just to avoid that
    U1_c += 1e-10 * np.random.randn(*U1_c.shape)
    U2_c += 1e-10 * np.random.randn(*U2_c.shape)

    # Compute SVCCA similarity
    svcca_similarity = np.mean(
        [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
    )
    return svcca_similarity
