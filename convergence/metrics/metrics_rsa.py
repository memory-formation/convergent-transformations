import torch

from typing import Literal, Union
from .metrics import metric

__all__ = ["rsa"]


@metric
def rsa(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    distance_metric: Union[Literal["corr"], float] = "corr",
) -> float:
    """
    Compute the Representational Similarity Analysis (RSA) between two sets of features.

    RSA is used to compare the representational dissimilarity matrices (RDMs) of two sets
    of features, typically from different regions or layers of a neural network, to measure
    their representational similarity.

    Parameters
    ----------
    feats_A : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_A) representing the first set of features.
    feats_B : torch.Tensor
        A 2D tensor of shape (n_samples, n_features_B) representing the second set of features.
    distance_metric : {"corr", float}, optional
        The metric to use for computing the dissimilarity matrices:
        - "corr": Uses the correlation distance (1 - Pearson correlation coefficient).
        - float: Uses the Minkowski distance with the specified p-value. Default is "corr".

    Returns
    -------
    float
        The RSA score, representing the correlation between the upper triangular parts of the RDMs
        of `feats_A` and `feats_B`. The value ranges between -1 and 1, where 1 indicates perfect
        similarity and -1 indicates perfect dissimilarity.

    References
    ----------
    Kriegeskorte, N., Mur, M., & Bandettini, P. (2008).
    Representational Similarity Analysis - Connecting the Branches of Systems Neuroscience.
    Frontiers in Systems Neuroscience, 2, 4.
    """
    if distance_metric == "corr":
        # Compute the correlation distance matrix for feats_A
        corr_matrix_A = torch.corrcoef(feats_A)
        dissimilarity_matrix_A = 1 - corr_matrix_A

        # Compute the correlation distance matrix for feats_B
        corr_matrix_B = torch.corrcoef(feats_B)
        dissimilarity_matrix_B = 1 - corr_matrix_B
    else:
        # Compute the Minkowski distance matrix for feats_A with given p
        p = distance_metric
        dissimilarity_matrix_A = torch.cdist(feats_A, feats_A, p=p)

        # Compute the Minkowski distance matrix for feats_B with given p
        dissimilarity_matrix_B = torch.cdist(feats_B, feats_B, p=p)

    # Extract the upper triangular part of the dissimilarity matrices (excluding the diagonal)
    triu_indices = torch.triu_indices(
        dissimilarity_matrix_A.size(0), dissimilarity_matrix_A.size(1), offset=1
    )

    upper_tri_A = dissimilarity_matrix_A[triu_indices[0], triu_indices[1]]
    upper_tri_B = dissimilarity_matrix_B[triu_indices[0], triu_indices[1]]

    # Compute the correlation between the two upper triangular parts
    combined = torch.stack((upper_tri_A, upper_tri_B))
    correlation = torch.corrcoef(combined)[0, 1]

    return correlation.item()
