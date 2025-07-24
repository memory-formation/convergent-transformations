from typing import Optional

import torch
import torch.nn.functional as F

from .metrics import measure, remove_outliers


__all__ = ["prepare_features", "compute_alignment"]


def prepare_features(feats: torch.Tensor, q: float = 0.95, exact: bool = False):
    """
    Prepare features by removing outliers and normalizing
    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    feats = remove_outliers(feats.float(), q=q, exact=exact)
    return feats


def compute_alignment(
    x_feats: torch.Tensor,
    y_feats: torch.Tensor,
    metric: str = "mutual_knn",
    precise: bool = True,
    normalize: bool = True,
    prepare: bool = True,
    device: str = "cuda",
    low_memory: Optional[bool] = False,
    include_all_x: Optional[bool] = True,
    include_all_y: Optional[bool] = True,
    **kwargs,
) -> list[dict]:
    """
    Computes alignment scores between features from two sets of layers.

    This function calculates alignment scores for all combinations of layers
    from the input feature sets `x_feats` and `y_feats`, using a specified
    metric. The results are returned as a list of dictionaries containing the
    alignment score and related information for each combination.

    Args:
        x_feats (torch.Tensor): A tensor of shape (N, L, D) representing features
            from the source, where N is the number of samples, L is the number
            of layers, and D is the feature dimensionality.
        y_feats (torch.Tensor): A tensor of shape (N, L, D) representing features
            from the target, with the same structure as `x_feats`.
        metric (str): The metric to use for measuring alignment between features.
        precise (bool): If True, uses exact quantizing during feature preparation.
            This can be set to False to speed up processing when running on CPU.
        normalize (bool): Whether to normalize the feature vectors before computing
            the alignment score.
        prepare (bool): If True, prepares the features by converting them to
            `float32` and moving them to the specified device (`device` argument).
            Uses precise quantizing if `precise` is True.
        device (str): The device to use for computation, such as "cuda" for GPU or
            "cpu" for CPU. Only relevant if `prepare` is True.
        **kwargs: Additional keyword arguments to be passed to the `measure`
            function, such as `topk` for KNN-based metrics.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - "score" (float): The computed alignment score.
            - "index_x" (int): The index of the layer from `x_feats` used in the
              alignment. If -1, all layers are combined.
            - "index_y" (int): The index of the layer from `y_feats` used in the
              alignment. If -1, all layers are combined.
            - "metric" (str): The metric used for the alignment.
            - "normalize" (bool): Whether normalization was applied to the features.
            - Additional fields included in **kwargs.

    Example:
        >>> import torch
        >>> x_feats = torch.randn(100, import torch
5, 64)  # 100 samples, 5 layers, 64 features
        >>> y_feats = torch.randn(100, 5, 64)
        >>> scores = compute_alignment(x_feats, y_feats, metric="mutual_knn", topk=10)
        >>> print(scores)

    Notes:
        - When `index_x` or `index_y` is -1, the features are flattened across
          all layers (i.e., reshaped from (N, L, D) to (N, L * D)).
        - If `prepare` is True, the features will be preprocessed to ensure
          consistency in precision and device placement.
        - Setting `precise` to False can speed up computations on CPU but may
          reduce alignment accuracy.
        - The `**kwargs` parameter allows passing additional options to the
          `measure` function, providing flexibility for different metrics.
    """

    if prepare:
        x_feats = prepare_features(x_feats.float().to(device), exact=precise)
        if low_memory:
            x_feats = x_feats.cpu()
            # Clear memory
            torch.cuda.empty_cache()
        y_feats = prepare_features(y_feats.float().to(device), exact=precise)
        if low_memory:
            y_feats = y_feats.cpu()
            # Clear memory
            torch.cuda.empty_cache()

    scores = []

    initial_x = -1 if include_all_x else 0
    for i in range(initial_x, x_feats.shape[1]):
        x = x_feats.flatten(1, 2) if i == -1 else x_feats[:, i, :]
        if normalize:
            x = F.normalize(x, p=2, dim=-1)

        if low_memory:
            x = x.to(device)
        initial_y = -1 if include_all_y else 0
        for j in range(initial_y, y_feats.shape[1]):
            y = y_feats.flatten(1, 2) if j == -1 else y_feats[:, j, :]

            if normalize:
                y = F.normalize(y, p=2, dim=-1)

            if low_memory:
                y = y.to(device)
            score = measure(metric, x, y, **kwargs)
            if low_memory:
                y = y.cpu()
                torch.cuda.empty_cache()

            result = {
                "score": score,
                "index_x": i,
                "index_y": j,
                "metric": metric,
                "normalize": normalize,
                **kwargs,
            }

            scores.append(result)

        if low_memory:
            x = x.cpu()
        # Clear memory
        torch.cuda.empty_cache()

    return scores
