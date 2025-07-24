from typing import TYPE_CHECKING, Callable
import torch


__all__ = ["measure", "list_metrics", "metric"]

SUPPORTED_METRICS = {}

def _as_tensor(x) -> "torch.Tensor":
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x
def list_metrics() -> list[str]:
    """
    List all the metrics available.

    This function returns a list of the names of all metrics that have been registered
    using the `@metric` decorator.

    Returns
    -------
    list of str
        A list of the names of all registered metrics.

    Examples
    --------
    >>> list_metrics()
    ['cka', 'unbiased_cka', 'svcca', 'rsa', ...]
    """
    return list(SUPPORTED_METRICS.keys())


def measure(
    metric_name: str, feats_A: "torch.Tensor", feats_B: "torch.Tensor", **kwargs
) -> float:
    """
    Compute a specified metric between two sets of features.

    This function looks up a registered metric by its name and applies it to the
    provided feature tensors. The metric must have been previously registered
    using the `@metric` decorator.

    Parameters
    ----------
    metric_name : str
        The name of the metric to compute. Must be one of the registered metrics.
    feats_A : torch.Tensor
        A PyTorch tensor representing the first set of features (e.g., from a neural network layer).
    feats_B : torch.Tensor
        A PyTorch tensor representing the second set of features.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the metric function.

    Returns
    -------
    float
        The computed metric value.

    Raises
    ------
    ValueError
        If the specified metric name is not recognized.

    Examples
    --------
    >>> measure('cka', feats_A, feats_B)
    0.85

    >>> measure('rsa', feats_A, feats_B, distance_metric="corr")
    0.72
    """

    metric_call = SUPPORTED_METRICS.get(metric_name)

    if not metric_call:
        raise ValueError(
            f"Unrecognized metric: {metric_name}. "
            f"Supported metrics are: {list_metrics()}"
        )
    
    feats_A = _as_tensor(feats_A)
    feats_B = _as_tensor(feats_B)

    return metric_call(feats_A, feats_B, **kwargs)


def metric(func: Callable) -> Callable:
    """
    Decorator to register a metric function in the SUPPORTED_METRICS dictionary.

    This decorator is used to register a function as a metric that can be applied
    to compare neural network features. The decorated function will be added to the
    `SUPPORTED_METRICS` dictionary and can be accessed by its function name.

    Parameters
    ----------
    func : Callable
        The metric function to be registered. The function should take two feature tensors
        as input and return a float representing the similarity or distance between them.

    Returns
    -------
    Callable
        The original function, unchanged.

    Examples
    --------
    >>> @metric
    >>> def my_custom_metric(feats_A, feats_B):
    >>>     # Compute metric
    >>>     return score

    >>> list_metrics()
    ['my_custom_metric', ...]
    """
    SUPPORTED_METRICS[func.__name__] = func
    return func
