# kernels.py

from typing import Callable, Dict, Any, Union, Literal
import torch
import numpy as np

__all__ = [
    "kernel",
    "list_kernels",
    "compute_kernel",
    "linear_kernel",
    "rbf_kernel",
    "knn_kernel",
    "polynomial_kernel",
    "cosine_kernel",
    "sigmoid_kernel",
    "pearson_kernel",
    "spearman_kernel",
    "kendall_kernel",
    "phi_k_kernel",
    "eigen_cosine_kernel",
    "svd_cosine_kernel",
]

KERNELS: Dict[str, Callable] = {}

ArrayLike = Union[np.ndarray, torch.Tensor]
KernelType = Union[
    Literal[
        "linear",
        "rbf",
        "knn",
        "polynomial",
        "cosine",
        "sigmoid",
        "pearson",
        "spearman",
        "kendall",
        "phi_k",
        "eigen-cosine",
        "svd-cosine",
    ],
    str,
]


def kernel(name: str) -> Callable:
    """
    Decorator to register a kernel function in the KERNELS dictionary.

    Parameters
    ----------
    name : str
        The name under which the kernel function will be registered.

    Returns
    -------
    Callable
        The decorator function.
    """

    def decorator(func: Callable) -> Callable:
        KERNELS[name] = func
        return func

    return decorator


def list_kernels() -> list[str]:
    """
    List all registered kernel names.

    Returns
    -------
    list[str]
        A list of names of registered kernels.
    """
    return list(KERNELS.keys())


def compute_kernel(
    features: ArrayLike, name: KernelType = "pearson", **kwargs: Any
) -> ArrayLike:
    """
    Compute a kernel using the registered kernel function.

    Parameters
    ----------
    name : str
        The name of the kernel function to use.
    features : Union[np.ndarray, torch.Tensor]
        The input features to compute the kernel on.
    **kwargs : Any
        Additional keyword arguments to pass to the kernel function.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        The computed kernel matrix.
    """
    if name not in KERNELS:
        raise ValueError(
            f"Kernel '{name}' is not registered. Available kernels: {list_kernels()}"
        )
    return KERNELS[name](features, **kwargs)


@kernel("linear")
def linear_kernel(features: ArrayLike) -> ArrayLike:
    """
    Computes the linear kernel (inner product) between all pairs of samples.

    The linear kernel is equivalent to the dot product of the feature vectors. It is commonly used
    in linear models like support vector machines (SVM) and other linear classifiers.

    Parameters
    ----------
    features : Union[np.ndarray, torch.Tensor]
        The input feature matrix of shape (n_samples, n_features), where `n_samples` is the number of samples
        and `n_features` is the dimensionality of the feature space.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        The computed linear kernel matrix of shape (n_samples, n_samples), where each entry (i, j)
        corresponds to the inner product between the feature vectors of samples i and j.
    """
    return features @ features.T


@kernel("rbf")
def rbf_kernel(features: ArrayLike, gamma: float = 0.1, p: float = 2.0) -> ArrayLike:
    """
    Computes the Radial Basis Function (RBF) kernel with a customizable distance metric.

    Parameters
    ----------
    features : Union[np.ndarray, torch.Tensor]
        The input feature matrix of shape (n_samples, n_features).
    gamma : float, optional
        The parameter for the RBF kernel (default is 0.1).
    p : float, optional
        The parameter for the Minkowski distance metric (default is 2.0, which corresponds to the Euclidean distance).

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        The computed RBF kernel matrix of shape (n_samples, n_samples).
    """
    if isinstance(features, torch.Tensor):
        pairwise_dists = torch.cdist(features, features, p=p)
        return torch.exp(-gamma * pairwise_dists)
    else:
        pairwise_dists = np.linalg.norm(
            features[:, np.newaxis] - features[np.newaxis, :], axis=2, ord=p
        )
        return np.exp(-gamma * pairwise_dists)


@kernel("knn")
def knn_kernel(features: ArrayLike, k: int = 5) -> ArrayLike:
    """
    Computes the k-nearest neighbors (k-NN) kernel matrix for a set of features.

    The k-NN kernel identifies the k nearest neighbors of each sample in the feature space.
    The resulting kernel matrix has entries of 1 where two samples are mutual k-nearest neighbors, and 0 elsewhere.

    Parameters
    ----------
    features : Union[np.ndarray, torch.Tensor]
        The input feature matrix of shape (n_samples, n_features), where `n_samples` is the number of samples
        and `n_features` is the dimensionality of the feature space. The input can be either a NumPy array or a PyTorch tensor.

    k : int, optional
        The number of nearest neighbors to consider for each sample. Default is 5.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        The k-NN kernel matrix of shape (n_samples, n_samples), where each entry (i, j) is 1 if sample j is among
        the k nearest neighbors of sample i, or vice versa. The matrix is symmetric by construction.

    Notes
    -----
    - If the input `features` is a PyTorch tensor, the output will also be a PyTorch tensor.
    - If the input `features` is a NumPy array, the output will also be a NumPy array.
    """
    from sklearn.neighbors import NearestNeighbors

    is_torch = isinstance(features, torch.Tensor)
    if is_torch:
        features = features.cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k).fit(features)
    knn_indices = nbrs.kneighbors(features, return_distance=False)

    n_samples = features.shape[0]
    knn_kernel_matrix = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        knn_kernel_matrix[i, knn_indices[i]] = 1
        knn_kernel_matrix[knn_indices[i], i] = 1

    if is_torch:
        knn_kernel_matrix = torch.tensor(knn_kernel_matrix)
    return knn_kernel_matrix


@kernel("polynomial")
def polynomial_kernel(
    features: ArrayLike, degree: int = 3, coef0: float = 1
) -> ArrayLike:
    """
    Compute the polynomial kernel between all pairs of samples.

    The polynomial kernel is a non-linear kernel that computes the similarity between samples in a feature space
    based on a polynomial function of the inner product of the input features. It is defined as:

        K(x, y) = (x · y + coef0) ** degree

    where `x` and `y` are feature vectors, `coef0` is a constant term, and `degree` is the degree of the polynomial.

    Parameters
    ----------
    features : ArrayLike
        The input feature matrix of shape (n_samples, n_features), where `n_samples` is the number of samples
        and `n_features` is the dimensionality of the feature space. The input can be either a NumPy array or a PyTorch tensor.

    degree : int, optional
        The degree of the polynomial kernel. Default is 3.

    coef0 : float, optional
        The constant term added to the inner product before raising to the power of `degree`. Default is 1.

    Returns
    -------
    ArrayLike
        The polynomial kernel matrix of shape (n_samples, n_samples), where each entry (i, j) represents the
        polynomial kernel applied to the pair of samples `i` and `j`.
    """
    if isinstance(features, torch.Tensor):
        return (torch.mm(features, features.T) + coef0) ** degree
    return (np.dot(features, features.T) + coef0) ** degree


@kernel("cosine")
def cosine_kernel(features: ArrayLike) -> ArrayLike:
    """
    Compute the cosine similarity kernel between all pairs of samples.

    The cosine similarity kernel measures the cosine of the angle between two feature vectors, which represents
    their similarity. It is defined as:

        K(x, y) = (x · y) / (||x|| * ||y||)

    where `x` and `y` are feature vectors, and `||x||` and `||y||` are their respective norms.

    Parameters
    ----------
    features : ArrayLike
        The input feature matrix of shape (n_samples, n_features), where `n_samples` is the number of samples
        and `n_features` is the dimensionality of the feature space. The input can be either a NumPy array or a PyTorch tensor.

    Returns
    -------
    ArrayLike
        The cosine similarity kernel matrix of shape (n_samples, n_samples), where each entry (i, j) represents
        the cosine similarity between the pair of samples `i` and `j`.

    """
    if isinstance(features, torch.Tensor):
        normed_features = features / torch.norm(features, dim=1, keepdim=True)
        return torch.mm(normed_features, normed_features.T)
    normed_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return np.dot(normed_features, normed_features.T)


@kernel("sigmoid")
def sigmoid_kernel(
    features: ArrayLike, alpha: float = 0.01, coef0: float = 0.0
) -> ArrayLike:
    """
    Compute the sigmoid kernel (or hyperbolic tangent kernel) between all pairs of samples.

    The sigmoid kernel is a non-linear kernel that computes the similarity between samples using the hyperbolic tangent
    function. It is defined as:

        K(x, y) = tanh(alpha * (x · y) + coef0)

    where `x` and `y` are feature vectors, `alpha` is a scaling factor, and `coef0` is a constant term.

    Parameters
    ----------
    features : ArrayLike
        The input feature matrix of shape (n_samples, n_features), where `n_samples` is the number of samples
        and `n_features` is the dimensionality of the feature space. The input can be either a NumPy array or a PyTorch tensor.

    alpha : float, optional
        The scaling factor applied to the inner product before applying the hyperbolic tangent function. Default is 0.01.

    coef0 : float, optional
        The constant term added to the scaled inner product. Default is 0.0.

    Returns
    -------
    ArrayLike
        The sigmoid kernel matrix of shape (n_samples, n_samples), where each entry (i, j) represents the
        similarity between the pair of samples `i` and `j` using the sigmoid kernel.

    """
    if isinstance(features, torch.Tensor):
        return torch.tanh(alpha * torch.mm(features, features.T) + coef0)
    return np.tanh(alpha * np.dot(features, features.T) + coef0)


@kernel("pearson")
def pearson_kernel(features: ArrayLike) -> ArrayLike:
    """
    Compute the Pearson correlation kernel between all pairs of samples.

    This kernel measures the linear correlation between pairs of samples, producing a kernel matrix where each entry
    represents the Pearson correlation coefficient between two samples.

    Parameters
    ----------
    features : ArrayLike
        The input feature matrix of shape (n_samples, n_features).

    Returns
    -------
    ArrayLike
        The Pearson correlation kernel matrix of shape (n_samples, n_samples).
    """
    is_torch = isinstance(features, torch.Tensor)
    if is_torch:
        features = features.cpu().numpy()

    # Compute Pearson correlation matrix
    corr_matrix = np.corrcoef(features)

    # Convert back to torch.Tensor if the input was a torch.Tensor
    if is_torch:
        corr_matrix = torch.tensor(corr_matrix)

    return corr_matrix


@kernel("spearman")
def spearman_kernel(features: ArrayLike) -> ArrayLike:
    """
    Compute the Spearman's rank correlation kernel between all pairs of samples.

    This kernel measures the rank-based correlation between pairs of samples, producing a kernel matrix where each entry
    represents the Spearman correlation coefficient between two samples.

    Parameters
    ----------
    features : ArrayLike
        The input feature matrix of shape (n_samples, n_features).

    Returns
    -------
    ArrayLike
        The Spearman correlation kernel matrix of shape (n_samples, n_samples).
    """
    from scipy.stats import spearmanr

    is_torch = isinstance(features, torch.Tensor)
    if is_torch:
        features = features.cpu().numpy()

    corr_matrix, _ = spearmanr(features, axis=1)

    if is_torch:
        corr_matrix = torch.tensor(corr_matrix)

    return corr_matrix

@kernel("phi_k")
def phi_k_kernel(features: ArrayLike) -> ArrayLike:
    """
    Compute the phi_k (φ_k) kernel between all pairs of samples.

    Parameters
    ----------
    features : ArrayLike
        A NumPy array or a Pandas DataFrame containing the feature data.
        The data can contain categorical, ordinal, or continuous variables.

    Returns
    -------
    ArrayLike
        The phi_k correlation matrix as a NumPy array.
    """
    import pandas as pd

    try:
        import phik # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'phik' package is required to compute the phi_k kernel. "
            "You can install it via 'pip install phik'."
        )
    # Ensure the input is a DataFrame for the phik function
    is_torch = isinstance(features, torch.Tensor)
    if is_torch:
        features = features.cpu().numpy()

    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features.T)

    # Compute the phi_k correlation matrix
    phi_k_corr = features.phik_matrix(interval_cols=features.columns)

    # Return the result as a NumPy array
    phi_k_corr = phi_k_corr.to_numpy()

    if is_torch:
        phi_k_corr = torch.tensor(phi_k_corr)
    
    return phi_k_corr


@kernel("eigen-cosine")
def eigen_cosine_kernel(features: ArrayLike, weight_by_eigenvalues: bool = False) -> ArrayLike:
    """
    Transforms the feature matrix to its eigenvector basis, optionally weights by eigenvalues, 
    and computes the cosine similarity between all pairs of points in the transformed space.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    weight_by_eigenvalues : bool, optional
        Whether to weight the transformed features by their corresponding eigenvalues. Default is False.

    Returns
    -------
    cosine_sim_matrix : np.ndarray
        Cosine similarity matrix of shape (n_samples, n_samples) in the eigenvector-transformed space.
    """
    from sklearn.preprocessing import normalize


    is_torch = isinstance(features, torch.Tensor)
    if is_torch:
        features = features.cpu().numpy()

    # Step 1: Compute the covariance matrix (features should be centered first)
    centered_features = features - np.mean(features, axis=0)
    covariance_matrix = np.cov(centered_features, rowvar=False)
    
    # Step 2: Perform eigen decomposition to get eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    
    # Step 3: Transform the features into the eigenbasis (project data onto eigenvectors)
    transformed_features = centered_features @ eigenvectors
    
    # Step 4: Optionally weight the transformed features by eigenvalues
    if weight_by_eigenvalues:
        transformed_features = transformed_features * eigenvalues
    
    # Step 5: Normalize the transformed features (to compute cosine similarity)
    normalized_features = normalize(transformed_features)
    
    # Step 6: Compute the cosine similarity matrix (dot product of normalized features)
    cosine_sim_matrix = normalized_features @ normalized_features.T

    if is_torch:
        cosine_sim_matrix = torch.tensor(cosine_sim_matrix)
    
    return cosine_sim_matrix

@kernel("svd-cosine")
def svd_cosine_kernel(features: ArrayLike, weight_by_singular_values: bool = False) -> ArrayLike:
    """
    Transforms the feature matrix to its SVD basis, optionally weights by singular values,
    and computes the cosine similarity between all pairs of points in the transformed space.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    weight_by_singular_values : bool, optional
        Whether to weight the transformed features by their corresponding singular values. Default is False.

    Returns
    -------
    cosine_sim_matrix : np.ndarray
        Cosine similarity matrix of shape (n_samples, n_samples) in the SVD-transformed space.
    """
    from sklearn.preprocessing import normalize
    is_torch = isinstance(features, torch.Tensor)
    if is_torch:
        features = features.cpu().numpy()

    # Step 1: Perform SVD on the centered feature matrix (U, S, V^T)
    centered_features = features - np.mean(features, axis=0)
    U, singular_values, Vt = np.linalg.svd(centered_features, full_matrices=False)
    
    # Step 2: Transform the features using U (the left singular vectors)
    transformed_features = U
    
    # Step 3: Optionally weight the transformed features by singular values
    if weight_by_singular_values:
        transformed_features = transformed_features * singular_values
    
    # Step 4: Normalize the transformed features (to compute cosine similarity)
    normalized_features = normalize(transformed_features)
    
    # Step 5: Compute the cosine similarity matrix (dot product of normalized features)
    cosine_sim_matrix = normalized_features @ normalized_features.T
    
    if is_torch:
        cosine_sim_matrix = torch.tensor(cosine_sim_matrix)

    return cosine_sim_matrix



