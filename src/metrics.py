"""Metrics for assessing quality of generated samples.

This module provides metrics to evaluate how well generated samples
match the distribution of original samples.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp


def maximum_mean_discrepancy(X_original, X_generated, kernel='rbf', gamma=1.0):
    """Compute Maximum Mean Discrepancy (MMD) between two distributions.
    
    MMD measures the distance between two distributions in a reproducing
    kernel Hilbert space (RKHS). Lower values indicate better match.
    
    Parameters
    ----------
    X_original : np.ndarray or torch.Tensor
        Original samples, shape (n_original, dim)
    X_generated : np.ndarray or torch.Tensor
        Generated samples, shape (n_generated, dim)
    kernel : str, default='rbf'
        Kernel type. Options: 'rbf' (Gaussian), 'linear'
    gamma : float, default=1.0
        Kernel bandwidth parameter for RBF kernel
        
    Returns
    -------
    float
        MMD value (lower is better)
        
    References
    ----------
    Gretton, A., et al. (2012). A kernel two-sample test.
    Journal of Machine Learning Research.
    """
    # Convert to numpy if needed
    if isinstance(X_original, torch.Tensor):
        X_original = X_original.cpu().numpy()
    if isinstance(X_generated, torch.Tensor):
        X_generated = X_generated.cpu().numpy()
    
    def compute_kernel(X, Y, kernel_type='rbf', gamma=1.0):
        """Compute kernel matrix between X and Y."""
        if kernel_type == 'rbf':
            # RBF kernel: k(x,y) = exp(-gamma * ||x-y||^2)
            distances_sq = cdist(X, Y, metric='sqeuclidean')
            return np.exp(-gamma * distances_sq)
        elif kernel_type == 'linear':
            # Linear kernel: k(x,y) = x^T y
            return X @ Y.T
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    n_original = X_original.shape[0]
    n_generated = X_generated.shape[0]
    
    # Compute kernel matrices
    K_XX = compute_kernel(X_original, X_original, kernel, gamma)
    K_YY = compute_kernel(X_generated, X_generated, kernel, gamma)
    K_XY = compute_kernel(X_original, X_generated, kernel, gamma)
    
    # MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
    # Exclude diagonal for unbiased estimator
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    term1 = K_XX.sum() / (n_original * (n_original - 1)) if n_original > 1 else 0
    term2 = K_YY.sum() / (n_generated * (n_generated - 1)) if n_generated > 1 else 0
    term3 = K_XY.sum() / (n_original * n_generated)
    
    mmd_squared = term1 + term2 - 2 * term3
    
    # Return MMD (take square root, ensure non-negative)
    return float(np.sqrt(max(0, mmd_squared)))


def two_sample_test(X_original, X_generated, test='ks'):
    """Perform two-sample tests to compare distributions.
    
    Tests the null hypothesis that the two samples come from the same
    distribution. Returns p-values for each dimension.
    
    Parameters
    ----------
    X_original : np.ndarray or torch.Tensor
        Original samples, shape (n_original, dim)
    X_generated : np.ndarray or torch.Tensor
        Generated samples, shape (n_generated, dim)
    test : str, default='ks'
        Type of test. Options: 'ks' (Kolmogorov-Smirnov)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'p_values': array of p-values for each dimension
        - 'min_p_value': minimum p-value across dimensions
        - 'mean_p_value': mean p-value across dimensions
        - 'num_rejected': number of dimensions where null is rejected at α=0.05
        
    Notes
    -----
    Higher p-values indicate better match (fail to reject null hypothesis).
    The test is performed separately for each dimension.
    """
    # Convert to numpy if needed
    if isinstance(X_original, torch.Tensor):
        X_original = X_original.cpu().numpy()
    if isinstance(X_generated, torch.Tensor):
        X_generated = X_generated.cpu().numpy()
    
    n_dims = X_original.shape[1]
    p_values = np.zeros(n_dims)
    
    if test == 'ks':
        # Kolmogorov-Smirnov test for each dimension
        for d in range(n_dims):
            statistic, p_value = ks_2samp(X_original[:, d], X_generated[:, d])
            p_values[d] = p_value
    else:
        raise ValueError(f"Unknown test type: {test}")
    
    # Count how many dimensions reject null at α=0.05
    num_rejected = np.sum(p_values < 0.05)
    
    return {
        'p_values': p_values,
        'min_p_value': float(np.min(p_values)),
        'mean_p_value': float(np.mean(p_values)),
        'num_rejected': int(num_rejected)
    }


def mean_and_covariance_distance(X_original, X_generated):
    """Compute distance between means and covariances of two distributions.
    
    Compares first and second moments of the distributions.
    
    Parameters
    ----------
    X_original : np.ndarray or torch.Tensor
        Original samples, shape (n_original, dim)
    X_generated : np.ndarray or torch.Tensor
        Generated samples, shape (n_generated, dim)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'mean_distance': L2 distance between means
        - 'cov_frobenius': Frobenius norm of covariance difference
        - 'cov_trace': Trace of absolute covariance difference
        
    Notes
    -----
    Lower values indicate better match for all metrics.
    """
    # Convert to numpy if needed
    if isinstance(X_original, torch.Tensor):
        X_original = X_original.cpu().numpy()
    if isinstance(X_generated, torch.Tensor):
        X_generated = X_generated.cpu().numpy()
    
    # Compute means
    mean_orig = np.mean(X_original, axis=0)
    mean_gen = np.mean(X_generated, axis=0)
    mean_distance = np.linalg.norm(mean_orig - mean_gen)
    
    # Compute covariances
    cov_orig = np.cov(X_original, rowvar=False)
    cov_gen = np.cov(X_generated, rowvar=False)
    
    # Handle 1D case where cov returns scalar
    if cov_orig.ndim == 0:
        cov_orig = np.array([[cov_orig]])
    if cov_gen.ndim == 0:
        cov_gen = np.array([[cov_gen]])
    
    cov_diff = cov_orig - cov_gen
    
    # Frobenius norm of covariance difference
    cov_frobenius = np.linalg.norm(cov_diff, 'fro')
    
    # Trace of absolute covariance difference
    cov_trace = np.abs(np.trace(cov_diff))
    
    return {
        'mean_distance': float(mean_distance),
        'cov_frobenius': float(cov_frobenius),
        'cov_trace': float(cov_trace)
    }


def compute_all_metrics(X_original, X_generated, mmd_kernel='rbf', 
                       mmd_gamma=1.0, test_type='ks', verbose=True):
    """Compute all available metrics for distribution comparison.
    
    Parameters
    ----------
    X_original : np.ndarray or torch.Tensor
        Original samples, shape (n_original, dim)
    X_generated : np.ndarray or torch.Tensor
        Generated samples, shape (n_generated, dim)
    mmd_kernel : str, default='rbf'
        Kernel type for MMD computation
    mmd_gamma : float, default=1.0
        Kernel bandwidth for MMD
    test_type : str, default='ks'
        Type of two-sample test to perform
    verbose : bool, default=True
        Whether to print results
        
    Returns
    -------
    dict
        Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Maximum Mean Discrepancy
    metrics['mmd'] = maximum_mean_discrepancy(
        X_original, X_generated, kernel=mmd_kernel, gamma=mmd_gamma
    )
    
    # Two-sample test p-values
    test_results = two_sample_test(X_original, X_generated, test=test_type)
    metrics['two_sample_test'] = test_results
    
    # Mean and covariance distances
    moment_metrics = mean_and_covariance_distance(X_original, X_generated)
    metrics['mean_distance'] = moment_metrics['mean_distance']
    metrics['cov_frobenius'] = moment_metrics['cov_frobenius']
    metrics['cov_trace'] = moment_metrics['cov_trace']
    
    if verbose:
        print("=" * 70)
        print("Distribution Comparison Metrics")
        print("=" * 70)
        print(f"Maximum Mean Discrepancy:     {metrics['mmd']:.6f}")
        print(f"\nTwo-Sample Test ({test_type.upper()}):")
        print(f"  Min p-value:                {test_results['min_p_value']:.6f}")
        print(f"  Mean p-value:               {test_results['mean_p_value']:.6f}")
        print(f"  Rejected at α=0.05:         {test_results['num_rejected']}/{len(test_results['p_values'])}")
        print(f"\nMoment Comparisons:")
        print(f"  Mean distance:              {metrics['mean_distance']:.6f}")
        print(f"  Covariance (Frobenius):     {metrics['cov_frobenius']:.6f}")
        print(f"  Covariance (Trace):         {metrics['cov_trace']:.6f}")
        print("=" * 70)
    
    return metrics