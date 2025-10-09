"""Tests for metrics module.

This module tests the distribution comparison metrics to ensure
they work correctly and produce expected results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import torch
from metrics import (
    maximum_mean_discrepancy,
    two_sample_test,
    mean_and_covariance_distance,
    compute_all_metrics
)


class TestMaximumMeanDiscrepancy:
    """Test MMD computation."""
    
    def test_mmd_identical_distributions(self):
        """MMD should be close to 0 for identical samples."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        
        mmd = maximum_mean_discrepancy(X, X)
        
        assert mmd >= 0
        assert mmd < 0.01  # Should be very small
    
    def test_mmd_same_distribution(self):
        """MMD should be small for samples from same distribution."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        np.random.seed(123)
        X2 = np.random.randn(100, 2)
        
        mmd = maximum_mean_discrepancy(X1, X2)
        
        assert mmd >= 0
        assert mmd < 0.3  # Should be relatively small
    
    def test_mmd_different_distributions(self):
        """MMD should be larger for samples from different distributions."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2) + 5  # Shifted mean
        
        mmd = maximum_mean_discrepancy(X1, X2)
        
        assert mmd > 0.5  # Should be noticeably larger
    
    def test_mmd_with_torch_tensors(self):
        """MMD should work with torch tensors."""
        torch.manual_seed(42)
        X1 = torch.randn(100, 2)
        X2 = torch.randn(100, 2)
        
        mmd = maximum_mean_discrepancy(X1, X2)
        
        assert isinstance(mmd, float)
        assert mmd >= 0
    
    def test_mmd_rbf_kernel(self):
        """Test MMD with RBF kernel."""
        np.random.seed(42)
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2)
        
        mmd = maximum_mean_discrepancy(X1, X2, kernel='rbf', gamma=1.0)
        
        assert mmd >= 0
    
    def test_mmd_linear_kernel(self):
        """Test MMD with linear kernel."""
        np.random.seed(42)
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2)
        
        mmd = maximum_mean_discrepancy(X1, X2, kernel='linear')
        
        assert mmd >= 0
    
    def test_mmd_different_gammas(self):
        """Test that gamma affects MMD computation."""
        np.random.seed(42)
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2) + 1
        
        mmd_gamma_small = maximum_mean_discrepancy(X1, X2, kernel='rbf', gamma=0.1)
        mmd_gamma_large = maximum_mean_discrepancy(X1, X2, kernel='rbf', gamma=10.0)
        
        # Different gammas should give different results
        assert mmd_gamma_small != mmd_gamma_large
    
    def test_mmd_invalid_kernel(self):
        """Test that invalid kernel raises error."""
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2)
        
        with pytest.raises(ValueError):
            maximum_mean_discrepancy(X1, X2, kernel='invalid')
    
    def test_mmd_high_dimensional(self):
        """Test MMD with high-dimensional data."""
        np.random.seed(42)
        X1 = np.random.randn(50, 10)
        X2 = np.random.randn(50, 10)
        
        mmd = maximum_mean_discrepancy(X1, X2)
        
        assert mmd >= 0


class TestTwoSampleTest:
    """Test two-sample testing functionality."""
    
    def test_two_sample_test_returns_dict(self):
        """Test that function returns dictionary with expected keys."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2)
        
        result = two_sample_test(X1, X2)
        
        assert isinstance(result, dict)
        assert 'p_values' in result
        assert 'min_p_value' in result
        assert 'mean_p_value' in result
        assert 'num_rejected' in result
    
    def test_two_sample_test_p_values_range(self):
        """Test that p-values are in valid range [0, 1]."""
        np.random.seed(42)
        X1 = np.random.randn(100, 3)
        X2 = np.random.randn(100, 3)
        
        result = two_sample_test(X1, X2)
        
        assert np.all(result['p_values'] >= 0)
        assert np.all(result['p_values'] <= 1)
        assert 0 <= result['min_p_value'] <= 1
        assert 0 <= result['mean_p_value'] <= 1
    
    def test_two_sample_test_same_distribution(self):
        """Test should have high p-values for same distribution."""
        np.random.seed(42)
        X1 = np.random.randn(200, 2)
        np.random.seed(123)
        X2 = np.random.randn(200, 2)
        
        result = two_sample_test(X1, X2)
        
        # Most dimensions should not reject null
        assert result['mean_p_value'] > 0.05
    
    def test_two_sample_test_different_distributions(self):
        """Test should have low p-values for different distributions."""
        np.random.seed(42)
        X1 = np.random.randn(200, 2)
        X2 = np.random.randn(200, 2) + 3  # Large shift
        
        result = two_sample_test(X1, X2)
        
        # Should reject null for most/all dimensions
        assert result['num_rejected'] > 0
        assert result['min_p_value'] < 0.05
    
    def test_two_sample_test_num_rejected(self):
        """Test that num_rejected is correct."""
        np.random.seed(42)
        X1 = np.random.randn(100, 5)
        X2 = np.random.randn(100, 5)
        
        result = two_sample_test(X1, X2)
        
        # Manually count rejections
        expected_rejections = np.sum(result['p_values'] < 0.05)
        assert result['num_rejected'] == expected_rejections
    
    def test_two_sample_test_with_torch_tensors(self):
        """Test with torch tensors."""
        torch.manual_seed(42)
        X1 = torch.randn(100, 2)
        X2 = torch.randn(100, 2)
        
        result = two_sample_test(X1, X2)
        
        assert isinstance(result, dict)
        assert len(result['p_values']) == 2
    
    def test_two_sample_test_invalid_test_type(self):
        """Test that invalid test type raises error."""
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2)
        
        with pytest.raises(ValueError):
            two_sample_test(X1, X2, test='invalid')
    
    def test_two_sample_test_1d(self):
        """Test with 1D data."""
        np.random.seed(42)
        X1 = np.random.randn(100, 1)
        X2 = np.random.randn(100, 1)
        
        result = two_sample_test(X1, X2)
        
        assert len(result['p_values']) == 1
        assert result['num_rejected'] <= 1


class TestMeanAndCovarianceDistance:
    """Test mean and covariance distance computation."""
    
    def test_mean_cov_distance_identical(self):
        """Test with identical samples."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        
        result = mean_and_covariance_distance(X, X)
        
        assert result['mean_distance'] < 1e-10
        assert result['cov_frobenius'] < 1e-10
        assert result['cov_trace'] < 1e-10
    
    def test_mean_cov_distance_returns_dict(self):
        """Test that function returns dictionary with expected keys."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2)
        
        result = mean_and_covariance_distance(X1, X2)
        
        assert isinstance(result, dict)
        assert 'mean_distance' in result
        assert 'cov_frobenius' in result
        assert 'cov_trace' in result
    
    def test_mean_cov_distance_all_positive(self):
        """Test that all distances are non-negative."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2)
        
        result = mean_and_covariance_distance(X1, X2)
        
        assert result['mean_distance'] >= 0
        assert result['cov_frobenius'] >= 0
        assert result['cov_trace'] >= 0
    
    def test_mean_cov_distance_shifted_mean(self):
        """Test with shifted mean."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2) + 2
        
        result = mean_and_covariance_distance(X1, X2)
        
        # Mean distance should be large
        assert result['mean_distance'] > 1.0
        # Covariance should be similar
        assert result['cov_frobenius'] < 1.0
    
    def test_mean_cov_distance_different_variance(self):
        """Test with different variance."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2) * 2  # Double the variance
        
        result = mean_and_covariance_distance(X1, X2)
        
        # Covariance distance should be large
        assert result['cov_frobenius'] > 0.5
    
    def test_mean_cov_distance_with_torch_tensors(self):
        """Test with torch tensors."""
        torch.manual_seed(42)
        X1 = torch.randn(100, 2)
        X2 = torch.randn(100, 2)
        
        result = mean_and_covariance_distance(X1, X2)
        
        assert isinstance(result['mean_distance'], float)
        assert isinstance(result['cov_frobenius'], float)
        assert isinstance(result['cov_trace'], float)
    
    def test_mean_cov_distance_1d(self):
        """Test with 1D data."""
        np.random.seed(42)
        X1 = np.random.randn(100, 1)
        X2 = np.random.randn(100, 1)
        
        result = mean_and_covariance_distance(X1, X2)
        
        assert result['mean_distance'] >= 0
        assert result['cov_frobenius'] >= 0
        assert result['cov_trace'] >= 0
    
    def test_mean_cov_distance_high_dimensional(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        X1 = np.random.randn(100, 10)
        X2 = np.random.randn(100, 10)
        
        result = mean_and_covariance_distance(X1, X2)
        
        assert result['mean_distance'] >= 0
        assert result['cov_frobenius'] >= 0
        assert result['cov_trace'] >= 0


class TestComputeAllMetrics:
    """Test the compute_all_metrics convenience function."""
    
    def test_compute_all_metrics_returns_dict(self):
        """Test that function returns dictionary."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2)
        
        metrics = compute_all_metrics(X1, X2, verbose=False)
        
        assert isinstance(metrics, dict)
    
    def test_compute_all_metrics_has_all_keys(self):
        """Test that all expected metrics are present."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2)
        
        metrics = compute_all_metrics(X1, X2, verbose=False)
        
        assert 'mmd' in metrics
        assert 'two_sample_test' in metrics
        assert 'mean_distance' in metrics
        assert 'cov_frobenius' in metrics
        assert 'cov_trace' in metrics
    
    def test_compute_all_metrics_verbose_false(self, capsys):
        """Test that verbose=False produces no output."""
        np.random.seed(42)
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2)
        
        metrics = compute_all_metrics(X1, X2, verbose=False)
        
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_compute_all_metrics_verbose_true(self, capsys):
        """Test that verbose=True produces output."""
        np.random.seed(42)
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2)
        
        metrics = compute_all_metrics(X1, X2, verbose=True)
        
        captured = capsys.readouterr()
        assert "Distribution Comparison Metrics" in captured.out
        assert "Maximum Mean Discrepancy" in captured.out
        assert "Two-Sample Test" in captured.out
    
    def test_compute_all_metrics_custom_mmd_params(self):
        """Test with custom MMD parameters."""
        np.random.seed(42)
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2)
        
        metrics = compute_all_metrics(
            X1, X2, 
            mmd_kernel='linear', 
            mmd_gamma=2.0, 
            verbose=False
        )
        
        assert 'mmd' in metrics
        assert metrics['mmd'] >= 0
    
    def test_compute_all_metrics_with_torch(self):
        """Test with torch tensors."""
        torch.manual_seed(42)
        X1 = torch.randn(50, 2)
        X2 = torch.randn(50, 2)
        
        metrics = compute_all_metrics(X1, X2, verbose=False)
        
        assert isinstance(metrics, dict)
        assert len(metrics) >= 5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_sample_size(self):
        """Test with small sample sizes."""
        np.random.seed(42)
        X1 = np.random.randn(10, 2)
        X2 = np.random.randn(10, 2)
        
        mmd = maximum_mean_discrepancy(X1, X2)
        result = two_sample_test(X1, X2)
        moment = mean_and_covariance_distance(X1, X2)
        
        assert mmd >= 0
        assert len(result['p_values']) == 2
        assert moment['mean_distance'] >= 0
    
    def test_single_sample(self):
        """Test with single sample (edge case)."""
        X1 = np.random.randn(1, 2)
        X2 = np.random.randn(1, 2)
        
        # MMD should still work
        mmd = maximum_mean_discrepancy(X1, X2)
        assert mmd >= 0
    
    def test_large_dimensional(self):
        """Test with large number of dimensions."""
        np.random.seed(42)
        X1 = np.random.randn(50, 20)
        X2 = np.random.randn(50, 20)
        
        metrics = compute_all_metrics(X1, X2, verbose=False)
        
        assert len(metrics['two_sample_test']['p_values']) == 20
    
    def test_different_sample_sizes(self):
        """Test with different sample sizes for X1 and X2."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(50, 2)
        
        metrics = compute_all_metrics(X1, X2, verbose=False)
        
        assert metrics['mmd'] >= 0
        assert len(metrics['two_sample_test']['p_values']) == 2