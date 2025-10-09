import pytest
import numpy as np
from scipy import stats
from src.dgps import (
    DataGenerator,
    NormalGenerator,
    ExponentialGenerator,
    UniformGenerator,
    LognormalGenerator,
    ChiSquareGenerator
)


class TestDataGeneratorInterface:
    """Test that all generators implement the required interface."""
    
    def test_normal_implements_interface(self):
        gen = NormalGenerator()
        assert isinstance(gen, DataGenerator)
        assert hasattr(gen, 'generate')
        assert hasattr(gen, 'name')
    
    def test_exponential_implements_interface(self):
        gen = ExponentialGenerator()
        assert isinstance(gen, DataGenerator)
        assert hasattr(gen, 'generate')
        assert hasattr(gen, 'name')
    
    def test_uniform_implements_interface(self):
        gen = UniformGenerator()
        assert isinstance(gen, DataGenerator)
        assert hasattr(gen, 'generate')
        assert hasattr(gen, 'name')
    
    def test_lognormal_implements_interface(self):
        gen = LognormalGenerator()
        assert isinstance(gen, DataGenerator)
        assert hasattr(gen, 'generate')
        assert hasattr(gen, 'name')
    
    def test_chisquare_implements_interface(self):
        gen = ChiSquareGenerator()
        assert isinstance(gen, DataGenerator)
        assert hasattr(gen, 'generate')
        assert hasattr(gen, 'name')


class TestNormalGenerator:
    """Test NormalGenerator functionality and statistical properties."""
    
    def test_generate_returns_correct_shape(self):
        gen = NormalGenerator()
        data = gen.generate(100)
        assert data.shape == (100,)
    
    def test_generate_with_seed_reproducibility(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        gen = NormalGenerator(loc=5, scale=2)
        data1 = gen.generate(100, rng=rng1)
        data2 = gen.generate(100, rng=rng2)
        np.testing.assert_array_equal(data1, data2)
    
    def test_name_property(self):
        gen = NormalGenerator(loc=5, scale=2)
        assert gen.name == "Normal(μ=5, σ=2)"
    
    def test_name_with_default_parameters(self):
        gen = NormalGenerator()
        assert gen.name == "Normal(μ=0, σ=1)"
    
    def test_statistical_properties(self):
        gen = NormalGenerator(loc=5, scale=2)
        data = gen.generate(10000)
        assert np.abs(np.mean(data) - 5) < 0.1
        assert np.abs(np.std(data) - 2) < 0.1
    
    def test_default_parameters(self):
        gen = NormalGenerator()
        data = gen.generate(10000)
        assert np.abs(np.mean(data) - 0) < 0.1
        assert np.abs(np.std(data) - 1) < 0.1
    
    def test_is_symmetric(self):
        gen = NormalGenerator(loc=0, scale=1)
        data = gen.generate(10000)
        skewness = stats.skew(data)
        assert np.abs(skewness) < 0.1  # Should be close to 0


class TestExponentialGenerator:
    """Test ExponentialGenerator functionality and statistical properties."""
    
    def test_generate_returns_correct_shape(self):
        gen = ExponentialGenerator()
        data = gen.generate(100)
        assert data.shape == (100,)
    
    def test_generate_all_positive(self):
        gen = ExponentialGenerator(scale=2)
        data = gen.generate(1000)
        assert np.all(data >= 0)
    
    def test_name_property(self):
        gen = ExponentialGenerator(scale=2)
        assert gen.name == "Exponential(λ=0.50)"
    
    def test_name_with_different_scale(self):
        gen = ExponentialGenerator(scale=0.5)
        assert gen.name == "Exponential(λ=2.00)"
    
    def test_name_with_default_parameters(self):
        gen = ExponentialGenerator()
        assert gen.name == "Exponential(λ=1.00)"
    
    def test_statistical_properties(self):
        gen = ExponentialGenerator(scale=2)
        data = gen.generate(10000)
        assert np.abs(np.mean(data) - 2) < 0.1
        assert np.abs(np.std(data) - 2) < 0.1
    
    def test_is_right_skewed(self):
        gen = ExponentialGenerator(scale=1)
        data = gen.generate(10000)
        skewness = stats.skew(data)
        assert skewness > 1.5  # Exponential has skewness of 2


class TestUniformGenerator:
    """Test UniformGenerator functionality and statistical properties."""
    
    def test_generate_returns_correct_shape(self):
        gen = UniformGenerator()
        data = gen.generate(100)
        assert data.shape == (100,)
    
    def test_data_within_bounds(self):
        gen = UniformGenerator(low=0, high=10)
        data = gen.generate(1000)
        assert np.all(data >= 0)
        assert np.all(data <= 10)
    
    def test_name_property(self):
        gen = UniformGenerator(low=0, high=10)
        assert gen.name == "Uniform(0, 10)"
    
    def test_name_with_default_parameters(self):
        gen = UniformGenerator()
        assert gen.name == "Uniform(0, 2)"
    
    def test_name_with_floats(self):
        gen = UniformGenerator(low=1.5, high=3.5)
        assert gen.name == "Uniform(1.5, 3.5)"
    
    def test_statistical_properties(self):
        gen = UniformGenerator(low=2, high=8)
        data = gen.generate(10000)
        expected_mean = 5.0
        expected_std = (8 - 2) / np.sqrt(12)
        assert np.abs(np.mean(data) - expected_mean) < 0.1
        assert np.abs(np.std(data) - expected_std) < 0.1
    
    def test_default_parameters(self):
        gen = UniformGenerator()
        data = gen.generate(10000)
        assert np.all(data >= 0)
        assert np.all(data <= 2)
        assert np.abs(np.mean(data) - 1.0) < 0.1
    
    def test_zero_skewness(self):
        gen = UniformGenerator(low=0, high=10)
        data = gen.generate(10000)
        skewness = stats.skew(data)
        assert np.abs(skewness) < 0.1  # Uniform should have ~0 skewness


class TestLognormalGenerator:
    """Test LognormalGenerator functionality and statistical properties."""
    
    def test_generate_returns_correct_shape(self):
        gen = LognormalGenerator()
        data = gen.generate(100)
        assert data.shape == (100,)
    
    def test_generate_all_positive(self):
        gen = LognormalGenerator(mean=0, sigma=1)
        data = gen.generate(1000)
        assert np.all(data > 0)
    
    def test_name_property(self):
        gen = LognormalGenerator(mean=0, sigma=1)
        assert gen.name == "Lognormal(μ=0, σ=1)"
    
    def test_name_with_different_parameters(self):
        gen = LognormalGenerator(mean=1.5, sigma=0.5)
        assert gen.name == "Lognormal(μ=1.5, σ=0.5)"
    
    def test_name_with_default_parameters(self):
        gen = LognormalGenerator()
        assert gen.name == "Lognormal(μ=0, σ=1)"
    
    def test_median_property(self):
        gen = LognormalGenerator(mean=1.5, sigma=1)
        data = gen.generate(10000)
        observed_median = np.median(data)
        expected_median = np.exp(1.5)
        assert np.abs(observed_median - expected_median) / expected_median < 0.1
    
    def test_is_right_skewed(self):
        gen = LognormalGenerator(mean=0, sigma=1)
        data = gen.generate(10000)
        skewness = stats.skew(data)
        assert skewness > 1.0  # Lognormal is right-skewed
    
    def test_default_parameters(self):
        gen = LognormalGenerator()
        data = gen.generate(10000)
        assert np.all(data > 0)
        expected_median = np.exp(0)
        observed_median = np.median(data)
        assert np.abs(observed_median - expected_median) < 0.2


class TestChiSquareGenerator:
    """Test ChiSquareGenerator functionality and statistical properties."""
    
    def test_generate_returns_correct_shape(self):
        gen = ChiSquareGenerator()
        data = gen.generate(100)
        assert data.shape == (100,)
    
    def test_generate_all_positive(self):
        gen = ChiSquareGenerator(df=5)
        data = gen.generate(1000)
        assert np.all(data >= 0)
    
    def test_name_property(self):
        gen = ChiSquareGenerator(df=10)
        assert gen.name == "ChiSquare(df=10)"
    
    def test_name_with_default_parameters(self):
        gen = ChiSquareGenerator()
        assert gen.name == "ChiSquare(df=5)"
    
    def test_statistical_properties(self):
        df = 10
        gen = ChiSquareGenerator(df=df)
        data = gen.generate(10000)
        assert np.abs(np.mean(data) - df) < 0.5
        assert np.abs(np.std(data) - np.sqrt(2 * df)) < 0.5
    
    def test_skewness_decreases_with_df(self):
        # Higher df -> less skewness
        gen_low_df = ChiSquareGenerator(df=2)
        gen_high_df = ChiSquareGenerator(df=20)
        data_low = gen_low_df.generate(5000)
        data_high = gen_high_df.generate(5000)
        skew_low = stats.skew(data_low)
        skew_high = stats.skew(data_high)
        assert skew_low > skew_high


class TestGeneratorInterchangeability:
    """Test that generators are interchangeable in simulation workflows."""
    
    def test_all_generators_have_same_interface(self):
        generators = [
            NormalGenerator(),
            ExponentialGenerator(),
            UniformGenerator(),
            LognormalGenerator(),
            ChiSquareGenerator()
        ]
        
        for gen in generators:
            data = gen.generate(100)
            assert isinstance(data, np.ndarray)
            assert len(data) == 100
            
            name = gen.name
            assert isinstance(name, str)
            assert len(name) > 0
    
    def test_rng_parameter_consistency(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        
        gen1 = NormalGenerator()
        gen2 = ExponentialGenerator()
        
        data1_run1 = gen1.generate(100, rng=rng1)
        data2_run1 = gen2.generate(100, rng=rng2)
        
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        
        data1_run2 = gen1.generate(100, rng=rng1)
        data2_run2 = gen2.generate(100, rng=rng2)
        
        np.testing.assert_array_equal(data1_run1, data1_run2)
        np.testing.assert_array_equal(data2_run1, data2_run2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_generate_one_sample(self):
        generators = [
            NormalGenerator(),
            ExponentialGenerator(),
            UniformGenerator(),
            LognormalGenerator(),
            ChiSquareGenerator()
        ]
        
        for gen in generators:
            data = gen.generate(1)
            assert len(data) == 1
            assert isinstance(data[0], (float, np.floating))
    
    def test_generate_large_sample(self):
        gen = NormalGenerator()
        data = gen.generate(100000)
        assert len(data) == 100000
    
    def test_uniform_with_equal_bounds(self):
        gen = UniformGenerator(low=5, high=5)
        data = gen.generate(100)
        assert np.all(data == 5)
    
    def test_exponential_small_scale(self):
        gen = ExponentialGenerator(scale=0.01)
        data = gen.generate(1000)
        assert np.all(data >= 0)
        assert np.abs(np.mean(data) - 0.01) < 0.005
    
    def test_normal_with_large_variance(self):
        gen = NormalGenerator(loc=0, scale=100)
        data = gen.generate(10000)
        assert np.abs(np.std(data) - 100) < 5
    
    def test_chisquare_df_one(self):
        gen = ChiSquareGenerator(df=1)
        data = gen.generate(1000)
        assert np.all(data >= 0)
        assert np.abs(np.mean(data) - 1) < 0.3