"""Data generation classes for simulation studies.

This module defines an abstract base class for data generators and provides
concrete implementations for common distributions.
"""

import numpy as np
from abc import ABC, abstractmethod


class DataGenerator(ABC):
    """Abstract base class for data generation.
    
    All data generators must implement three methods:
    - generate(n): Create n samples from the distribution
    - name(): Return a descriptive name for reporting
    - null_value(): Return the true parameter value under H0
    
    This ensures all generators can be used interchangeably in simulation
    studies without modifying the simulation code.
    """
    
    @abstractmethod
    def generate(self, n, rng=None):
        """Generate n samples from the distribution.
        
        Parameters
        ----------
        n : int
            Number of samples to generate
        rng : np.random.Generator, optional
            Random number generator. If None, uses np.random.default_rng (no seed) 
        
        Returns
        -------
        np.ndarray
            Array of n samples from the distribution
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return descriptive name of the distribution.
        
        Returns
        -------
        str
            Human-readable name including parameter values
        """
        pass


class NormalGenerator(DataGenerator):
    """Generate data from a normal distribution.
    
    Parameters
    ----------
    loc : float, default=0
        Mean of the distribution (location parameter)
    scale : float, default=1
        Standard deviation of the distribution (scale parameter)
    
    Examples
    --------
    >>> gen = NormalGenerator(loc=5, scale=2)
    >>> data = gen.generate(100)
    >>> gen.name()
    'Normal(μ=5, σ=2)'
    """
    
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale
    
    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.loc, self.scale, n)
    
    @property
    def name(self):
        return f"Normal(μ={self.loc}, σ={self.scale})"


class ExponentialGenerator(DataGenerator):
    """Generate data from an exponential distribution.
    
    Parameters
    ----------
    scale : float, default=1
        Scale parameter, which equals both the mean and standard deviation.
        The rate parameter λ = 1/scale.
    
    Examples
    --------
    >>> gen = ExponentialGenerator(scale=2)
    >>> data = gen.generate(100)
    >>> gen.name()
    'Exponential(λ=0.50)'
    """
    
    def __init__(self, scale=1):
        self.scale = scale
    
    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.exponential(self.scale, n)
    
    @property
    def name(self):
        return f"Exponential(λ={1/self.scale:.2f})"


class UniformGenerator(DataGenerator):
    """Generate data from a uniform distribution.
    
    Parameters
    ----------
    low : float, default=0
        Lower bound of the distribution
    high : float, default=2
        Upper bound of the distribution
    
    Examples
    --------
    >>> gen = UniformGenerator(low=0, high=10)
    >>> data = gen.generate(100)
    >>> gen.name
    'Uniform(0, 10)'
    """
    
    def __init__(self, low=0, high=2):
        self.low = low
        self.high = high
    
    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.low, self.high, n)
    
    @property
    def name(self):
        return f"Uniform({self.low}, {self.high})"


class LognormalGenerator(DataGenerator):
    """Generate data from a lognormal distribution.
    
    Parameters
    ----------
    mean : float, default=0
        Mean of the underlying normal distribution (not the lognormal mean)
    sigma : float, default=1
        Standard deviation of the underlying normal distribution
    
    Notes
    -----
    The mean of the lognormal distribution is exp(mean + sigma²/2).
    The median is exp(mean), which we use as the null value for location tests.
    
    Examples
    --------
    >>> gen = LognormalGenerator(mean=0, sigma=1)
    >>> data = gen.generate(100)
    >>> gen.name
    'Lognormal(μ=0, σ=1)'
    """
    
    def __init__(self, mean=0, sigma=1):
        self.mean = mean
        self.sigma = sigma
    
    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.lognormal(self.mean, self.sigma, n)
    
    @property
    def name(self):
        return f"Lognormal(μ={self.mean}, σ={self.sigma})"


class ChiSquareGenerator(DataGenerator):
    """Generate data from a chi-square distribution.
    
    Parameters
    ----------
    df : int, default=5
        Degrees of freedom, which equals the mean of the distribution
    
    Examples
    --------
    >>> gen = ChiSquareGenerator(df=10)
    >>> data = gen.generate(100)
    >>> gen.name()
    'ChiSquare(df=10)'
    """
    
    def __init__(self, df=5):
        self.df = df
    
    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.chisquare(self.df, n)
    
    @property
    def name(self):
        return f"ChiSquare(df={self.df})"