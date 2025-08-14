"""
Distillationer package for data distillation using VL models.
"""

from .distillationer import Distillationer
from .managers import CounterManager, SamplerManager

# Import from counter subpackage
from .counter import (
    BaseCounterManager,
    EpochCounter,
    QualityCounter,
    ErrorCounter,
    SuccessCounter,
    CustomCounter
)

# Import from sampler subpackage
from .sampler import (
    BaseSamplerManager,
    RandomSampler,
    QualitySampler,
    DiversitySampler,
    StratifiedSampler,
    UncertaintySampler
)

__all__ = [
    # Main classes
    'Distillationer',
    'CounterManager',
    'SamplerManager',
    
    # Base classes
    'BaseCounterManager',
    'BaseSamplerManager',
    
    # Counter implementations
    'EpochCounter',
    'QualityCounter',
    'ErrorCounter',
    'SuccessCounter',
    'CustomCounter',
    
    # Sampler implementations
    'RandomSampler',
    'QualitySampler',
    'DiversitySampler',
    'StratifiedSampler',
    'UncertaintySampler',
]

__version__ = "0.1.0" 