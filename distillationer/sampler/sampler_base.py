"""
Base classes for sampler management with auto-registration.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class BaseSamplerManager(ABC):
    """Abstract base class for sampler managers with auto-registration."""
    
    # Registry for all sampler managers
    _registry = {}
    
    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        # Register by class name (e.g., RandomSampler -> random)
        class_name = cls.__name__
        
        # Convert class name to registry key
        registry_key = cls._convert_class_name_to_key(class_name)
        cls._registry[registry_key] = cls
    
    @staticmethod
    def _convert_class_name_to_key(class_name: str) -> str:
        """Convert class name to registry key."""
        # Remove 'Sampler' suffix
        name = class_name.replace('Sampler', '')
        
        # Convert camelCase to snake_case
        import re
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the sampler with configuration parameters."""
        pass
    
    @abstractmethod
    def sample(self, data: List[Any], **kwargs) -> List[Any]:
        """
        Sample from the given data.
        
        Args:
            data: List of data items to sample from
            **kwargs: Additional sampling parameters
            
        Returns:
            List of sampled items
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current sampler status."""
        pass
    
    @classmethod
    def get_registry(cls) -> dict:
        """Get the registry of all sampler manager classes."""
        return cls._registry.copy()
    
    @classmethod
    def create_sampler(cls, sampler_type: str, **kwargs) -> 'BaseSamplerManager':
        """Create sampler instance based on sampler type."""
        registry = cls.get_registry()
        sampler_class = registry.get(sampler_type)
        
        if sampler_class is None:
            # Get available samplers for error message
            available_samplers = list(registry.keys())
            error_msg = f"No sampler found for: {sampler_type}\n"
            error_msg += f"Available samplers: {available_samplers}"
            raise ValueError(error_msg)
        
        return sampler_class(**kwargs)
    
    @classmethod
    def get_available_samplers(cls) -> list:
        """Get list of available sampler types."""
        return list(cls.get_registry().keys()) 