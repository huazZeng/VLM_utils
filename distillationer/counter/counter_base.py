"""
Base classes for counter management with auto-registration.
"""

from typing import Dict, Any, List, Callable
from abc import ABC, abstractmethod


class BaseCounterManager(ABC):
    """Abstract base class for counter managers with auto-registration."""
    
    # Registry for all counter managers
    _registry = {}
    
    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        # Register by class name (e.g., EpochCounter -> epoch)
        class_name = cls.__name__
        
        # Convert class name to registry key
        registry_key = cls._convert_class_name_to_key(class_name)
        cls._registry[registry_key] = cls
    
    @staticmethod
    def _convert_class_name_to_key(class_name: str) -> str:
        """Convert class name to registry key."""
        # Remove 'Counter' suffix
        name = class_name.replace('Counter', '')
        
        # Convert camelCase to snake_case
        import re
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the counter with configuration parameters."""
        pass
    
    @abstractmethod
    def should_increment(self, json_data: Dict[str, Any]) -> bool:
        """
        Determine if counter should be incremented based on JSON data.
        
        Args:
            json_data: JSON data to evaluate
            
        Returns:
            True if counter should be incremented
        """
        pass
    
    @abstractmethod
    def increment(self) -> None:
        """Increment the counter."""
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """Get current count."""
        pass
    
    @abstractmethod
    def set_count(self, count: int) -> None:
        """Set counter to specific value."""
        pass
    
    @abstractmethod
    def should_stop(self) -> bool:
        """Check if distillation should stop based on counter state."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset counter to initial state."""
        pass
    
    @classmethod
    def get_registry(cls) -> dict:
        """Get the registry of all counter manager classes."""
        return cls._registry.copy()
    
    @classmethod
    def create_counter(cls, counter_type: str, **kwargs) -> 'BaseCounterManager':
        """Create counter instance based on counter type."""
        registry = cls.get_registry()
        counter_class = registry.get(counter_type)
        
        if counter_class is None:
            # Get available counters for error message
            available_counters = list(registry.keys())
            error_msg = f"No counter found for: {counter_type}\n"
            error_msg += f"Available counters: {available_counters}"
            raise ValueError(error_msg)
        
        return counter_class(**kwargs)
    
    @classmethod
    def get_available_counters(cls) -> list:
        """Get list of available counter types."""
        return list(cls.get_registry().keys()) 