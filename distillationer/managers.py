"""
Manager classes for coordinating counters and samplers.
"""

from typing import Dict, Any, List, Optional, Union
from .counter import BaseCounterManager
from .sampler import BaseSamplerManager


class CounterManager:
    """Manager class for coordinating multiple counters."""
    
    def __init__(self, counters: Optional[List[Union[str, BaseCounterManager]]] = None, 
                 counter_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize counter manager.
        
        Args:
            counters: List of counter types or counter instances
            counter_configs: Configuration for each counter type
        """
        self.counters = []
        self.counter_configs = counter_configs or {}
        
        if counters:
            for counter in counters:
                if isinstance(counter, str):
                    # Create counter from type string
                    config = self.counter_configs.get(counter, {})
                    counter_instance = BaseCounterManager.create_counter(counter, **config)
                    self.counters.append(counter_instance)
                elif isinstance(counter, BaseCounterManager):
                    # Use provided counter instance
                    self.counters.append(counter)
                else:
                    raise ValueError(f"Invalid counter: {counter}")
    
    def add_counter(self, counter_type: str, **kwargs) -> None:
        """
        Add a new counter.
        
        Args:
            counter_type: Type of counter to add
            **kwargs: Configuration parameters for the counter
        """
        counter_instance = BaseCounterManager.create_counter(counter_type, **kwargs)
        self.counters.append(counter_instance)
    
    def process_json(self, json_data: Dict[str, Any]) -> None:
        """
        Process JSON data through all counters.
        
        Args:
            json_data: JSON data to evaluate
        """
        for counter in self.counters:
            if counter.should_increment(json_data):
                counter.increment()
    
    def should_stop(self) -> bool:
        """
        Check if any counter indicates stopping.
        
        Returns:
            True if any counter indicates stopping
        """
        return any(counter.should_stop() for counter in self.counters)
    
    def reset_all(self) -> None:
        """Reset all counters to initial state."""
        for counter in self.counters:
            counter.reset()
    
    def get_all_counts(self) -> Dict[str, int]:
        """
        Get current counts of all counters.
        
        Returns:
            Dictionary with counter names and their counts
        """
        counts = {}
        for i, counter in enumerate(self.counters):
            counter_name = f"counter_{i}_{counter.__class__.__name__}"
            counts[counter_name] = counter.get_count()
        return counts
    
    def get_stopping_counters(self) -> List[str]:
        """
        Get list of counters that indicate stopping.
        
        Returns:
            List of counter names that should stop
        """
        stopping_counters = []
        for i, counter in enumerate(self.counters):
            if counter.should_stop():
                counter_name = f"counter_{i}_{counter.__class__.__name__}"
                stopping_counters.append(counter_name)
        return stopping_counters
    
    @staticmethod
    def get_available_counters() -> List[str]:
        """Get list of available counter types."""
        return BaseCounterManager.get_available_counters()


class SamplerManager:
    """Manager class for coordinating samplers."""
    
    def __init__(self, sampler_type: str, **kwargs):
        """
        Initialize sampler manager.
        
        Args:
            sampler_type: Type of sampler to use
            **kwargs: Configuration parameters for the sampler
        """
        self.sampler = BaseSamplerManager.create_sampler(sampler_type, **kwargs)
        self.sampler_type = sampler_type
    
    def sample(self, data: List[Any], **kwargs) -> List[Any]:
        """
        Sample from data using the configured sampler.
        
        Args:
            data: List of data items to sample from
            **kwargs: Additional sampling parameters
            
        Returns:
            List of sampled items
        """
        return self.sampler.sample(data, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current sampler status.
        
        Returns:
            Dictionary with sampler status
        """
        return self.sampler.get_status()
    
    def change_sampler(self, sampler_type: str, **kwargs) -> None:
        """
        Change to a different sampler type.
        
        Args:
            sampler_type: New sampler type
            **kwargs: Configuration parameters for the new sampler
        """
        self.sampler = BaseSamplerManager.create_sampler(sampler_type, **kwargs)
        self.sampler_type = sampler_type
    
    @staticmethod
    def get_available_samplers() -> List[str]:
        """Get list of available sampler types."""
        return BaseSamplerManager.get_available_samplers() 