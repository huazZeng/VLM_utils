"""
Concrete counter implementations for distillation.
"""

from typing import Dict, Any, List, Optional
from .counter_base import BaseCounterManager


class EpochCounter(BaseCounterManager):
    """Counter that tracks the number of epochs/iterations."""
    
    def __init__(self, max_epochs: int = 100):
        """
        Initialize epoch counter.
        
        Args:
            max_epochs: Maximum number of epochs before stopping
        """
        self.max_epochs = max_epochs
        self.current_count = 0
    
    def should_increment(self, json_data: Dict[str, Any]) -> bool:
        """Always increment for epoch counter."""
        return True
    
    def increment(self) -> None:
        """Increment epoch counter."""
        self.current_count += 1
    
    def get_count(self) -> int:
        """Get current count."""
        return self.current_count
    
    def set_count(self, count: int) -> None:
        """Set counter to specific value."""
        self.current_count = count
    
    def should_stop(self) -> bool:
        """Check if maximum epochs reached."""
        return self.current_count >= self.max_epochs
    
    def reset(self) -> None:
        """Reset epoch counter."""
        self.current_count = 0


class QualityCounter(BaseCounterManager):
    """Counter that tracks quality-based conditions."""
    
    def __init__(self, quality_threshold: float = 0.8, max_count: int = 1000):
        """
        Initialize quality counter.
        
        Args:
            quality_threshold: Quality threshold to trigger increment
            max_count: Maximum count before stopping
        """
        self.quality_threshold = quality_threshold
        self.max_count = max_count
        self.current_count = 0
    
    def should_increment(self, json_data: Dict[str, Any]) -> bool:
        """
        Check if JSON data meets quality threshold.
        
        Args:
            json_data: JSON data to evaluate
            
        Returns:
            True if quality meets threshold
        """
        # Example: check if result has expected structure
        if not json_data or not isinstance(json_data, dict):
            return False
        
        # Check if result has content
        result = json_data.get('result')
        if result is None:
            return False
        
        # Simple quality check: if result has content, consider it good quality
        if isinstance(result, dict) and len(result) > 0:
            return True
        elif isinstance(result, list) and len(result) > 0:
            return True
        
        return False
    
    def increment(self) -> None:
        """Increment quality counter."""
        self.current_count += 1
    
    def get_count(self) -> int:
        """Get current count."""
        return self.current_count
    
    def set_count(self, count: int) -> None:
        """Set counter to specific value."""
        self.current_count = count
    
    def should_stop(self) -> bool:
        """Check if maximum count reached."""
        return self.current_count >= self.max_count
    
    def reset(self) -> None:
        """Reset quality counter."""
        self.current_count = 0


class ErrorCounter(BaseCounterManager):
    """Counter that tracks error conditions."""
    
    def __init__(self, max_errors: int = 50):
        """
        Initialize error counter.
        
        Args:
            max_errors: Maximum number of errors before stopping
        """
        self.max_errors = max_errors
        self.current_count = 0
    
    def should_increment(self, json_data: Dict[str, Any]) -> bool:
        """
        Check if JSON data indicates an error.
        
        Args:
            json_data: JSON data to evaluate
            
        Returns:
            True if error condition detected
        """
        # Check for error conditions
        if not json_data:
            return True
        
        # Check if result is None or empty
        result = json_data.get('result')
        if result is None:
            return True
        
        # Check if result is empty
        if isinstance(result, dict) and len(result) == 0:
            return True
        elif isinstance(result, list) and len(result) == 0:
            return True
        
        return False
    
    def increment(self) -> None:
        """Increment error counter."""
        self.current_count += 1
    
    def get_count(self) -> int:
        """Get current count."""
        return self.current_count
    
    def set_count(self, count: int) -> None:
        """Set counter to specific value."""
        self.current_count = count
    
    def should_stop(self) -> bool:
        """Check if maximum errors reached."""
        return self.current_count >= self.max_errors
    
    def reset(self) -> None:
        """Reset error counter."""
        self.current_count = 0


class SuccessCounter(BaseCounterManager):
    """Counter that tracks successful processing."""
    
    def __init__(self, min_success: int = 100):
        """
        Initialize success counter.
        
        Args:
            min_success: Minimum number of successes before stopping
        """
        self.min_success = min_success
        self.current_count = 0
    
    def should_increment(self, json_data: Dict[str, Any]) -> bool:
        """
        Check if JSON data indicates successful processing.
        
        Args:
            json_data: JSON data to evaluate
            
        Returns:
            True if successful processing detected
        """
        # Check for success conditions
        if not json_data:
            return False
        
        # Check if result exists and has content
        result = json_data.get('result')
        if result is None:
            return False
        
        # Check if result has meaningful content
        if isinstance(result, dict) and len(result) > 0:
            return True
        elif isinstance(result, list) and len(result) > 0:
            return True
        
        return False
    
    def increment(self) -> None:
        """Increment success counter."""
        self.current_count += 1
    
    def get_count(self) -> int:
        """Get current count."""
        return self.current_count
    
    def set_count(self, count: int) -> None:
        """Set counter to specific value."""
        self.current_count = count
    
    def should_stop(self) -> bool:
        """Check if minimum successes reached."""
        return self.current_count >= self.min_success
    
    def reset(self) -> None:
        """Reset success counter."""
        self.current_count = 0


class CustomCounter(BaseCounterManager):
    """Counter with custom condition function."""
    
    def __init__(self, condition_func, max_count: int = 1000):
        """
        Initialize custom counter.
        
        Args:
            condition_func: Function that takes json_data and returns bool
            max_count: Maximum count before stopping
        """
        self.condition_func = condition_func
        self.max_count = max_count
        self.current_count = 0
    
    def should_increment(self, json_data: Dict[str, Any]) -> bool:
        """
        Use custom condition function to determine increment.
        
        Args:
            json_data: JSON data to evaluate
            
        Returns:
            Result of custom condition function
        """
        return self.condition_func(json_data)
    
    def increment(self) -> None:
        """Increment custom counter."""
        self.current_count += 1
    
    def get_count(self) -> int:
        """Get current count."""
        return self.current_count
    
    def set_count(self, count: int) -> None:
        """Set counter to specific value."""
        self.current_count = count
    
    def should_stop(self) -> bool:
        """Check if maximum count reached."""
        return self.current_count >= self.max_count
    
    def reset(self) -> None:
        """Reset custom counter."""
        self.current_count = 0 