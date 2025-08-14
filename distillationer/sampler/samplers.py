"""
Concrete sampler implementations for distillation.
"""

import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .sampler_base import BaseSamplerManager


class RandomSampler(BaseSamplerManager):
    """Random sampling strategy."""
    
    def __init__(self, sample_size: int = 100, seed: Optional[int] = None):
        """
        Initialize random sampler.
        
        Args:
            sample_size: Number of samples to select
            seed: Random seed for reproducibility
        """
        self.sample_size = sample_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def sample(self, data: List[Any], **kwargs) -> List[Any]:
        """
        Randomly sample from data.
        
        Args:
            data: List of data items to sample from
            **kwargs: Additional parameters (ignored for random sampling)
            
        Returns:
            List of randomly sampled items
        """
        if len(data) <= self.sample_size:
            return data.copy()
        
        return random.sample(data, self.sample_size)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sampler status."""
        return {
            "sampler_type": "random",
            "sample_size": self.sample_size
        }


class QualitySampler(BaseSamplerManager):
    """Quality-based sampling strategy."""
    
    def __init__(self, sample_size: int = 100, quality_key: str = "quality", reverse: bool = False):
        """
        Initialize quality sampler.
        
        Args:
            sample_size: Number of samples to select
            quality_key: Key to access quality score in data items
            reverse: If True, select lowest quality items (for error cases)
        """
        self.sample_size = sample_size
        self.quality_key = quality_key
        self.reverse = reverse
    
    def sample(self, data: List[Any], **kwargs) -> List[Any]:
        """
        Sample based on quality scores.
        
        Args:
            data: List of data items with quality scores
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of quality-sampled items
        """
        if len(data) <= self.sample_size:
            return data.copy()
        
        # Sort by quality score
        sorted_data = sorted(data, key=lambda x: x.get(self.quality_key, 0), reverse=not self.reverse)
        
        return sorted_data[:self.sample_size]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sampler status."""
        return {
            "sampler_type": "quality",
            "sample_size": self.sample_size,
            "quality_key": self.quality_key,
            "reverse": self.reverse
        }


class DiversitySampler(BaseSamplerManager):
    """Diversity-based sampling strategy using clustering."""
    
    def __init__(self, sample_size: int = 100, feature_key: str = "features", method: str = "kmeans"):
        """
        Initialize diversity sampler.
        
        Args:
            sample_size: Number of samples to select
            feature_key: Key to access feature vector in data items
            method: Clustering method ('kmeans', 'random_centroids')
        """
        self.sample_size = sample_size
        self.feature_key = feature_key
        self.method = method
    
    def sample(self, data: List[Any], **kwargs) -> List[Any]:
        """
        Sample based on diversity using clustering.
        
        Args:
            data: List of data items with feature vectors
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of diverse sampled items
        """
        if len(data) <= self.sample_size:
            return data.copy()
        
        # Extract features
        features = []
        valid_indices = []
        
        for i, item in enumerate(data):
            if self.feature_key in item:
                features.append(item[self.feature_key])
                valid_indices.append(i)
        
        if len(features) < self.sample_size:
            # Fall back to random sampling if not enough features
            return random.sample(data, min(self.sample_size, len(data)))
        
        features = np.array(features)
        
        if self.method == "kmeans":
            return self._kmeans_sampling(data, features, valid_indices)
        elif self.method == "random_centroids":
            return self._random_centroids_sampling(data, features, valid_indices)
        else:
            raise ValueError(f"Unknown diversity sampling method: {self.method}")
    
    def _kmeans_sampling(self, data: List[Any], features: np.ndarray, valid_indices: List[int]) -> List[Any]:
        """K-means based diversity sampling."""
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=self.sample_size, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            # Select one sample from each cluster
            selected_indices = []
            for cluster_id in range(self.sample_size):
                cluster_samples = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                if cluster_samples:
                    selected_indices.append(valid_indices[cluster_samples[0]])
            
            return [data[i] for i in selected_indices]
            
        except ImportError:
            # Fall back to random sampling if sklearn not available
            return random.sample(data, self.sample_size)
    
    def _random_centroids_sampling(self, data: List[Any], features: np.ndarray, valid_indices: List[int]) -> List[Any]:
        """Random centroids based diversity sampling."""
        # Select random centroids
        centroid_indices = random.sample(range(len(features)), min(self.sample_size, len(features)))
        centroids = features[centroid_indices]
        
        # Assign each sample to nearest centroid
        selected_indices = set()
        for i, feature in enumerate(features):
            if len(selected_indices) >= self.sample_size:
                break
            
            distances = [np.linalg.norm(feature - centroid) for centroid in centroids]
            nearest_centroid_idx = np.argmin(distances)
            
            if nearest_centroid_idx < len(centroid_indices):
                selected_indices.add(valid_indices[centroid_indices[nearest_centroid_idx]])
        
        return [data[i] for i in list(selected_indices)[:self.sample_size]]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sampler status."""
        return {
            "sampler_type": "diversity",
            "sample_size": self.sample_size,
            "feature_key": self.feature_key,
            "method": self.method
        }


class StratifiedSampler(BaseSamplerManager):
    """Stratified sampling strategy based on categories."""
    
    def __init__(self, sample_size: int = 100, category_key: str = "category", proportional: bool = True):
        """
        Initialize stratified sampler.
        
        Args:
            sample_size: Number of samples to select
            category_key: Key to access category in data items
            proportional: If True, sample proportionally to category sizes
        """
        self.sample_size = sample_size
        self.category_key = category_key
        self.proportional = proportional
    
    def sample(self, data: List[Any], **kwargs) -> List[Any]:
        """
        Stratified sampling based on categories.
        
        Args:
            data: List of data items with categories
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of stratified sampled items
        """
        if len(data) <= self.sample_size:
            return data.copy()
        
        # Group data by category
        categories = {}
        for item in data:
            category = item.get(self.category_key, "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        
        selected_samples = []
        
        if self.proportional:
            # Proportional sampling
            total_items = len(data)
            for category, items in categories.items():
                category_ratio = len(items) / total_items
                category_sample_size = max(1, int(self.sample_size * category_ratio))
                selected_samples.extend(random.sample(items, min(category_sample_size, len(items))))
        else:
            # Equal sampling from each category
            samples_per_category = max(1, self.sample_size // len(categories))
            for category, items in categories.items():
                selected_samples.extend(random.sample(items, min(samples_per_category, len(items))))
        
        # If we have too many samples, randomly select the required number
        if len(selected_samples) > self.sample_size:
            selected_samples = random.sample(selected_samples, self.sample_size)
        
        return selected_samples
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sampler status."""
        return {
            "sampler_type": "stratified",
            "sample_size": self.sample_size,
            "category_key": self.category_key,
            "proportional": self.proportional
        }


class UncertaintySampler(BaseSamplerManager):
    """Uncertainty-based sampling strategy."""
    
    def __init__(self, sample_size: int = 100, uncertainty_key: str = "uncertainty", method: str = "highest"):
        """
        Initialize uncertainty sampler.
        
        Args:
            sample_size: Number of samples to select
            uncertainty_key: Key to access uncertainty score in data items
            method: Sampling method ('highest', 'lowest', 'mixed')
        """
        self.sample_size = sample_size
        self.uncertainty_key = uncertainty_key
        self.method = method
    
    def sample(self, data: List[Any], **kwargs) -> List[Any]:
        """
        Sample based on uncertainty scores.
        
        Args:
            data: List of data items with uncertainty scores
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of uncertainty-sampled items
        """
        if len(data) <= self.sample_size:
            return data.copy()
        
        # Filter items with uncertainty scores
        valid_items = [item for item in data if self.uncertainty_key in item]
        
        if len(valid_items) < self.sample_size:
            # Fall back to random sampling if not enough items
            return random.sample(data, self.sample_size)
        
        # Sort by uncertainty
        sorted_items = sorted(valid_items, key=lambda x: x[self.uncertainty_key], reverse=(self.method == "highest"))
        
        if self.method == "mixed":
            # Mix high and low uncertainty items
            high_uncertainty = sorted_items[:self.sample_size//2]
            low_uncertainty = sorted_items[-(self.sample_size//2):]
            selected_items = high_uncertainty + low_uncertainty
            random.shuffle(selected_items)
            return selected_items[:self.sample_size]
        else:
            return sorted_items[:self.sample_size]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sampler status."""
        return {
            "sampler_type": "uncertainty",
            "sample_size": self.sample_size,
            "uncertainty_key": self.uncertainty_key,
            "method": self.method
        } 