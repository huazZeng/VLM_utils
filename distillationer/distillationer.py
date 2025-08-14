"""
Distillationer that extends VLProcessor with counters and samplers.
"""

import os
import json
import asyncio
import shutil
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

from vlm_test.vl_detect import VLProcessor
from .managers import CounterManager, SamplerManager


class Distillationer(VLProcessor):
    """
    Extended VLProcessor with counter and sampler functionality.
    """
    
    def __init__(self, 
                 model_name: str = 'gemini-2.5-pro', 
                 output_path: str = './temp', 
                 base_url: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 prompt_type: str = 'gemini',
                 counters: Optional[List[Union[str, Any]]] = None,
                 counter_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                 sampler_type: str = 'random',
                 sampler_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Distillationer with VLProcessor functionality plus counters and samplers.
        
        Args:
            model_name: Name of the model to use
            output_path: Base output path for results
            base_url: Custom base URL for API (optional)
            api_key: Custom API key (optional)
            prompt_type: Model_task combination (e.g., "qwen_detection", "gemini")
            counters: List of counter types or instances
            counter_configs: Configuration for each counter type
            sampler_type: Type of sampler to use
            sampler_config: Configuration for the sampler
        """
        # Initialize parent VLProcessor
        super().__init__(model_name, output_path, base_url, api_key, prompt_type)
        
        # Initialize counter manager
        self.counter_manager = CounterManager(counters, counter_configs)
        
        # Initialize sampler manager
        sampler_kwargs = sampler_config or {}
        self.sampler_manager = SamplerManager(sampler_type, **sampler_kwargs)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
    
    def add_counter(self, counter_type: str, **kwargs) -> None:
        """
        Add a new counter to the distillation process.
        
        Args:
            counter_type: Type of counter to add
            **kwargs: Configuration parameters for the counter
        """
        self.counter_manager.add_counter(counter_type, **kwargs)
    
    def change_sampler(self, sampler_type: str, **kwargs) -> None:
        """
        Change the sampler for the distillation process.
        
        Args:
            sampler_type: New sampler type
            **kwargs: Configuration parameters for the new sampler
        """
        self.sampler_manager.change_sampler(sampler_type, **kwargs)
    
    async def process_folder_with_counters(self, 
                                         folder_path: str, 
                                         semaphore: int = 32, 
                                         test_mode: bool = False, 
                                         refine: bool = False, 
                                         save_mode: str = 'divided') -> List[Dict[str, Any]]:
        """
        Process all images in a folder with counter checking.
        
        Args:
            folder_path: Path to the folder containing images
            semaphore: Semaphore count to limit concurrent operations
            test_mode: Whether to generate visualization
            refine: Whether to use refinement process
            save_mode: Save mode ('divided' or 'all')
            
        Returns:
            List of results for each image
        """
        # Get all image files in the folder recursively
        from vlm_test.file_manager import FileManager
        image_paths = FileManager.find_image_files(folder_path)
        
        if not image_paths:
            print(f"No supported image files found in {folder_path}")
            return []
        
        print(f"Starting distillation with {len(image_paths)} images")
        print(f"Counters: {[c.__class__.__name__ for c in self.counter_manager.counters]}")
        print(f"Sampler: {self.sampler_manager.sampler_type}")
        
        # Reset counters
        self.counter_manager.reset_all()
        
        # Create semaphore object
        semaphore_obj = asyncio.Semaphore(semaphore)
        
        # Process all images asynchronously with counter checking
        results = await self._process_multiple_images_with_counters(
            image_paths, semaphore_obj, test_mode, refine, save_mode
        )
        
        # Sample from output folder
        sampled_files = await self._sample_output_folder()
        
        # Save sampling results
        sample_results = self._save_sampling_results(sampled_files)
        
        print(f"Distillation completed!")
        print(f"Total processed: {len(results)}")
        print(f"Sampled files: {len(sampled_files)}")
        
        return results
    
    async def _process_multiple_images_with_counters(self, 
                                                    image_paths: List[str], 
                                                    semaphore: Optional[asyncio.Semaphore] = None,
                                                    test_mode: bool = False, 
                                                    refine: bool = False, 
                                                    save_mode: str = 'divided') -> List[Dict[str, Any]]:
        """
        Process multiple images asynchronously with counter checking.
        
        Args:
            image_paths: List of paths to input images
            semaphore: Semaphore to limit concurrent operations
            test_mode: Whether to generate visualization
            refine: Whether to use refinement process
            save_mode: Save mode ('divided' or 'all')
            
        Returns:
            List of results for each image
        """
        # Create tasks for all images
        if semaphore is None:
            # Process without concurrency limit
            if refine:
                tasks = [self.process_single_image_async_refine(img_path, test_mode, save_mode) 
                        for img_path in image_paths]
            else:
                tasks = [self.process_single_image_async(img_path, test_mode, save_mode) 
                        for img_path in image_paths]
        else:
            # Limit concurrency with semaphore
            if refine:
                async def process_with_semaphore(img_path):
                    async with semaphore:
                        return await self.process_single_image_async_refine(img_path, test_mode, save_mode)
            else:
                async def process_with_semaphore(img_path):
                    async with semaphore:
                        return await self.process_single_image_async(img_path, test_mode, save_mode)
            
            tasks = [process_with_semaphore(img_path) for img_path in image_paths]
        
        # Process all requests concurrently with progress bar and counter checking
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing images"):
            try:
                result = await future
                results.append(result)
                
                # Process JSON data through counters
                if result and result.get('result') is not None:
                    self.counter_manager.process_json(result)
                
                # Check if any counter indicates stopping
                if self.counter_manager.should_stop():
                    print(f"\nCounter condition met. Cancelling remaining tasks...")
                    print(f"Counter counts: {self.counter_manager.get_all_counts()}")
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break
                    
            except asyncio.CancelledError:
                print("Task was cancelled")
                continue
            except Exception as e:
                print(f"Error during image processing: {e}")
                results.append(None)
        
        # Wait for any remaining tasks to complete
        remaining_tasks = [task for task in tasks if not task.done()]
        if remaining_tasks:
            print(f"Waiting for {len(remaining_tasks)} remaining tasks to complete...")
            await asyncio.gather(*remaining_tasks, return_exceptions=True)
        
        return [r for r in results if r is not None]
    
    async def _sample_output_folder(self) -> List[str]:
        """
        Sample files from the output folder using the configured sampler.
        
        Returns:
            List of sampled file paths
        """
        print("Sampling from output folder...")
        
        # Get all JSON files from output folder
        json_files = []
        
        for root, dirs, files in os.walk(self.output_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        print(f"Found {len(json_files)} JSON files in output folder")
        
        if not json_files:
            print("No JSON files found for sampling")
            return []
        
        # Use sampler to select files
        sampled_files = self.sampler_manager.sample(json_files)
        
        print(f"Sampled {len(sampled_files)} files")
        return sampled_files
    
    def _save_sampling_results(self, sampled_files: List[str]) -> Dict[str, Any]:
        """
        Save sampled files to the sample folder.
        
        Args:
            sampled_files: List of sampled file paths
            
        Returns:
            Dictionary with sampling results
        """
        # Create sample folder
        sample_folder = os.path.join(self.output_path, 'sample')
        os.makedirs(sample_folder, exist_ok=True)
        
        # Copy sampled files to sample folder
        copied_files = []
        for file_path in sampled_files:
            try:
                # Get relative path from output folder
                rel_path = os.path.relpath(file_path, self.output_path)
                target_path = os.path.join(sample_folder, rel_path)
                
                # Create target directory if needed
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Copy file
                shutil.copy2(file_path, target_path)
                copied_files.append(target_path)
                
            except Exception as e:
                print(f"Error copying {file_path}: {e}")
        
        # Save sampling summary
        summary = {
            'total_sampled': len(sampled_files),
            'successfully_copied': len(copied_files),
            'sample_folder': sample_folder,
            'sampler_type': self.sampler_manager.sampler_type,
            'sampler_status': self.sampler_manager.get_status(),
            'sampled_files': [os.path.relpath(f, self.output_path) for f in sampled_files]
        }
        
        summary_path = os.path.join(sample_folder, 'sampling_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Sampling results saved to: {sample_folder}")
        print(f"Successfully copied {len(copied_files)} files")
        
        return summary
    
    def get_counter_status(self) -> Dict[str, Any]:
        """
        Get current counter status.
        
        Returns:
            Dictionary with counter status
        """
        return {
            'counter_counts': self.counter_manager.get_all_counts(),
            'should_stop': self.counter_manager.should_stop(),
            'stopping_counters': self.counter_manager.get_stopping_counters()
        }
    
    def get_sampler_status(self) -> Dict[str, Any]:
        """
        Get current sampler status.
        
        Returns:
            Dictionary with sampler status
        """
        return self.sampler_manager.get_status()
    
    @staticmethod
    def get_available_counters() -> List[str]:
        """Get list of available counter types."""
        return CounterManager.get_available_counters()
    
    @staticmethod
    def get_available_samplers() -> List[str]:
        """Get list of available sampler types."""
        return SamplerManager.get_available_samplers()
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available models."""
        from vlm_test.model_config import ModelConfig
        return ModelConfig.get_available_models()
    
    @staticmethod
    def get_available_model_tasks() -> List[str]:
        """Get list of available model tasks."""
        from vlm_test.prompt.manager import PromptManager
        return PromptManager.get_available_model_tasks()
