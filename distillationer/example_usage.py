"""
Example usage of Distillationer.
"""

import asyncio
from .distillationer import Distillationer


async def main():
    """Example usage of Distillationer."""
    
    # Initialize Distillationer with counters and sampler
    distillationer = Distillationer(
        model_name='gemini-2.5-pro',
        output_path='./distillation_output',
        prompt_type='gemini',
        counters=['epoch', 'success'],  # Use epoch and success counters
        counter_configs={
            'epoch': {'max_epochs': 100},  # Stop after 100 epochs
            'success': {'min_success': 50}  # Stop after 50 successful results
        },
        sampler_type='random',
        sampler_config={'sample_size': 20}  # Sample 20 files
    )
    
    # Add additional counter if needed
    distillationer.add_counter('error', max_errors=10)
    
    # Process folder with counter checking
    results = await distillationer.process_folder_with_counters(
        folder_path='./input_images',
        semaphore=16,  # Limit concurrent processing
        test_mode=False,
        refine=False,
        save_mode='divided'
    )
    
    # Print results
    print(f"Processing completed!")
    print(f"Total results: {len(results)}")
    print(f"Counter status: {distillationer.get_counter_status()}")
    print(f"Sampler status: {distillationer.get_sampler_status()}")


if __name__ == '__main__':
    asyncio.run(main()) 