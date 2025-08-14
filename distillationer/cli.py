"""
Command line interface for Distillationer.
"""

import os
import asyncio
import argparse
from .distillationer import Distillationer


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description='Process images with distillation using VL models')
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str,
                            help='Input directory containing images')
    input_group.add_argument('--input-file', type=str,
                            help='Input single image file to process')
    
    # Output arguments
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory for results')
    
    # Model arguments
    parser.add_argument('-m', '--model', type=str, default='gemini-2.5-pro',
                        choices=Distillationer.get_available_models(),
                        help='Model to use for processing')
    
    # Prompt arguments
    parser.add_argument('-p', '--prompt-type', type=str, default='Gemini_Detection',
                        choices=Distillationer.get_available_model_tasks(),
                        help='Model_task combination (e.g., "qwen_detection", "gemini")')
    
    # API configuration
    parser.add_argument('--base-url', type=str, default=None,
                        help='Custom base URL for API (optional)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Custom API key (optional)')
    
    # Counter arguments
    parser.add_argument('--counters', type=str, nargs='+', default=['epoch'],
                        choices=Distillationer.get_available_counters(),
                        help='Counter types to use')
    parser.add_argument('--epoch-max', type=int, default=100,
                        help='Maximum epochs for epoch counter')
    parser.add_argument('--success-min', type=int, default=50,
                        help='Minimum successes for success counter')
    parser.add_argument('--error-max', type=int, default=10,
                        help='Maximum errors for error counter')
    
    # Sampler arguments
    parser.add_argument('--sampler', type=str, default='random',
                        choices=Distillationer.get_available_samplers(),
                        help='Sampler type to use')
    parser.add_argument('--sample-size', type=int, default=20,
                        help='Number of files to sample')
    
    # Processing options
    parser.add_argument('-s', '--semaphore', type=int, default=32,
                        help='Semaphore for async processing')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Test mode, will visualize the result')
    parser.add_argument('-r', '--refine', action='store_true',
                        help='Refine mode')
    parser.add_argument('--save-mode', type=str, default='divided', 
                        choices=['all', 'divided'],
                        help='Save mode: all (save all results in one file) or divided (save each result separately)')
    
    return parser


async def main():
    """Main function to handle command line execution."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Prepare counter configurations
    counter_configs = {}
    for counter in args.counters:
        if counter == 'epoch':
            counter_configs[counter] = {'max_epochs': args.epoch_max}
        elif counter == 'success':
            counter_configs[counter] = {'min_success': args.success_min}
        elif counter == 'error':
            counter_configs[counter] = {'max_errors': args.error_max}
        else:
            counter_configs[counter] = {}
    
    # Initialize Distillationer
    distillationer = Distillationer(
        model_name=args.model,
        output_path=args.output,
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_type=args.prompt_type,
        counters=args.counters,
        counter_configs=counter_configs,
        sampler_type=args.sampler,
        sampler_config={'sample_size': args.sample_size}
    )
    
    print(f"Model: {args.model}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Counters: {args.counters}")
    print(f"Sampler: {args.sampler}")
    print(f"Refine mode: {args.refine}")
    print(f"Test mode: {args.test}")
    print(f"Save mode: {args.save_mode}")
    
    # Process either a folder or a single file based on input
    if args.input_file:
        # Process single file (use parent VLProcessor method)
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
        if args.refine:
            result = await distillationer.process_single_image_async_refine(
                args.input_file, args.test, args.save_mode
            )
        else:
            result = await distillationer.process_single_image_async(
                args.input_file, args.test, args.save_mode
            )
        print(f"Processed single file: {args.input_file}")
        
    else:
        # Process folder with counter checking
        results = await distillationer.process_folder_with_counters(
            args.input, args.semaphore, test_mode=args.test, 
            refine=args.refine, save_mode=args.save_mode
        )
        print(f"Processed {len(results)} images from folder: {args.input}")


if __name__ == '__main__':
    asyncio.run(main()) 