"""
Configuration loading and merging utilities
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_benchmark_config(benchmark_path: str) -> Dict[str, Any]:
    """
    Load benchmark configuration with model and dataset compositions.

    Args:
        benchmark_path: Path to benchmark config file

    Returns:
        Merged configuration dictionary
    """
    # Load main benchmark config
    benchmark_config = load_config_file(benchmark_path)

    # Load model config
    model_config_path = benchmark_config.get('model_config')
    if model_config_path:
        model_config = load_config_file(model_config_path)
    else:
        model_config = {}

    # Load dataset configs
    dataset_configs = []
    dataset_list = []
    for dataset_path in benchmark_config.get('datasets', []):
        ds_config = load_config_file(dataset_path)
        dataset_configs.append(ds_config)
        dataset_list.append(ds_config['dataset']['name'])

    # Merge configurations
    merged = {
        'vllm': model_config.get('vllm', {}),
        'aisbench': model_config.get('aisbench', {}),
    }

    # Add datasets list
    merged['aisbench']['datasets'] = dataset_list

    # Add work_dir from output config
    if 'output' in benchmark_config:
        merged['aisbench']['work_dir'] = benchmark_config['output'].get('work_dir')

    # Add debug flag from runtime
    if 'runtime' in benchmark_config:
        merged['aisbench']['debug'] = benchmark_config['runtime'].get('debug', False)

    # Store dataset configs for later use
    merged['_dataset_configs'] = {ds['dataset']['name']: ds for ds in dataset_configs}
    merged['_benchmark_name'] = benchmark_config['benchmark']['name']

    return merged


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config with command line arguments (CLI args take precedence)."""
    vllm_config = config.get('vllm', {})
    ais_config = config.get('aisbench', {})

    # vLLM parameter mappings (arg_name: (config_key, default_value))
    vllm_mappings = {
        'model_path': ('model_path', None),
        'host': ('host', 'localhost'),
        'port': ('port', 8000),
        'tensor_parallel_size': ('tensor_parallel_size', None),
        'pipeline_parallel_size': ('pipeline_parallel_size', None),
        'quantization': ('quantization', None),
        'rope_scaling': ('rope_scaling', None),
        'max_model_len': ('max_model_len', None),
        'gpu_memory_utilization': ('gpu_memory_utilization', None),
        'trust_remote_code': ('trust_remote_code', None),
        'dtype': ('dtype', None),
        'max_num_seqs': ('max_num_seqs', None),
        'enable_prefix_caching': ('enable_prefix_caching', None),
        'disable_log_requests': ('disable_log_requests', None),
        'tokenizer': ('tokenizer', None),
        'revision': ('revision', None),
        'served_model_name': ('served_model_name', None),
        'vllm_extra_args': ('extra_args', None),
        'vllm_timeout': ('timeout', 300),
        'vllm_log_file': ('log_file', None),
    }

    for arg_name, (config_key, default) in vllm_mappings.items():
        arg_val = getattr(args, arg_name, None)
        config_val = vllm_config.get(config_key)
        if (arg_val is None or arg_val == default) and config_val is not None:
            setattr(args, arg_name, config_val)

    # AISBench parameter mappings
    ais_mappings = {
        'datasets': ('datasets', None),
        'ais_model': ('model', None),
        'mode': ('mode', 'all'),
        'work_dir': ('work_dir', None),
        'debug': ('debug', None),
        'max_num_workers': ('max_num_workers', None),
        'num_prompts': ('num_prompts', None),
        'dump_eval_details': ('dump_eval_details', None),
        'config': ('config_file', None),
        'ais_extra_args': ('extra_args', None),
    }

    for arg_name, (config_key, default) in ais_mappings.items():
        arg_val = getattr(args, arg_name, None)
        config_val = ais_config.get(config_key)
        if (arg_val is None or arg_val == default) and config_val is not None:
            setattr(args, arg_name, config_val)

    # Special handling for summarizer and merge_ds
    if not hasattr(args, 'summarizer') or not args.summarizer:
        args.summarizer = ais_config.get('summarizer')
    if not hasattr(args, 'merge_ds') or not args.merge_ds:
        args.merge_ds = ais_config.get('merge_ds')

    # Load model_config overrides from YAML
    if 'model_config' in ais_config:
        args.model_config = ais_config['model_config']

    # Store dataset-specific configs if available (new-style config)
    if '_dataset_configs' in config:
        args._dataset_configs = config['_dataset_configs']
    if '_benchmark_name' in config:
        args._benchmark_name = config['_benchmark_name']

    return args
