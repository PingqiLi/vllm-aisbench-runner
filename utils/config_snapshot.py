"""
Configuration snapshot utilities for experiment reproducibility
"""

import os
import yaml
import platform
import sys
from datetime import datetime
from typing import Dict, Any
import argparse


def save_config_snapshot(args: argparse.Namespace, experiment_dir: str, start_time: datetime):
    """
    Save comprehensive configuration snapshot for reproducibility.
    Creates:
    - config_snapshot.yaml: Full reproducible configuration (can be used with --config-file)
    - config_original.yaml: Original config file for reference (if applicable)
    - metadata.yaml: Runtime information
    """
    if not experiment_dir:
        return

    import shutil

    config_dir = os.path.join(experiment_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    try:
        # 1. Save original config file for reference
        if hasattr(args, 'config_file') and args.config_file:
            original_config_path = os.path.join(config_dir, "config_original.yaml")
            shutil.copy(args.config_file, original_config_path)

        # 2. Generate full expanded configuration snapshot (reproducible)
        snapshot = generate_full_config_snapshot(args, start_time)
        snapshot_path = os.path.join(config_dir, "config_snapshot.yaml")
        with open(snapshot_path, 'w') as f:
            yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # 3. Save runtime metadata
        metadata = generate_metadata(args, start_time)
        metadata_path = os.path.join(config_dir, "metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        print(f"[Setup] Config snapshot saved to: {config_dir}/")
        print(f"[Setup] To reproduce this run: python run.py --config-file {snapshot_path}")

    except Exception as e:
        print(f"[Setup] Warning: Failed to save config snapshot: {e}")


def generate_full_config_snapshot(args: argparse.Namespace, start_time: datetime) -> Dict[str, Any]:
    """
    Generate comprehensive configuration snapshot with all parameters.

    For task-based architecture: Returns suite config with inlined tasks (方案B)
    For legacy architecture: Returns benchmark config with datasets
    """
    # Check if using task-based architecture
    if hasattr(args, '_tasks') and args._tasks:
        return _generate_task_based_snapshot(args, start_time)
    else:
        return _generate_legacy_snapshot(args, start_time)


def _generate_task_based_snapshot(args: argparse.Namespace, start_time: datetime) -> Dict[str, Any]:
    """Generate reproducible config for task-based architecture (方案B)."""

    # Suite metadata
    suite_name = getattr(args, '_suite_name', 'custom')
    snapshot = {
        'suite': {
            'name': suite_name,
            'description': f'Reproduced from run at {start_time.isoformat()}' if start_time else 'Reproduced run',
            'type': 'accuracy',  # or extract from original suite config
        },
        '_tasks': [],  # Inline all task configurations
        'output': {
            'work_dir': args.work_dir if args.work_dir else 'outputs/default',
        },
        'runtime': {
            'debug': args.debug if args.debug else False,
        }
    }

    # Inline each task with complete configuration
    for task_config in args._tasks:
        task_snapshot = {}

        # Task metadata
        if 'task' in task_config:
            task_snapshot['task'] = task_config['task'].copy()

        # vLLM configuration
        if 'vllm' in task_config:
            task_snapshot['vllm'] = task_config['vllm'].copy()

        # AISBench configuration
        if 'aisbench' in task_config:
            task_snapshot['aisbench'] = task_config['aisbench'].copy()

        snapshot['_tasks'].append(task_snapshot)

    return snapshot


def _generate_legacy_snapshot(args: argparse.Namespace, start_time: datetime) -> Dict[str, Any]:
    """Generate config snapshot for legacy dataset-based architecture."""
    # Base vLLM configuration
    vllm_keys = ['model_path', 'host', 'port', 'tensor_parallel_size', 'pipeline_parallel_size',
                 'quantization', 'rope_scaling', 'max_model_len', 'dtype',
                 'gpu_memory_utilization', 'trust_remote_code', 'max_num_seqs',
                 'enable_prefix_caching', 'disable_log_requests', 'tokenizer',
                 'revision', 'served_model_name']

    vllm_config = {}
    for k in vllm_keys:
        val = getattr(args, k, None)
        if val is not None:
            vllm_config[k] = val

    if args.vllm_timeout:
        vllm_config['timeout'] = args.vllm_timeout
    if args.vllm_log_file:
        vllm_config['log_file'] = args.vllm_log_file
    if args.vllm_extra_args:
        vllm_config['extra_args'] = args.vllm_extra_args

    # AISBench configuration
    ais_keys = ['datasets', 'mode', 'work_dir', 'max_num_workers', 'debug',
                'dump_eval_details', 'num_prompts']
    ais_config = {}
    for k in ais_keys:
        val = getattr(args, k, None)
        if val is not None:
            ais_config[k] = val

    if args.ais_model:
        ais_config['model'] = args.ais_model
    if hasattr(args, 'summarizer') and args.summarizer:
        ais_config['summarizer'] = args.summarizer
    if hasattr(args, 'merge_ds') and args.merge_ds:
        ais_config['merge_ds'] = args.merge_ds

    # Dataset-specific configurations
    dataset_configs = {}
    if hasattr(args, '_dataset_configs'):
        for dataset_name, ds_config in args._dataset_configs.items():
            dataset_configs[dataset_name] = {
                'description': ds_config.get('dataset', {}).get('description', ''),
                'model_config': ds_config.get('model_config', {}),
            }
            if 'vllm_config_override' in ds_config:
                dataset_configs[dataset_name]['vllm_config_override'] = ds_config['vllm_config_override']

    snapshot = {
        'benchmark': {
            'name': getattr(args, '_benchmark_name', 'custom'),
            'timestamp': start_time.isoformat() if start_time else None,
        },
        'vllm': vllm_config,
        'aisbench': ais_config,
        'datasets': dataset_configs,
    }

    return snapshot


def generate_metadata(args: argparse.Namespace, start_time: datetime) -> Dict[str, Any]:
    """Generate runtime metadata for reproducibility."""
    metadata = {
        'runtime': {
            'timestamp': start_time.isoformat() if start_time else None,
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'working_directory': os.getcwd(),
        },
        'versions': {
            'python': platform.python_version(),
            'os': platform.system(),
        },
        'command': {
            'executable': sys.argv[0],
            'args': sys.argv[1:],
            'full_command': ' '.join(sys.argv),
        },
        'reproduction': {
            'command': 'python run.py --config-file outputs/<experiment>/configs/config_snapshot.yaml',
            'note': 'Use config_snapshot.yaml to reproduce this exact run with all parameters',
        }
    }

    # Try to get package versions
    try:
        import subprocess
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                for pkg in ['vllm', 'torch', 'transformers', 'opencompass']:
                    if line.lower().startswith(pkg):
                        parts = line.split()
                        if len(parts) >= 2:
                            metadata['versions'][parts[0]] = parts[1]
    except:
        pass

    return metadata


def save_dataset_config(args: argparse.Namespace, dataset_name: str, experiment_dir: str):
    """Save the actual configuration used for this specific dataset."""
    if not experiment_dir:
        return

    config_dir = os.path.join(experiment_dir, "configs", "per_dataset")
    os.makedirs(config_dir, exist_ok=True)

    # Clean dataset name for filename
    safe_name = dataset_name.replace('_gen_0_shot_cot_chat_prompt', '') \
                             .replace('_gen_0_shot_chat_prompt', '') \
                             .replace('.py', '') \
                             .replace('/', '_')

    # Build dataset-specific config
    dataset_config = {
        'dataset': {'name': dataset_name},
        'vllm': {},
        'aisbench': {},
    }

    # Capture current vLLM config
    vllm_keys = ['model_path', 'host', 'port', 'tensor_parallel_size',
                 'max_model_len', 'rope_scaling', 'dtype', 'gpu_memory_utilization',
                 'trust_remote_code', 'max_num_seqs', 'enable_prefix_caching']
    for k in vllm_keys:
        val = getattr(args, k, None)
        if val is not None:
            dataset_config['vllm'][k] = val

    # Capture AISBench config
    if args.ais_model:
        dataset_config['aisbench']['model'] = args.ais_model
    if args.num_prompts:
        dataset_config['aisbench']['num_prompts'] = args.num_prompts

    # Add dataset-specific model_config if available
    if hasattr(args, '_dataset_configs'):
        ds_cfg = args._dataset_configs.get(dataset_name, {})
        if 'model_config' in ds_cfg:
            dataset_config['model_config'] = ds_cfg['model_config']
        if 'vllm_config_override' in ds_cfg:
            dataset_config['vllm_config_override_applied'] = ds_cfg['vllm_config_override']

    config_path = os.path.join(config_dir, f"{safe_name}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
