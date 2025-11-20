"""
Configuration loading and merging utilities
"""

import argparse
import yaml
from typing import Dict, Any


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_suite_config(suite_path: str) -> Dict[str, Any]:
    """
    Load suite configuration.

    Supports two formats:
    1. tasks: [list of file paths] - Load from external files
    2. _tasks: [list of inline task configs] - Use inlined tasks (for reproducible snapshots)

    Args:
        suite_path: Path to suite config file

    Returns:
        Suite configuration with loaded tasks
    """
    suite_config = load_config_file(suite_path)

    # Check if tasks are already inlined (from config_snapshot.yaml)
    if '_tasks' in suite_config:
        # Tasks are already inlined, use them directly
        task_configs = suite_config['_tasks']
    else:
        # Load all task configurations from external files
        task_configs = []
        for task_path in suite_config.get('tasks', []):
            task_config = load_config_file(task_path)
            task_configs.append(task_config)

    # Get repeat count (default 1)
    repeat_count = suite_config.get('suite', {}).get('repeat', 1)

    # If repeat > 1, expand tasks with run IDs
    if repeat_count > 1:
        expanded_tasks = []
        for run_id in range(1, repeat_count + 1):
            for task in task_configs:
                # Create a copy of the task with run_id
                task_copy = task.copy()
                if 'task' not in task_copy:
                    task_copy['task'] = {}
                task_copy['task']['run_id'] = run_id
                expanded_tasks.append(task_copy)
        task_configs = expanded_tasks

    # Prepare merged config
    merged = {
        'suite': suite_config.get('suite', {}),
        'output': suite_config.get('output', {}),
        'runtime': suite_config.get('runtime', {}),
        '_tasks': task_configs,
        '_suite_name': suite_config.get('suite', {}).get('name', 'custom'),
        '_repeat': repeat_count,
    }

    return merged


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge suite config with command line arguments (CLI args take precedence).

    CLI arguments can override values in all tasks. For example:
    - --model-path will override model_path in all tasks
    - --tensor-parallel-size will override tensor_parallel_size in all tasks
    """
    # Store tasks and suite name
    args._tasks = config['_tasks']
    args._suite_name = config['_suite_name']

    # Set work_dir from suite config if not provided via CLI
    if 'work_dir' in config.get('output', {}):
        if not args.work_dir:
            args.work_dir = config['output']['work_dir']

    # Set debug from suite config if not provided via CLI
    if 'debug' in config.get('runtime', {}):
        if not args.debug:
            args.debug = config['runtime']['debug']

    # Apply CLI overrides to all tasks
    # This allows using --model-path, --tensor-parallel-size etc to override task configs
    _apply_cli_overrides_to_tasks(args)

    return args


def _apply_cli_overrides_to_tasks(args: argparse.Namespace):
    """Apply CLI argument overrides to all task configurations."""
    if not hasattr(args, '_tasks') or not args._tasks:
        return

    # vLLM parameters that can be overridden
    vllm_overrides = {
        'model_path': args.model_path,
        'host': args.host,
        'port': args.port,
        'tensor_parallel_size': args.tensor_parallel_size,
        'pipeline_parallel_size': args.pipeline_parallel_size,
        'quantization': args.quantization,
        'rope_scaling': args.rope_scaling,
        'max_model_len': args.max_model_len,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'trust_remote_code': args.trust_remote_code,
        'dtype': args.dtype,
        'max_num_seqs': args.max_num_seqs,
        'enable_prefix_caching': args.enable_prefix_caching,
        'disable_log_requests': args.disable_log_requests,
        'tokenizer': args.tokenizer,
        'revision': args.revision,
        'served_model_name': args.served_model_name,
        'enforce_eager': args.enforce_eager,
    }

    # AISBench parameters that can be overridden
    aisbench_overrides = {
        'mode': args.mode,
        'max_num_workers': args.max_num_workers,
        'num_prompts': args.num_prompts,
        'dump_eval_details': args.dump_eval_details,
    }

    # Apply overrides to each task
    for task in args._tasks:
        # Override vLLM config
        if 'vllm' in task:
            for key, value in vllm_overrides.items():
                if value is not None:
                    task['vllm'][key] = value

        # Override AISBench config
        if 'aisbench' in task:
            for key, value in aisbench_overrides.items():
                if value is not None:
                    task['aisbench'][key] = value
