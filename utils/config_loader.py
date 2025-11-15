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

    # Prepare merged config
    merged = {
        'suite': suite_config.get('suite', {}),
        'output': suite_config.get('output', {}),
        'runtime': suite_config.get('runtime', {}),
        '_tasks': task_configs,
        '_suite_name': suite_config.get('suite', {}).get('name', 'custom'),
    }

    return merged


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """Merge suite config with command line arguments (CLI args take precedence)."""
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

    return args
