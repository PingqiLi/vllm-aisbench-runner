"""
AISBench configuration file patcher.

Dynamically patches vllm_api_general_chat.py with task-specific configurations.
"""

import os
import re
import shutil
from typing import Dict, Any, Optional


def find_ais_bench_config_path() -> Optional[str]:
    """
    Find the vllm_api_general_chat.py configuration file path.

    Returns:
        Path to vllm_api_general_chat.py, or None if not found
    """
    try:
        import ais_bench
        ais_bench_root = os.path.dirname(ais_bench.__file__)
        config_path = os.path.join(
            ais_bench_root,
            'benchmark',
            'configs',
            'models',
            'vllm_api',
            'vllm_api_general_chat.py'
        )

        if os.path.exists(config_path):
            return config_path
        else:
            print(f"[Warning] vllm_api_general_chat.py not found at: {config_path}")
            return None
    except ImportError:
        print("[Warning] ais_bench not installed, cannot patch config")
        return None


def backup_config(config_path: str) -> str:
    """
    Create a backup of the configuration file.

    Args:
        config_path: Path to the config file

    Returns:
        Path to the backup file
    """
    backup_path = config_path + '.backup'

    # Only create backup if it doesn't exist (preserve original)
    if not os.path.exists(backup_path):
        shutil.copy2(config_path, backup_path)
        print(f"[Config Patcher] Backup created: {backup_path}")

    return backup_path


def restore_config(config_path: str, backup_path: str):
    """
    Restore configuration from backup.

    Args:
        config_path: Path to the config file
        backup_path: Path to the backup file
    """
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, config_path)
        print(f"[Config Patcher] Configuration restored from backup")


def patch_vllm_api_config(
    batch_size: Optional[int] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    max_out_len: Optional[int] = None
) -> bool:
    """
    Patch vllm_api_general_chat.py with task-specific configurations.

    Args:
        batch_size: Batch size for inference
        generation_kwargs: Generation parameters (temperature, top_k, etc.)
        max_out_len: Maximum output length

    Returns:
        True if successful, False otherwise
    """
    config_path = find_ais_bench_config_path()
    if not config_path:
        return False

    # Create backup
    backup_path = backup_config(config_path)

    try:
        # Read current config
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Patch batch_size
        if batch_size is not None:
            content = re.sub(
                r'batch_size\s*=\s*\d+',
                f'batch_size={batch_size}',
                content
            )
            print(f"[Config Patcher] Set batch_size={batch_size}")

        # Patch max_out_len
        if max_out_len is not None:
            content = re.sub(
                r'max_out_len\s*=\s*\d+',
                f'max_out_len={max_out_len}',
                content
            )
            print(f"[Config Patcher] Set max_out_len={max_out_len}")

        # Patch generation_kwargs
        if generation_kwargs is not None:
            # Build new generation_kwargs dict string
            kwargs_lines = []
            for key, value in generation_kwargs.items():
                if isinstance(value, str):
                    kwargs_lines.append(f'            {key} = "{value}",')
                elif value is None:
                    kwargs_lines.append(f'            {key} = None,')
                else:
                    kwargs_lines.append(f'            {key} = {value},')

            new_kwargs = 'generation_kwargs = dict(\n' + '\n'.join(kwargs_lines) + '\n        ),'

            # Replace generation_kwargs block
            pattern = r'generation_kwargs\s*=\s*dict\([^)]*\),'
            content = re.sub(pattern, new_kwargs, content, flags=re.DOTALL)

            print(f"[Config Patcher] Updated generation_kwargs: {generation_kwargs}")

        # Write patched config
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"[Config Patcher] Successfully patched: {config_path}")
        return True

    except Exception as e:
        print(f"[Config Patcher] Error patching config: {e}")
        # Restore from backup on error
        if os.path.exists(backup_path):
            restore_config(config_path, backup_path)
        return False


def cleanup_backup():
    """Remove backup file if it exists."""
    config_path = find_ais_bench_config_path()
    if config_path:
        backup_path = config_path + '.backup'
        if os.path.exists(backup_path):
            os.remove(backup_path)
            print(f"[Config Patcher] Backup removed: {backup_path}")
