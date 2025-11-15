"""
Experiment directory management utilities
"""

import os
from datetime import datetime


def create_experiment_dir(work_dir: str) -> str:
    """
    Create experiment directory with timestamp.

    Args:
        work_dir: Base work directory from config

    Returns:
        Path to experiment directory
    """
    # Create work directory
    os.makedirs(work_dir, exist_ok=True)

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(work_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"[Setup] Experiment directory: {experiment_dir}")
    return experiment_dir


def rename_output_folder(experiment_dir: str, dataset_name: str):
    """
    Rename output folder with dataset name.
    New format: {dataset_name} (simplified, no model/precision prefix)
    """
    try:
        if not experiment_dir or not os.path.exists(experiment_dir):
            return

        # Clean up dataset name
        clean_name = dataset_name.split('_gen')[0] if '_gen' in dataset_name else dataset_name

        # Find timestamp directories inside experiment_dir
        subdirs = [d for d in os.listdir(experiment_dir)
                  if os.path.isdir(os.path.join(experiment_dir, d)) and d[0].isdigit()]

        if not subdirs:
            return

        # Get the most recent directory
        subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(experiment_dir, x)), reverse=True)
        latest_dir = subdirs[0]

        # New name format: just the dataset name
        old_path = os.path.join(experiment_dir, latest_dir)
        new_path = os.path.join(experiment_dir, clean_name)

        # Rename if the new name is different
        if old_path != new_path and not os.path.exists(new_path):
            os.rename(old_path, new_path)

    except Exception:
        pass  # Silently ignore rename errors
