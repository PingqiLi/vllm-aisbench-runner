"""
Statistical analysis utilities for repeated benchmark runs.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any


def find_result_files(experiment_dir: str, dataset_name: str, run_count: int) -> List[str]:
    """
    Find result files for repeated runs.

    Args:
        experiment_dir: Experiment directory path
        dataset_name: Dataset name (e.g., 'aime2024')
        run_count: Number of repeated runs

    Returns:
        List of paths to result summary files
    """
    result_files = []

    for run_id in range(1, run_count + 1):
        # Try different possible result file locations
        dir_name = f"{dataset_name}_{run_id}"
        dataset_dir = os.path.join(experiment_dir, dir_name)

        if not os.path.exists(dataset_dir):
            print(f"[Warning] Result directory not found: {dataset_dir}")
            continue

        # Look for summary files
        possible_files = [
            os.path.join(dataset_dir, 'summary.json'),
            os.path.join(dataset_dir, 'result.json'),
            os.path.join(dataset_dir, 'summary', 'summary.json'),
        ]

        # Also search for any .json files in the directory
        if os.path.exists(dataset_dir):
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith('.json') and 'summary' in file.lower():
                        possible_files.append(os.path.join(root, file))

        for file_path in possible_files:
            if os.path.exists(file_path):
                result_files.append(file_path)
                break

    return result_files


def extract_accuracy(result_file: str) -> Optional[float]:
    """
    Extract accuracy from result file.

    Args:
        result_file: Path to result JSON file

    Returns:
        Accuracy value (0-1), or None if not found
    """
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)

        # Try different possible keys for accuracy
        accuracy_keys = [
            'accuracy',
            'acc',
            'score',
            'result',
            'metric',
        ]

        # Search in nested structure
        def search_accuracy(obj, keys):
            if isinstance(obj, dict):
                for key in keys:
                    if key in obj:
                        val = obj[key]
                        if isinstance(val, (int, float)):
                            return float(val)
                # Recursively search nested dicts
                for value in obj.values():
                    result = search_accuracy(value, keys)
                    if result is not None:
                        return result
            return None

        accuracy = search_accuracy(data, accuracy_keys)

        if accuracy is not None:
            # Convert to 0-1 range if needed
            if accuracy > 1:
                accuracy = accuracy / 100.0
            return accuracy

        return None

    except Exception as e:
        print(f"[Warning] Error reading {result_file}: {e}")
        return None


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for repeated runs.

    Args:
        values: List of accuracy values

    Returns:
        Dictionary with statistical metrics
    """
    if not values:
        return {}

    values_array = np.array(values)
    n = len(values)

    mean = np.mean(values_array)
    std = np.std(values_array, ddof=1) if n > 1 else 0.0
    stderr = std / np.sqrt(n) if n > 1 else 0.0

    # 95% confidence interval
    ci_95 = 1.96 * stderr

    # Coefficient of variation
    cv = (std / mean) if mean != 0 else 0.0

    return {
        'mean': mean,
        'std': std,
        'stderr': stderr,
        'ci_95_lower': mean - ci_95,
        'ci_95_upper': mean + ci_95,
        'cv': cv,
        'min': np.min(values_array),
        'max': np.max(values_array),
        'range': np.max(values_array) - np.min(values_array),
        'n': n,
    }


def print_statistics_report(dataset_name: str, values: List[float], stats: Dict[str, float]):
    """
    Print formatted statistics report.

    Args:
        dataset_name: Name of the dataset
        values: List of individual run results
        stats: Statistical metrics
    """
    print("\n" + "=" * 80)
    print(f"📊 REPEATED RUN STATISTICS: {dataset_name}")
    print("=" * 80)

    # Individual runs
    print(f"\n📋 Individual Runs (n={stats['n']}):")
    for i, value in enumerate(values, 1):
        print(f"  Run {i}: {value:.4f} ({value*100:.2f}%)")

    print(f"\n📈 Statistical Summary:")
    print(f"  Mean:       {stats['mean']:.4f} ({stats['mean']*100:.2f}%)")
    print(f"  Std Dev:    {stats['std']:.4f} ({stats['std']*100:.2f}%)")
    print(f"  Std Error:  {stats['stderr']:.4f} ({stats['stderr']*100:.2f}%)")
    print(f"  Min-Max:    {stats['min']:.4f} - {stats['max']:.4f}")
    print(f"  Range:      {stats['range']:.4f} ({stats['range']*100:.2f}%)")

    print(f"\n🎯 95% Confidence Interval:")
    print(f"  [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]")
    print(f"  [{stats['ci_95_lower']*100:.2f}%, {stats['ci_95_upper']*100:.2f}%]")

    print(f"\n📊 Variability:")
    print(f"  CV (Coefficient of Variation): {stats['cv']:.4f} ({stats['cv']*100:.2f}%)")

    # Interpretation
    print(f"\n💡 Interpretation:")
    if stats['std'] < 0.01:
        print("  ✅ Very stable results (std < 1%)")
        print("  → High confidence in measurements")
    elif stats['std'] < 0.02:
        print("  ✅ Stable results (std < 2%)")
        print("  → Acceptable for comparison")
    elif stats['std'] < 0.03:
        print("  ⚠️  Moderate variability (std < 3%)")
        print("  → Consider more runs or lower batch_size")
    else:
        print("  ❌ High variability (std ≥ 3%)")
        print("  → Results unreliable, reduce batch_size or increase runs")

    # Detectable difference
    min_detectable = 2 * stats['ci_95_upper'] - 2 * stats['ci_95_lower']
    print(f"\n🔍 Minimum Detectable Difference (95% confidence):")
    print(f"  {min_detectable:.4f} ({min_detectable*100:.2f}%)")

    print("=" * 80 + "\n")


def analyze_repeated_runs(experiment_dir: str, dataset_name: str, run_count: int) -> Optional[Dict[str, Any]]:
    """
    Analyze results from repeated benchmark runs.

    Args:
        experiment_dir: Experiment directory path
        dataset_name: Dataset name
        run_count: Number of repeated runs

    Returns:
        Dictionary with analysis results, or None if failed
    """
    print(f"\n[Statistics] Analyzing {run_count} repeated runs for {dataset_name}...")

    # Find result files
    result_files = find_result_files(experiment_dir, dataset_name, run_count)

    if not result_files:
        print(f"[Statistics] ❌ No result files found")
        return None

    if len(result_files) < run_count:
        print(f"[Statistics] ⚠️  Warning: Found {len(result_files)}/{run_count} result files")

    # Extract accuracies
    accuracies = []
    for file_path in result_files:
        accuracy = extract_accuracy(file_path)
        if accuracy is not None:
            accuracies.append(accuracy)
        else:
            print(f"[Statistics] ⚠️  Warning: Could not extract accuracy from {file_path}")

    if not accuracies:
        print(f"[Statistics] ❌ No accuracy values found")
        return None

    # Calculate statistics
    stats = calculate_statistics(accuracies)

    # Print report
    print_statistics_report(dataset_name, accuracies, stats)

    return {
        'dataset': dataset_name,
        'values': accuracies,
        'statistics': stats,
    }
