#!/usr/bin/env python3
"""
Integrated vLLM + AISBench Benchmark Runner

Launches vLLM service and runs AISBench benchmarks.

Example:
    python run.py --config-file configs/benchmarks/qwen3-30b-acc.yaml
"""

import argparse
import os
import subprocess
import sys
import yaml
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from utils import (
    load_suite_config, merge_config_with_args,
    VLLMManager, save_config_snapshot, save_dataset_config,
    create_experiment_dir, rename_output_folder
)
from utils.ais_config_patcher import patch_vllm_api_config


class BenchmarkRunner:
    """Coordinates vLLM service and AISBench evaluation."""

    def __init__(self, args):
        self.args = args
        self.start_time = None
        self.end_time = None
        self.experiment_timestamp = None
        self.experiment_dir = None
        self.custom_model_config_path = None
        self.vllm_manager = None




    def build_aisbench_command(self) -> list:
        """Build the ais_bench command with all specified parameters."""
        cmd = ["ais_bench"]

        # Model configuration
        if hasattr(self, 'custom_model_config_path') and self.custom_model_config_path:
            # Use custom config file (path without .py extension)
            config_path_no_ext = self.custom_model_config_path.replace('.py', '')
            cmd.extend(["--models", config_path_no_ext])
        elif self.args.ais_model:
            cmd.extend(["--models", self.args.ais_model])
        else:
            # Default to vllm_api_general_chat
            cmd.extend(["--models", "vllm_api_general_chat"])

        # Dataset configuration
        if self.args.datasets:
            cmd.extend(["--datasets"] + self.args.datasets)

        # Mode
        if self.args.mode:
            cmd.extend(["--mode", self.args.mode])

        # Summarizer
        if hasattr(self.args, 'summarizer') and self.args.summarizer:
            cmd.extend(["--summarizer", self.args.summarizer])

        # Merge datasets
        if hasattr(self.args, 'merge_ds') and self.args.merge_ds:
            cmd.append("--merge-ds")

        # Work directory - use experiment_dir for consistency
        cmd.extend(["--work-dir", self.experiment_dir])

        # Debug mode
        if self.args.debug:
            cmd.append("--debug")

        # Max workers
        if self.args.max_num_workers:
            cmd.extend(["--max-num-workers", str(self.args.max_num_workers)])

        # Number of prompts
        if self.args.num_prompts:
            cmd.extend(["--num-prompts", str(self.args.num_prompts)])

        # Dump evaluation details
        if self.args.dump_eval_details:
            cmd.append("--dump-eval-details")

        # Config file (if provided directly)
        if self.args.config:
            cmd.insert(1, self.args.config)

        # Additional AISBench arguments
        if self.args.ais_extra_args:
            cmd.extend(self.args.ais_extra_args.split())

        return cmd

    def run_aisbench(self, dataset_idx: int = 0, total_datasets: int = 1) -> bool:
        """Run AISBench evaluation."""
        cmd = self.build_aisbench_command()

        dataset_name = self.args.datasets[0] if self.args.datasets else "unknown"
        print(f"\n[Benchmark {dataset_idx}/{total_datasets}] Running: {dataset_name}")
        print(f"[Benchmark] Command: {' '.join(cmd)}\n")

        # Save dataset-specific config before running
        save_dataset_config(self.args, dataset_name, self.experiment_dir)

        try:
            result = subprocess.run(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )

            if result.returncode == 0:
                print(f"\n[Benchmark {dataset_idx}/{total_datasets}] ✓ Completed: {dataset_name}")
                rename_output_folder(self.experiment_dir, dataset_name)
                return True
            else:
                print(f"\n[Benchmark {dataset_idx}/{total_datasets}] ✗ Failed: {dataset_name} (code {result.returncode})")
                return False

        except Exception as e:
            print(f"\n[Benchmark {dataset_idx}/{total_datasets}] ✗ Error: {e}")
            return False

    def run(self) -> int:
        """
        Main execution flow - run each task/dataset separately with fresh vLLM instance.

        Returns:
            0 on success, 1 on failure
        """
        self.start_time = datetime.now()

        # Create experiment group directory
        base_work_dir = self.args.work_dir if self.args.work_dir else "outputs/default"
        self.experiment_dir = create_experiment_dir(base_work_dir)

        # Save configuration snapshot
        save_config_snapshot(self.args, self.experiment_dir, self.start_time)

        # Run task-based evaluation
        return self.run_tasks()

    def run_tasks(self) -> int:
        """Run evaluation using new task-based architecture."""
        tasks = self.args._tasks
        total_tasks = len(tasks)
        failed_tasks = []

        print("\n" + "=" * 80)
        print(f"[Runner] Starting {total_tasks} task(s)")
        print(f"[Runner] Suite: {self.args._suite_name}")
        print("=" * 80 + "\n")

        for idx, task in enumerate(tasks, 1):
            task_name = task.get('task', {}).get('name', f'task-{idx}')
            run_id = task.get('task', {}).get('run_id')

            # Add run_id to display name if present
            display_name = f"{task_name} (run {run_id})" if run_id else task_name

            print("\n" + "=" * 80)
            print(f"[Progress] Task {idx}/{total_tasks}: {display_name}")
            print("=" * 80)

            # Apply task configuration to args
            self._apply_task_config(task)

            # Patch AISBench config file with task-specific parameters
            self._patch_ais_config(task)

            # Initialize vLLM manager for this task
            self.vllm_manager = VLLMManager(self.args, self.experiment_dir)

            try:
                # Launch vLLM with task-specific config
                dataset_name = task.get('aisbench', {}).get('dataset')
                if not self.vllm_manager.launch(dataset_name=dataset_name):
                    failed_tasks.append(task_name)
                    continue

                # Run evaluation
                success = self.run_aisbench(dataset_idx=idx, total_datasets=total_tasks)
                if not success:
                    failed_tasks.append(task_name)
                else:
                    # Rename output directory to dataset name on success
                    dataset_name = task.get('task', {}).get('dataset', '')
                    run_id = task.get('task', {}).get('run_id')

                    if dataset_name:
                        # Add run_id suffix if this is a repeated run
                        if run_id:
                            output_name = f"{dataset_name}_{run_id}"
                        else:
                            output_name = dataset_name
                        rename_output_folder(self.experiment_dir, output_name)

            except KeyboardInterrupt:
                print("\n\n[Runner] Interrupted by user")
                self.vllm_manager.shutdown()
                return 1
            except Exception as e:
                print(f"[Runner] Error running {task_name}: {e}")
                failed_tasks.append(task_name)
            finally:
                self.vllm_manager.shutdown()

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        print("\n" + "=" * 80)
        print(f"[Summary] Total time: {duration:.2f}s")
        print(f"[Summary] Completed: {total_tasks - len(failed_tasks)}/{total_tasks}")

        if failed_tasks:
            print(f"[Summary] Failed tasks: {', '.join(failed_tasks)}")
            return 1

        print("[Summary] ✓ All tasks completed successfully")
        return 0

    def _apply_task_config(self, task: Dict[str, Any]):
        """Apply task configuration to args."""
        # Apply vLLM config from task
        vllm_config = task.get('vllm', {})
        for key, value in vllm_config.items():
            setattr(self.args, key, value)

        # Apply AISBench config from task
        ais_config = task.get('aisbench', {})
        self.args.datasets = [ais_config.get('dataset')]
        self.args.ais_model = ais_config.get('model')

        # Set other AISBench parameters
        if 'max_out_len' in ais_config:
            if not hasattr(self.args, 'model_config'):
                self.args.model_config = {}
            self.args.model_config['max_out_len'] = ais_config['max_out_len']

        if 'generation_kwargs' in ais_config:
            if not hasattr(self.args, 'model_config'):
                self.args.model_config = {}
            self.args.model_config['generation_kwargs'] = ais_config['generation_kwargs']

        if 'max_num_workers' in ais_config:
            self.args.max_num_workers = ais_config['max_num_workers']

        if 'batch_size' in ais_config:
            if not hasattr(self.args, 'model_config'):
                self.args.model_config = {}
            self.args.model_config['batch_size'] = ais_config['batch_size']

        if 'merge_ds' in ais_config:
            self.args.merge_ds = ais_config['merge_ds']

        if 'mode' in ais_config:
            self.args.mode = ais_config['mode']

        if 'dump_eval_details' in ais_config:
            self.args.dump_eval_details = ais_config['dump_eval_details']

        if 'summarizer' in ais_config:
            self.args.summarizer = ais_config['summarizer']

    def _patch_ais_config(self, task: Dict[str, Any]):
        """
        Patch AISBench vllm_api_general_chat.py with task-specific parameters.

        Args:
            task: Task configuration dictionary
        """
        ais_config = task.get('aisbench', {})

        # Extract parameters to patch
        batch_size = ais_config.get('batch_size')
        max_out_len = ais_config.get('max_out_len')
        generation_kwargs = ais_config.get('generation_kwargs')

        # Only patch if at least one parameter is specified
        if batch_size is not None or generation_kwargs is not None or max_out_len is not None:
            print(f"\n[Config Patcher] Patching AISBench model config...")
            success = patch_vllm_api_config(
                batch_size=batch_size,
                generation_kwargs=generation_kwargs,
                max_out_len=max_out_len
            )
            if not success:
                print("[Config Patcher] Warning: Failed to patch config, using defaults")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Integrated vLLM + AISBench Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using benchmark config (recommended)
  python run.py --config-file configs/benchmarks/qwen3-30b-acc.yaml

  # Override specific datasets
  python run.py --config-file configs/benchmarks/qwen3-30b-acc.yaml --datasets ceval mmlu

  # Command-line only
  python run.py --model-path Qwen/Qwen3-30B-A3B --datasets gsm8k --tensor-parallel-size 2
        """
    )

    # Configuration file
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to YAML configuration file'
    )

    # vLLM parameters
    vllm_group = parser.add_argument_group('vLLM Options')
    vllm_group.add_argument('--model-path', type=str, help='HuggingFace model path')
    vllm_group.add_argument('--host', type=str, default='localhost', help='vLLM host (default: localhost)')
    vllm_group.add_argument('--port', type=int, default=8000, help='vLLM port (default: 8000)')
    vllm_group.add_argument('--tensor-parallel-size', type=int, help='Tensor parallelism size')
    vllm_group.add_argument('--pipeline-parallel-size', type=int, help='Pipeline parallelism size')
    vllm_group.add_argument('--quantization', type=str, help='Quantization method (e.g., "ascend" for W4A4)')
    vllm_group.add_argument('--rope-scaling', type=str, help='RoPE scaling config (JSON string)')
    vllm_group.add_argument('--max-model-len', type=int, help='Maximum model context length')
    vllm_group.add_argument('--gpu-memory-utilization', type=float, help='GPU memory utilization (0-1)')
    vllm_group.add_argument('--trust-remote-code', action='store_true', help='Trust remote code')
    vllm_group.add_argument('--dtype', type=str, help='Data type (e.g., bfloat16, auto)')
    vllm_group.add_argument('--max-num-seqs', type=int, help='Maximum concurrent sequences')
    vllm_group.add_argument('--enable-prefix-caching', action='store_true', help='Enable prefix caching')
    vllm_group.add_argument('--disable-log-requests', action='store_true', help='Disable request logging')
    vllm_group.add_argument('--tokenizer', type=str, help='Tokenizer path')
    vllm_group.add_argument('--revision', type=str, help='Model revision')
    vllm_group.add_argument('--served-model-name', type=str, help='Model name for API')
    vllm_group.add_argument('--enforce-eager', action='store_true', help='Enforce eager execution (disable CUDA graphs)')
    vllm_group.add_argument('--vllm-extra-args', type=str, help='Additional vLLM arguments (space-separated, e.g., "--arg1 --arg2 value")')
    vllm_group.add_argument('--vllm-timeout', type=int, default=600, help='vLLM startup timeout (default: 600s)')
    vllm_group.add_argument('--vllm-log-file', type=str, help='vLLM log file path')

    # AISBench parameters
    ais_group = parser.add_argument_group('AISBench Options')
    ais_group.add_argument('--datasets', nargs='+', help='Dataset configuration names')
    ais_group.add_argument('--ais-model', type=str, help='AISBench model configuration')
    ais_group.add_argument('--mode', type=str, choices=['all', 'perf', 'infer'], default='all',
                          help='Evaluation mode (default: all)')
    ais_group.add_argument('--work-dir', type=str, help='Output directory')
    ais_group.add_argument('--debug', action='store_true', help='Enable debug mode')
    ais_group.add_argument('--max-num-workers', type=int, help='Maximum concurrent workers')
    ais_group.add_argument('--num-prompts', type=int, help='Number of prompts to evaluate (debug)')
    ais_group.add_argument('--dump-eval-details', action='store_true', help='Dump evaluation details')
    ais_group.add_argument('--config', type=str, help='AISBench config file')
    ais_group.add_argument('--ais-extra-args', type=str, help='Additional AISBench arguments (space-separated)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config file (required)
    if not args.config_file:
        print("Error: --config-file is required")
        print("Usage: python run.py --config-file configs/suites/qwen3-30b-bf16-acc.yaml")
        return 1

    config = load_suite_config(args.config_file)
    args = merge_config_with_args(config, args)

    # Run benchmarks
    runner = BenchmarkRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
