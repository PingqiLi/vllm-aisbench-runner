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

from utils import load_config_file, load_benchmark_config, merge_config_with_args, VLLMManager


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

    def _apply_vllm_config_override(self, dataset_name: str) -> bool:
        """
        Apply vLLM configuration override from dataset config if specified.

        Args:
            dataset_name: Dataset configuration name

        Returns:
            True if vLLM config was overridden (requires restart), False otherwise
        """
        if not hasattr(self.args, '_dataset_configs'):
            return False

        dataset_config = self.args._dataset_configs.get(dataset_name)
        if not dataset_config or 'vllm_config_override' not in dataset_config:
            return False

        vllm_override = dataset_config['vllm_config_override']
        print(f"[Setup] Applying vLLM config override for {dataset_name}:")

        # Apply each override to args
        for key, value in vllm_override.items():
            # Convert config key to args attribute name
            arg_name = key
            old_value = getattr(self.args, arg_name, None)
            setattr(self.args, arg_name, value)
            print(f"  - {key}: {old_value} → {value}")

        return True

    def _get_dataset_recommended_max_out_len(self, dataset_name: str) -> int:
        """
        Get recommended max_out_len based on dataset characteristics.

        Args:
            dataset_name: Dataset configuration name

        Returns:
            Recommended max_out_len value
        """
        dataset_name_lower = dataset_name.lower()

        # Long-form reasoning datasets need more tokens
        if any(keyword in dataset_name_lower for keyword in ['gpqa', 'aime', 'math']):
            return 4096

        # Long context datasets
        if 'longbench' in dataset_name_lower:
            return 8192

        # Code generation datasets
        if 'code' in dataset_name_lower or 'livecode' in dataset_name_lower:
            return 2048

        # Standard QA datasets (CEVAL, MMLU, etc.)
        return 2048

    def _create_model_config_override(self, dataset_name: Optional[str] = None) -> Optional[str]:
        """
        Create a temporary model config file with overrides from YAML.

        Args:
            dataset_name: Current dataset name for auto-tuning max_out_len

        Returns:
            Path to the generated config file, or None if no overrides
        """
        # Get dataset-specific config if available
        dataset_config = None
        if dataset_name and hasattr(self.args, '_dataset_configs'):
            dataset_config = self.args._dataset_configs.get(dataset_name)

        # Start with base model_config or empty dict
        if hasattr(self.args, 'model_config') and self.args.model_config:
            model_config = self.args.model_config.copy()
        elif dataset_config and 'model_config' in dataset_config:
            # Use dataset's model_config
            model_config = dataset_config['model_config'].copy()
        else:
            # No config overrides, use auto-detection
            model_config = {}

        # Override with dataset-specific config if available
        if dataset_config and 'model_config' in dataset_config:
            ds_model_config = dataset_config['model_config']
            model_config.update(ds_model_config)
            print(f"[Setup] Using dataset-specific config for {dataset_name}")

        # Auto-adjust max_out_len based on dataset if not explicitly set
        if dataset_name and 'max_out_len' not in model_config:
            recommended_len = self._get_dataset_recommended_max_out_len(dataset_name)
            model_config['max_out_len'] = recommended_len
            print(f"[Setup] Auto-adjusted max_out_len to {recommended_len} for {dataset_name}")

        # If still no config, return None
        if not model_config:
            return None

        # Create temporary config directory
        temp_config_dir = os.path.join(self.experiment_dir, "temp_model_configs")
        os.makedirs(temp_config_dir, exist_ok=True)

        # Generate model config file with dataset-specific name
        if dataset_name:
            # Use dataset name in config file for clarity
            safe_dataset_name = dataset_name.replace('_gen_0_shot_cot_chat_prompt.py', '').replace('.py', '')
            config_file = os.path.join(temp_config_dir, f"vllm_api_custom_{safe_dataset_name}.py")
        else:
            config_file = os.path.join(temp_config_dir, "vllm_api_custom.py")

        # Extract overrides
        overrides = {}
        generation_kwargs_overrides = {}

        for key, value in model_config.items():
            if key == 'generation_kwargs' and isinstance(value, dict):
                generation_kwargs_overrides = value
            else:
                overrides[key] = value

        # Build the config file content
        config_content = '''from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

# Auto-generated model config with overrides from YAML
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-custom',
        path="",
        model="",
        request_rate={request_rate},
        retry={retry},
        host_ip="{host_ip}",
        host_port={host_port},
        max_out_len={max_out_len},
        batch_size={batch_size},
        trust_remote_code={trust_remote_code},
        generation_kwargs=dict(
            temperature={temperature},
            top_k={top_k},
            top_p={top_p},
            seed={seed},
            repetition_penalty={repetition_penalty},
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
'''

        # Default values from vllm_api_general_chat.py
        defaults = {
            'host_ip': 'localhost',
            'host_port': 8080,
            'max_out_len': 512,
            'batch_size': 1,
            'request_rate': 0,
            'retry': 2,
            'trust_remote_code': False,
        }

        generation_defaults = {
            'temperature': 0.5,
            'top_k': 10,
            'top_p': 0.95,
            'seed': None,
            'repetition_penalty': 1.03,
        }

        # Apply overrides
        for key, default_value in defaults.items():
            overrides.setdefault(key, default_value)

        for key, default_value in generation_defaults.items():
            generation_kwargs_overrides.setdefault(key, default_value)

        # Format the config content
        formatted_content = config_content.format(
            host_ip=overrides['host_ip'],
            host_port=overrides['host_port'],
            max_out_len=overrides['max_out_len'],
            batch_size=overrides['batch_size'],
            request_rate=overrides['request_rate'],
            retry=overrides['retry'],
            trust_remote_code=overrides['trust_remote_code'],
            temperature=generation_kwargs_overrides['temperature'],
            top_k=generation_kwargs_overrides['top_k'],
            top_p=generation_kwargs_overrides['top_p'],
            seed=generation_kwargs_overrides['seed'],
            repetition_penalty=generation_kwargs_overrides['repetition_penalty'],
        )

        # Write the config file
        with open(config_file, 'w') as f:
            f.write(formatted_content)

        print(f"[Setup] Generated model config with overrides: {config_file}")
        return config_file

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

        # Work directory
        if self.args.work_dir:
            cmd.extend(["--work-dir", self.args.work_dir])
        else:
            # Default work directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cmd.extend(["--work-dir", f"outputs/vllm_ais_runner_{timestamp}"])

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

    def create_experiment_dir(self) -> str:
        """
        Create experiment group directory with timestamp.

        Returns:
            Path to the experiment directory
        """
        # Generate experiment timestamp (format: 2025-01-11_12-00-00)
        self.experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Get base work_dir
        base_work_dir = self.args.work_dir if self.args.work_dir else "outputs/default"

        # Create experiment directory: {work_dir}/{timestamp}/
        self.experiment_dir = os.path.join(base_work_dir, self.experiment_timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)

        print(f"[Setup] Experiment directory: {self.experiment_dir}")
        return self.experiment_dir

    def save_config_snapshot(self):
        """
        Save comprehensive configuration snapshot for reproducibility.
        Creates:
        - config_snapshot.yaml: Full expanded configuration with all overrides
        - config_original.yaml: Original benchmark config file (reference)
        - reproduce.sh: Script to reproduce this exact run
        - metadata.yaml: Runtime information (timestamp, versions, etc.)
        """
        if not self.experiment_dir:
            return

        import shutil
        import platform

        config_dir = os.path.join(self.experiment_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)

        try:
            # 1. Save original config file for reference
            if hasattr(self.args, 'config_file') and self.args.config_file:
                original_config_path = os.path.join(config_dir, "config_original.yaml")
                shutil.copy(self.args.config_file, original_config_path)

            # 2. Generate full expanded configuration snapshot
            snapshot = self._generate_full_config_snapshot()
            snapshot_path = os.path.join(config_dir, "config_snapshot.yaml")
            with open(snapshot_path, 'w') as f:
                yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)

            # 3. Save runtime metadata
            metadata = self._generate_metadata()
            metadata_path = os.path.join(config_dir, "metadata.yaml")
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

            # 4. Generate reproduction script
            reproduce_script = self._generate_reproduce_script()
            reproduce_path = os.path.join(config_dir, "reproduce.sh")
            with open(reproduce_path, 'w') as f:
                f.write(reproduce_script)
            os.chmod(reproduce_path, 0o755)

            print(f"[Setup] Configuration snapshot saved to: {config_dir}/")
            print(f"  - config_snapshot.yaml (full expanded config)")
            print(f"  - metadata.yaml (runtime info)")
            print(f"  - reproduce.sh (reproduction script)")

        except Exception as e:
            print(f"[Setup] Warning: Failed to save config snapshot: {e}")
            import traceback
            traceback.print_exc()

    def _generate_full_config_snapshot(self) -> dict:
        """Generate comprehensive configuration snapshot with all parameters."""
        # Base vLLM configuration
        vllm_keys = ['model_path', 'host', 'port', 'tensor_parallel_size', 'pipeline_parallel_size',
                     'quantization', 'rope_scaling', 'max_model_len', 'dtype',
                     'gpu_memory_utilization', 'trust_remote_code', 'max_num_seqs',
                     'enable_prefix_caching', 'disable_log_requests', 'tokenizer',
                     'revision', 'served_model_name']

        vllm_config = {}
        for k in vllm_keys:
            val = getattr(self.args, k, None)
            if val is not None:
                vllm_config[k] = val

        if self.args.vllm_timeout:
            vllm_config['timeout'] = self.args.vllm_timeout
        if self.args.vllm_log_file:
            vllm_config['log_file'] = self.args.vllm_log_file
        if self.args.vllm_extra_args:
            vllm_config['extra_args'] = self.args.vllm_extra_args

        # AISBench configuration
        ais_keys = ['datasets', 'mode', 'work_dir', 'max_num_workers', 'debug',
                    'dump_eval_details', 'num_prompts']
        ais_config = {}
        for k in ais_keys:
            val = getattr(self.args, k, None)
            if val is not None:
                ais_config[k] = val

        if self.args.ais_model:
            ais_config['model'] = self.args.ais_model
        if hasattr(self.args, 'summarizer') and self.args.summarizer:
            ais_config['summarizer'] = self.args.summarizer
        if hasattr(self.args, 'merge_ds') and self.args.merge_ds:
            ais_config['merge_ds'] = self.args.merge_ds

        # Dataset-specific configurations
        dataset_configs = {}
        if hasattr(self.args, '_dataset_configs'):
            for dataset_name, ds_config in self.args._dataset_configs.items():
                dataset_configs[dataset_name] = {
                    'description': ds_config.get('dataset', {}).get('description', ''),
                    'model_config': ds_config.get('model_config', {}),
                }
                # Include vllm_config_override if present
                if 'vllm_config_override' in ds_config:
                    dataset_configs[dataset_name]['vllm_config_override'] = ds_config['vllm_config_override']

        snapshot = {
            'benchmark': {
                'name': getattr(self.args, '_benchmark_name', 'custom'),
                'timestamp': self.start_time.isoformat() if self.start_time else None,
            },
            'vllm': vllm_config,
            'aisbench': ais_config,
            'datasets': dataset_configs,
        }

        return snapshot

    def _generate_metadata(self) -> dict:
        """Generate runtime metadata for reproducibility."""
        import platform
        import sys

        metadata = {
            'runtime': {
                'timestamp': self.start_time.isoformat() if self.start_time else None,
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
            }
        }

        # Try to get package versions
        try:
            import subprocess
            result = subprocess.run(['pip', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse key packages
                for line in result.stdout.split('\n'):
                    for pkg in ['vllm', 'torch', 'transformers', 'opencompass']:
                        if line.lower().startswith(pkg):
                            parts = line.split()
                            if len(parts) >= 2:
                                metadata['versions'][parts[0]] = parts[1]
        except:
            pass

        return metadata

    def _generate_reproduce_script(self) -> str:
        """Generate shell script to reproduce this exact run."""
        script_lines = [
            "#!/bin/bash",
            "# Reproduction script for benchmark run",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "set -e  # Exit on error",
            "",
            "# Check if running from correct directory",
            "if [ ! -f \"run.py\" ]; then",
            "    echo \"Error: Please run this script from the benchmark_runner directory\"",
            "    exit 1",
            "fi",
            "",
        ]

        # Add the actual command
        if hasattr(self.args, 'config_file') and self.args.config_file:
            cmd_parts = [
                "python run.py",
                f"--config-file {self.args.config_file}",
            ]

            # Add optional arguments if they were specified
            if self.args.num_prompts:
                cmd_parts.append(f"--num-prompts {self.args.num_prompts}")
            if self.args.debug:
                cmd_parts.append("--debug")

            script_lines.append("# Run the benchmark")
            script_lines.append(" \\\n    ".join(cmd_parts))
        else:
            # Build command from individual args
            cmd_parts = ["python run.py"]

            if self.args.model_path:
                cmd_parts.append(f"--model-path {self.args.model_path}")
            if self.args.datasets:
                datasets_str = " ".join(self.args.datasets)
                cmd_parts.append(f"--datasets {datasets_str}")
            if self.args.num_prompts:
                cmd_parts.append(f"--num-prompts {self.args.num_prompts}")

            script_lines.append(" \\\n    ".join(cmd_parts))

        script_lines.append("")
        return "\n".join(script_lines)

    def rename_output_folder(self):
        """
        Rename the output folder with dataset name only.
        New format: {dataset_name} (simplified, no model/precision prefix)
        """
        try:
            # Work in the experiment directory
            if not self.experiment_dir or not os.path.exists(self.experiment_dir):
                return

            # Extract clean dataset name
            dataset_name = self.args.datasets[0] if self.args.datasets else "unknown"

            # Clean up dataset name: remove _gen_0_shot_cot_chat_prompt suffix
            # Examples:
            #   ceval_gen_0_shot_cot_chat_prompt -> ceval
            #   aime2024_gen_0_shot_chat_prompt -> aime2024
            #   livecodebench_code_generate_lite_gen_0_shot_chat -> livecodebench
            clean_name = dataset_name.split('_gen')[0] if '_gen' in dataset_name else dataset_name

            # Find timestamp directories inside experiment_dir
            subdirs = [d for d in os.listdir(self.experiment_dir)
                      if os.path.isdir(os.path.join(self.experiment_dir, d)) and d[0].isdigit()]

            if not subdirs:
                return

            # Get the most recent directory
            subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(self.experiment_dir, x)), reverse=True)
            latest_dir = subdirs[0]

            # New name format: just the dataset name
            old_path = os.path.join(self.experiment_dir, latest_dir)
            new_path = os.path.join(self.experiment_dir, clean_name)

            # Rename if the new name is different
            if old_path != new_path and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                # Quietly renamed, no output needed
            elif os.path.exists(new_path):
                pass  # Silently skip if exists

        except Exception:
            pass  # Silently ignore rename errors

    def save_dataset_config(self, dataset_name: str):
        """Save the actual configuration used for this specific dataset."""
        if not self.experiment_dir:
            return

        config_dir = os.path.join(self.experiment_dir, "configs", "per_dataset")
        os.makedirs(config_dir, exist_ok=True)

        # Clean dataset name for filename
        safe_name = dataset_name.replace('_gen_0_shot_cot_chat_prompt', '') \
                                 .replace('_gen_0_shot_chat_prompt', '') \
                                 .replace('.py', '') \
                                 .replace('/', '_')

        # Build dataset-specific config
        dataset_config = {
            'dataset': {
                'name': dataset_name,
            },
            'vllm': {},
            'aisbench': {},
        }

        # Capture current vLLM config (including any overrides)
        vllm_keys = ['model_path', 'host', 'port', 'tensor_parallel_size',
                     'max_model_len', 'rope_scaling', 'dtype', 'gpu_memory_utilization',
                     'trust_remote_code', 'max_num_seqs', 'enable_prefix_caching']
        for k in vllm_keys:
            val = getattr(self.args, k, None)
            if val is not None:
                dataset_config['vllm'][k] = val

        # Capture AISBench config
        if self.args.ais_model:
            dataset_config['aisbench']['model'] = self.args.ais_model
        if self.args.num_prompts:
            dataset_config['aisbench']['num_prompts'] = self.args.num_prompts

        # Add dataset-specific model_config if available
        if hasattr(self.args, '_dataset_configs'):
            ds_cfg = self.args._dataset_configs.get(dataset_name, {})
            if 'model_config' in ds_cfg:
                dataset_config['model_config'] = ds_cfg['model_config']
            if 'vllm_config_override' in ds_cfg:
                dataset_config['vllm_config_override_applied'] = ds_cfg['vllm_config_override']

        config_path = os.path.join(config_dir, f"{safe_name}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    def run_aisbench(self, dataset_idx: int = 0, total_datasets: int = 1) -> bool:
        """Run AISBench evaluation."""
        cmd = self.build_aisbench_command()

        dataset_name = self.args.datasets[0] if self.args.datasets else "unknown"
        print(f"\n[Benchmark {dataset_idx}/{total_datasets}] Running: {dataset_name}")
        print(f"[Benchmark] Command: {' '.join(cmd)}\n")

        # Save dataset-specific config before running
        self.save_dataset_config(dataset_name)

        try:
            result = subprocess.run(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )

            if result.returncode == 0:
                print(f"\n[Benchmark {dataset_idx}/{total_datasets}] ✓ Completed: {dataset_name}")
                self.rename_output_folder()
                return True
            else:
                print(f"\n[Benchmark {dataset_idx}/{total_datasets}] ✗ Failed: {dataset_name} (code {result.returncode})")
                return False

        except Exception as e:
            print(f"\n[Benchmark {dataset_idx}/{total_datasets}] ✗ Error: {e}")
            return False

    def run(self) -> int:
        """
        Main execution flow - run each dataset separately with fresh vLLM instance.

        Returns:
            0 on success, 1 on failure
        """
        self.start_time = datetime.now()

        # Create experiment group directory
        self.create_experiment_dir()

        # Save configuration snapshot
        self.save_config_snapshot()

        # Initialize vLLM manager
        self.vllm_manager = VLLMManager(self.args, self.experiment_dir)

        # Update work_dir to point to experiment directory
        if self.experiment_dir:
            self.args.work_dir = self.experiment_dir

        datasets = self.args.datasets if self.args.datasets else []

        if not datasets:
            print("Error: No datasets specified")
            return 1

        total_datasets = len(datasets)
        failed_datasets = []

        print("\n" + "=" * 80)
        print(f"[Runner] Starting {total_datasets} benchmark(s)")
        print(f"[Runner] Model: {self.args.model_path}")
        print("=" * 80 + "\n")

        for idx, dataset in enumerate(datasets, 1):
            print("\n" + "=" * 80)
            print(f"[Progress] Dataset {idx}/{total_datasets}: {dataset}")
            print("=" * 80)

            original_datasets = self.args.datasets
            self.args.datasets = [dataset]

            # Check if dataset requires vLLM config override
            needs_vllm_restart = self._apply_vllm_config_override(dataset)

            # Generate dataset-specific model config if needed
            if hasattr(self.args, 'model_config') and self.args.model_config:
                self.custom_model_config_path = self._create_model_config_override(dataset_name=dataset)

            try:
                # Launch vLLM with dataset-specific log file
                # If needs_vllm_restart, the config has been updated in args
                if not self.vllm_manager.launch(dataset_name=dataset):
                    failed_datasets.append(dataset)
                    self.args.datasets = original_datasets
                    continue

                success = self.run_aisbench(dataset_idx=idx, total_datasets=total_datasets)

                if not success:
                    failed_datasets.append(dataset)

            except KeyboardInterrupt:
                print("\n\n[Runner] Interrupted by user")
                self.vllm_manager.shutdown()
                return 1
            except Exception as e:
                print(f"[Runner] Error running {dataset}: {e}")
                failed_datasets.append(dataset)
            finally:
                self.vllm_manager.shutdown()
                self.args.datasets = original_datasets

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        print("\n" + "=" * 80)
        print(f"[Summary] Total time: {duration:.2f}s")
        print(f"[Summary] Completed: {total_datasets - len(failed_datasets)}/{total_datasets}")
        if failed_datasets:
            print(f"[Summary] Failed: {', '.join(failed_datasets)}")
        print("=" * 80 + "\n")

        return 0 if not failed_datasets else 1


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
    vllm_group.add_argument('--vllm-extra-args', type=str, help='Additional vLLM arguments (space-separated)')
    vllm_group.add_argument('--vllm-timeout', type=int, default=300, help='vLLM startup timeout (default: 300s)')
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

    # Load config file if provided
    if args.config_file:
        config = load_benchmark_config(args.config_file)
        args = merge_config_with_args(config, args)
        # Store the config file path for reference
        args.config_file = args.config_file

    # Validate required parameters
    if not args.model_path:
        print("Error: --model-path is required (or specify in config file)")
        return 1

    if not args.datasets:
        print("Error: --datasets is required (or specify in config file)")
        return 1

    # Run benchmarks
    runner = BenchmarkRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
