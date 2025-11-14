#!/usr/bin/env python3
"""
Integrated vLLM + AISBench Benchmark Runner

Launches vLLM service and runs AISBench benchmarks.

Example:
    python run.py --config-file configs/benchmarks/qwen3-30b-accuracy.yaml
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import requests
import yaml
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path


class VLLMAISBenchRunner:
    """Integrated runner for vLLM service and AISBench evaluation."""

    def __init__(self, args):
        self.args = args
        self.vllm_process: Optional[subprocess.Popen] = None
        self.start_time = None
        self.end_time = None
        self.experiment_timestamp = None
        self.experiment_dir = None
        self.custom_model_config_path = None

    def build_vllm_command(self) -> List[str]:
        """Build the vLLM serve command with all specified parameters."""
        cmd = ["vllm", "serve", self.args.model_path]

        # vLLM specific parameters
        if self.args.host:
            cmd.extend(["--host", self.args.host])
        if self.args.port:
            cmd.extend(["--port", str(self.args.port)])
        if self.args.tensor_parallel_size:
            cmd.extend(["--tensor-parallel-size", str(self.args.tensor_parallel_size)])
        if self.args.pipeline_parallel_size:
            cmd.extend(["--pipeline-parallel-size", str(self.args.pipeline_parallel_size)])
        if self.args.quantization:
            cmd.extend(["--quantization", self.args.quantization])
        if self.args.rope_scaling:
            cmd.extend(["--rope-scaling", json.dumps(self.args.rope_scaling)])
        if self.args.max_model_len:
            cmd.extend(["--max-model-len", str(self.args.max_model_len)])
        if self.args.gpu_memory_utilization:
            cmd.extend(["--gpu-memory-utilization", str(self.args.gpu_memory_utilization)])
        if self.args.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.args.dtype:
            cmd.extend(["--dtype", self.args.dtype])
        if self.args.max_num_seqs:
            cmd.extend(["--max-num-seqs", str(self.args.max_num_seqs)])
        if self.args.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.args.disable_log_requests:
            cmd.append("--disable-log-requests")
        if self.args.tokenizer:
            cmd.extend(["--tokenizer", self.args.tokenizer])
        if self.args.revision:
            cmd.extend(["--revision", self.args.revision])
        if self.args.served_model_name:
            cmd.extend(["--served-model-name", self.args.served_model_name])

        # Additional vLLM arguments
        if self.args.vllm_extra_args:
            cmd.extend(self.args.vllm_extra_args.split())

        return cmd

    def build_aisbench_command(self) -> List[str]:
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

    def wait_for_vllm_ready(self, timeout: int = 300) -> bool:
        """
        Wait for vLLM service to be ready by checking the health endpoint.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if service is ready, False if timeout
        """
        url = f"http://{self.args.host}:{self.args.port}/health"
        print(f"[vLLM] Waiting for service (timeout: {timeout}s)...")

        start_time = time.time()
        check_interval = 10  # Check every 10 seconds
        last_print_time = start_time

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    elapsed = int(time.time() - start_time)
                    print(f"[vLLM] ✓ Service ready (took {elapsed}s)")
                    return True
            except requests.exceptions.RequestException:
                pass

            # Check if process is still running
            if self.vllm_process and self.vllm_process.poll() is not None:
                print("[vLLM] ✗ Process terminated unexpectedly")
                return False

            # Print progress every 30 seconds
            current_time = time.time()
            if current_time - last_print_time >= 30:
                elapsed = int(current_time - start_time)
                remaining = timeout - elapsed
                print(f"[vLLM] Still waiting... ({elapsed}s elapsed, {remaining}s remaining)")
                last_print_time = current_time

            time.sleep(check_interval)

        print(f"[vLLM] ✗ Timeout after {timeout}s")
        return False

    def kill_existing_vllm(self):
        """Kill any existing vLLM processes."""
        try:
            result = subprocess.run(
                ["pkill", "-f", "vllm"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("[vLLM] Killed existing processes")
                time.sleep(2)
        except Exception as e:
            pass  # Silently ignore if no processes found

    def launch_vllm(self) -> bool:
        """Launch vLLM service."""
        self.kill_existing_vllm()

        cmd = self.build_vllm_command()
        print(f"\n[vLLM] Starting service... (TP={self.args.tensor_parallel_size}, dtype={self.args.dtype})")
        if self.args.vllm_log_file:
            print(f"[vLLM] Logs: {self.args.vllm_log_file}")

        try:
            # Launch vLLM in a separate process
            if self.args.vllm_log_file:
                # Ensure log directory exists
                log_dir = os.path.dirname(self.args.vllm_log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)

                log_file = open(self.args.vllm_log_file, 'w')
                self.vllm_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # Create new process group for clean shutdown
                )
            else:
                self.vllm_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )

            # Wait for service to be ready
            if not self.wait_for_vllm_ready(timeout=self.args.vllm_timeout):
                return False

            return True

        except Exception as e:
            print(f"[vLLM] ✗ Failed to launch: {e}")
            return False

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
        """Save configuration snapshot to experiment directory."""
        if not self.experiment_dir:
            return

        config_path = os.path.join(self.experiment_dir, "config.yaml")

        try:
            if hasattr(self.args, 'config_file') and self.args.config_file:
                import shutil
                shutil.copy(self.args.config_file, config_path)
                print(f"[Setup] Config saved: {config_path}")
            else:
                # Generate config from command-line arguments
                vllm_keys = ['model_path', 'host', 'port', 'tensor_parallel_size', 'pipeline_parallel_size',
                             'quantization', 'dtype', 'max_model_len', 'gpu_memory_utilization',
                             'trust_remote_code', 'max_num_seqs', 'enable_prefix_caching']
                vllm_config = {k: getattr(self.args, k) for k in vllm_keys if getattr(self.args, k, None)}
                if self.args.vllm_timeout:
                    vllm_config['timeout'] = self.args.vllm_timeout
                if self.args.vllm_log_file:
                    vllm_config['log_file'] = self.args.vllm_log_file

                ais_keys = ['datasets', 'mode', 'work_dir', 'max_num_workers', 'debug', 'dump_eval_details']
                ais_config = {k: getattr(self.args, k) for k in ais_keys if getattr(self.args, k, None)}
                if self.args.ais_model:
                    ais_config['model'] = self.args.ais_model
                if hasattr(self.args, 'summarizer') and self.args.summarizer:
                    ais_config['summarizer'] = self.args.summarizer
                if hasattr(self.args, 'merge_ds') and self.args.merge_ds:
                    ais_config['merge_ds'] = self.args.merge_ds

                config = {'vllm': vllm_config, 'aisbench': ais_config}
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"[Setup] Config generated: {config_path}")

        except Exception as e:
            print(f"[Setup] Warning: Failed to save config: {e}")

    def extract_model_info_from_config(self) -> tuple:
        """Extract model name and precision from config file path."""
        if hasattr(self.args, 'config_file') and self.args.config_file:
            config_path = Path(self.args.config_file)
            model_dir = config_path.parent.name
            config_name = config_path.stem

            # Extract precision, handling _perf suffix
            parts = config_name.split('_')
            if 'perf' in parts:
                precision = 'perf'
            else:
                precision = parts[-1] if len(parts) > 1 else 'unknown'

            return model_dir, precision
        return 'unknown', 'unknown'

    def rename_output_folder(self):
        """
        Rename the output folder with model name, precision and benchmark name.
        New format: {model}_{precision}_{benchmark} (no timestamp)
        """
        try:
            # Work in the experiment directory
            if not self.experiment_dir or not os.path.exists(self.experiment_dir):
                return

            model_name, precision = self.extract_model_info_from_config()
            dataset_name = self.args.datasets[0] if self.args.datasets else "unknown"

            # Find timestamp directories inside experiment_dir
            subdirs = [d for d in os.listdir(self.experiment_dir)
                      if os.path.isdir(os.path.join(self.experiment_dir, d)) and d[0].isdigit()]

            if not subdirs:
                return

            # Get the most recent directory
            subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(self.experiment_dir, x)), reverse=True)
            latest_dir = subdirs[0]

            # New name format: {model}_{precision}_{benchmark} (no timestamp)
            old_path = os.path.join(self.experiment_dir, latest_dir)
            new_name = f"{model_name}_{precision}_{dataset_name}"
            new_path = os.path.join(self.experiment_dir, new_name)

            # Rename if the new name is different
            if old_path != new_path and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                # Quietly renamed, no output needed
            elif os.path.exists(new_path):
                pass  # Silently skip if exists

        except Exception as e:
            pass  # Silently ignore rename errors

    def run_aisbench(self, dataset_idx: int = 0, total_datasets: int = 1) -> bool:
        """Run AISBench evaluation."""
        cmd = self.build_aisbench_command()

        dataset_name = self.args.datasets[0] if self.args.datasets else "unknown"
        print(f"\n[Benchmark {dataset_idx}/{total_datasets}] Running: {dataset_name}")
        print(f"[Benchmark] Command: {' '.join(cmd)}\n")

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

    def cleanup(self):
        """Clean up vLLM process."""
        if self.vllm_process:
            print("[vLLM] Shutting down...")
            try:
                # Send SIGTERM to the entire process group
                os.killpg(os.getpgid(self.vllm_process.pid), signal.SIGTERM)

                # Wait for graceful shutdown
                try:
                    self.vllm_process.wait(timeout=10)
                    print("[vLLM] ✓ Shutdown complete")
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.vllm_process.pid), signal.SIGKILL)
                    self.vllm_process.wait()
                    print("[vLLM] ✓ Terminated")
            except Exception as e:
                print(f"[vLLM] Warning: Cleanup error: {e}")

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

        # Note: Model config will be generated per-dataset in the loop below
        # to support dataset-specific max_out_len auto-adjustment

        # Update work_dir to point to experiment directory
        original_work_dir = self.args.work_dir
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

            # Generate dataset-specific model config if needed
            if hasattr(self.args, 'model_config') and self.args.model_config:
                self.custom_model_config_path = self._create_model_config_override(dataset_name=dataset)

            try:
                if not self.launch_vllm():
                    failed_datasets.append(dataset)
                    self.args.datasets = original_datasets
                    continue

                success = self.run_aisbench(dataset_idx=idx, total_datasets=total_datasets)

                if not success:
                    failed_datasets.append(dataset)

            except KeyboardInterrupt:
                print("\n\n[Runner] Interrupted by user")
                self.cleanup()
                return 1
            except Exception as e:
                print(f"[Runner] Error running {dataset}: {e}")
                failed_datasets.append(dataset)
            finally:
                self.cleanup()
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Integrated vLLM + AISBench Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using benchmark config (recommended)
  python run.py --config-file configs/benchmarks/qwen3-30b-accuracy.yaml

  # Override specific datasets
  python run.py --config-file configs/benchmarks/qwen3-30b-accuracy.yaml --datasets ceval mmlu

  # Command-line only
  python run.py --model-path Qwen/Qwen3-30B-A3B --datasets gsm8k --tensor-parallel-size 2
        """
    )

    # Configuration file
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to YAML configuration file. CLI arguments override config file values.'
    )

    # vLLM arguments
    vllm_group = parser.add_argument_group('vLLM Arguments')
    vllm_group.add_argument(
        '--model-path',
        type=str,
        help='Path or name of the model to serve with vLLM'
    )
    vllm_group.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind the vLLM server (default: localhost)'
    )
    vllm_group.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind the vLLM server (default: 8000)'
    )
    vllm_group.add_argument(
        '--tensor-parallel-size',
        type=int,
        help='Number of tensor parallel replicas'
    )
    vllm_group.add_argument(
        '--pipeline-parallel-size',
        type=int,
        help='Number of pipeline parallel stages'
    )
    vllm_group.add_argument(
        '--quantization',
        type=str,
        choices=['awq', 'gptq', 'squeezellm', 'fp8', 'int8'],
        help='Quantization method to use'
    )
    vllm_group.add_argument(
        '--rope-scaling',
        type=json.loads,
        help='RoPE scaling configuration as JSON string, e.g., \'{"type": "dynamic", "factor": 2.0}\''
    )
    vllm_group.add_argument(
        '--max-model-len',
        type=int,
        help='Maximum sequence length for the model'
    )
    vllm_group.add_argument(
        '--gpu-memory-utilization',
        type=float,
        help='GPU memory utilization (0-1, default: 0.9)'
    )
    vllm_group.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Trust remote code when loading the model'
    )
    vllm_group.add_argument(
        '--dtype',
        type=str,
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='Data type for model weights and activations'
    )
    vllm_group.add_argument(
        '--max-num-seqs',
        type=int,
        help='Maximum number of sequences per iteration'
    )
    vllm_group.add_argument(
        '--enable-prefix-caching',
        action='store_true',
        help='Enable prefix caching'
    )
    vllm_group.add_argument(
        '--disable-log-requests',
        action='store_true',
        help='Disable request logging'
    )
    vllm_group.add_argument(
        '--tokenizer',
        type=str,
        help='Path to custom tokenizer'
    )
    vllm_group.add_argument(
        '--revision',
        type=str,
        help='Model revision to use'
    )
    vllm_group.add_argument(
        '--served-model-name',
        type=str,
        help='Model name used in API'
    )
    vllm_group.add_argument(
        '--vllm-extra-args',
        type=str,
        help='Additional vLLM arguments as a single string (e.g., "--arg1 value1 --arg2 value2")'
    )
    vllm_group.add_argument(
        '--vllm-timeout',
        type=int,
        default=300,
        help='Timeout in seconds to wait for vLLM to be ready (default: 300)'
    )
    vllm_group.add_argument(
        '--vllm-log-file',
        type=str,
        help='File to write vLLM logs (default: stdout)'
    )

    # AISBench arguments
    ais_group = parser.add_argument_group('AISBench Arguments')
    ais_group.add_argument(
        '--datasets',
        nargs='+',
        help='Dataset(s) to evaluate (e.g., gsm8k mmlu)'
    )
    ais_group.add_argument(
        '--ais-model',
        type=str,
        help='AISBench model config to use (default: vllm_api_general_chat)'
    )
    ais_group.add_argument(
        '--mode',
        type=str,
        choices=['all', 'infer', 'eval', 'viz', 'perf', 'perf_viz'],
        default='all',
        help='AISBench running mode (default: all)'
    )
    ais_group.add_argument(
        '--summarizer',
        type=str,
        help='Summarizer type for results (e.g., "example")'
    )
    ais_group.add_argument(
        '--merge-ds',
        action='store_true',
        help='Merge dataset results'
    )
    ais_group.add_argument(
        '--work-dir',
        type=str,
        help='Working directory for outputs'
    )
    ais_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    ais_group.add_argument(
        '--max-num-workers',
        type=int,
        help='Maximum number of workers'
    )
    ais_group.add_argument(
        '--num-prompts',
        type=int,
        help='Number of prompts to evaluate'
    )
    ais_group.add_argument(
        '--dump-eval-details',
        action='store_true',
        help='Dump evaluation details'
    )
    ais_group.add_argument(
        '--config',
        type=str,
        help='Path to custom AISBench config file'
    )
    ais_group.add_argument(
        '--ais-extra-args',
        type=str,
        help='Additional AISBench arguments as a single string'
    )

    args = parser.parse_args()
    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Load and merge config file if provided
    if args.config_file:
        if not Path(args.config_file).exists():
            print(f"[Error] Config file not found: {args.config_file}")
            sys.exit(1)
        try:
            # Try new-style benchmark config first
            config = load_benchmark_config(args.config_file)
            args = merge_config_with_args(config, args)

            # Display config info
            if hasattr(args, '_benchmark_name'):
                print(f"[Setup] Loaded benchmark: {args._benchmark_name}")
            print(f"[Setup] Config file: {args.config_file}\n")
        except Exception as e:
            print(f"[Error] Failed to load config: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Validate required arguments
    if not args.model_path:
        print("[Error] --model-path is required (either via CLI or config file)")
        sys.exit(1)
    if not args.datasets and not args.config:
        print("[Error] Either --datasets or --config must be specified (either via CLI or config file)")
        sys.exit(1)

    runner = VLLMAISBenchRunner(args)
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
