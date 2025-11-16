"""
vLLM service management utilities
"""

import os
import sys
import json
import time
import signal
import subprocess
import requests
from typing import List, Optional
from datetime import datetime


class VLLMManager:
    """Manager for vLLM service lifecycle."""

    def __init__(self, args, experiment_dir: str):
        """
        Initialize vLLM manager.

        Args:
            args: Parsed command-line arguments
            experiment_dir: Directory for this experiment session
        """
        self.args = args
        self.experiment_dir = experiment_dir
        self.vllm_process: Optional[subprocess.Popen] = None
        self._log_file_handle = None

    def build_command(self) -> List[str]:
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
            # rope_scaling is already a JSON string from config
            if isinstance(self.args.rope_scaling, str):
                cmd.extend(["--rope-scaling", self.args.rope_scaling])
            else:
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

    def wait_for_ready(self, timeout: int = 300, log_file_path: str = None) -> bool:
        """
        Wait for vLLM service to be ready by checking the health endpoint.

        Args:
            timeout: Maximum time to wait in seconds
            log_file_path: Path to log file for debugging

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
                if log_file_path:
                    print(f"[vLLM] Check logs for details: {log_file_path}")
                    # Show last 50 lines of log
                    try:
                        with open(log_file_path, 'r') as f:
                            lines = f.readlines()
                            last_lines = lines[-50:]
                            print("\n[vLLM] Last 50 lines of log:")
                            print("".join(last_lines))
                    except Exception:
                        pass
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

    def kill_existing(self):
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
        except Exception:
            pass  # Silently ignore if no processes found

    def _get_log_file_path(self, dataset_name: Optional[str] = None) -> str:
        """
        Get log file path for current benchmark run.
        Creates separate log files for each dataset to prevent overwriting.

        Args:
            dataset_name: Current dataset being benchmarked

        Returns:
            Path to log file
        """
        if self.args.vllm_log_file:
            # User specified a log file
            base_log = self.args.vllm_log_file

            if dataset_name:
                # Append dataset name to avoid overwriting
                base_name = os.path.splitext(base_log)[0]
                ext = os.path.splitext(base_log)[1] or '.log'

                # Clean dataset name for filename
                safe_dataset = dataset_name.replace('_gen_0_shot_cot_chat_prompt.py', '') \
                                          .replace('.py', '') \
                                          .replace('/', '_')

                return f"{base_name}_{safe_dataset}{ext}"
            else:
                return base_log
        else:
            # Default log location in experiment directory
            log_dir = os.path.join(self.experiment_dir, "vllm_logs")
            os.makedirs(log_dir, exist_ok=True)

            if dataset_name:
                safe_dataset = dataset_name.replace('_gen_0_shot_cot_chat_prompt.py', '') \
                                          .replace('.py', '') \
                                          .replace('/', '_')
                return os.path.join(log_dir, f"vllm_{safe_dataset}.log")
            else:
                return os.path.join(log_dir, "vllm.log")

    def launch(self, dataset_name: Optional[str] = None) -> bool:
        """
        Launch vLLM service.

        Args:
            dataset_name: Current dataset name (for log file naming)

        Returns:
            True if launch successful, False otherwise
        """
        self.kill_existing()

        cmd = self.build_command()
        log_file_path = self._get_log_file_path(dataset_name)

        print(f"\n[vLLM] Starting service... (TP={self.args.tensor_parallel_size}, dtype={self.args.dtype})")
        print(f"[vLLM] Command: {' '.join(cmd)}")
        print(f"[vLLM] Logs: {log_file_path}")

        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            # Open log file
            self._log_file_handle = open(log_file_path, 'w')

            # Launch vLLM in a separate process
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=self._log_file_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )

            # Wait for service to be ready
            if not self.wait_for_ready(timeout=self.args.vllm_timeout, log_file_path=log_file_path):
                return False

            return True

        except Exception as e:
            print(f"[vLLM] ✗ Failed to launch: {e}")
            if self._log_file_handle:
                self._log_file_handle.close()
                self._log_file_handle = None
            return False

    def shutdown(self):
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
            finally:
                # Close log file handle
                if self._log_file_handle:
                    self._log_file_handle.close()
                    self._log_file_handle = None
                self.vllm_process = None
