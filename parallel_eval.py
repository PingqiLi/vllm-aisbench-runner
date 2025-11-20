#!/usr/bin/env python3
"""
Parallel benchmark evaluation using multiple vLLM instances.

Run multiple TP=1 vLLM instances on different NPUs/GPUs in parallel to speed up evaluation.

Usage:
    python parallel_eval.py --rank 0,1,2,3 --model-path /path/to/model --dataset ceval

Features:
- Automatic port allocation (8000, 8001, 8002, ...)
- Dataset splitting across instances
- Auto-patch vllm_api_general_chat.py per instance
- Parallel ais_bench execution
- Result aggregation
"""

import argparse
import subprocess
import time
import os
import sys
import json
import shutil
import csv
from pathlib import Path
from typing import List, Dict
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelEvaluator:
    def __init__(self, ranks: List[int], model_path: str, dataset: str,
                 base_port: int = 8000, quantization: str = None):
        self.ranks = ranks
        self.model_path = model_path
        self.dataset = dataset
        self.base_port = base_port
        self.quantization = quantization
        self.num_instances = len(ranks)

        self.vllm_processes = []
        self.output_dir = f"outputs/parallel_{dataset}_{int(time.time())}"
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[Init] Running {self.num_instances} parallel instances")
        print(f"[Init] NPU ranks: {ranks}")
        print(f"[Init] Ports: {[self.base_port + i for i in range(self.num_instances)]}")
        print(f"[Init] Output: {self.output_dir}")

    def find_ais_bench_config(self) -> str:
        """Find vllm_api_general_chat.py path."""
        try:
            import ais_bench
            config_path = os.path.join(
                os.path.dirname(ais_bench.__file__),
                'benchmark/configs/models/vllm_api/vllm_api_general_chat.py'
            )
            if os.path.exists(config_path):
                return config_path
            else:
                print(f"[Error] Config not found at: {config_path}")
                sys.exit(1)
        except ImportError:
            print("[Error] ais_bench not installed")
            sys.exit(1)

    def backup_config(self, config_path: str):
        """Backup original config."""
        backup_path = config_path + '.parallel_backup'
        if not os.path.exists(backup_path):
            shutil.copy2(config_path, backup_path)
            print(f"[Config] Backup created: {backup_path}")

    def patch_config_port(self, config_path: str, port: int):
        """Patch vllm_api_general_chat.py with specific port."""
        with open(config_path, 'r') as f:
            content = f.read()

        # Replace host_port
        import re
        content = re.sub(
            r'host_port\s*=\s*\d+',
            f'host_port = {port}',
            content
        )

        with open(config_path, 'w') as f:
            f.write(content)

        print(f"[Config] Patched port to {port}")

    def restore_config(self, config_path: str):
        """Restore original config."""
        backup_path = config_path + '.parallel_backup'
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, config_path)
            print(f"[Config] Restored from backup")

    def launch_vllm(self, rank: int, port: int) -> subprocess.Popen:
        """Launch a vLLM instance on specific rank and port."""
        env = os.environ.copy()
        env['ASCEND_RT_VISIBLE_DEVICES'] = str(rank)

        cmd = [
            'vllm', 'serve', self.model_path,
            '--host', 'localhost',
            '--port', str(port),
            '--tensor-parallel-size', '1',
            '--max-model-len', '32768',
        ]

        if self.quantization:
            cmd.extend(['--quantization', self.quantization])

        log_file = os.path.join(self.output_dir, f'vllm_rank{rank}_port{port}.log')
        log_f = open(log_file, 'w')

        print(f"[vLLM] Launching on NPU {rank}, port {port}")
        print(f"[vLLM] Log: {log_file}")
        print(f"[vLLM] Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True
        )

        return process

    def wait_for_vllm_ready(self, port: int, timeout: int = 600) -> bool:
        """Wait for vLLM to be ready."""
        import requests
        url = f"http://localhost:{port}/v1/models"

        print(f"[vLLM] Waiting for port {port} to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"[vLLM] ✓ Port {port} ready")
                    return True
            except:
                pass
            time.sleep(5)

        print(f"[vLLM] ✗ Port {port} timeout")
        return False

    def find_dataset_path(self) -> str:
        """Find ceval dataset path."""
        try:
            import ais_bench
            ais_bench_root = os.path.dirname(ais_bench.__file__)
            dataset_path = os.path.join(
                ais_bench_root,
                'datasets',
                'ceval',
                'formal_ceval'
            )
            if os.path.exists(dataset_path):
                return dataset_path
            else:
                print(f"[Error] CEval dataset not found at: {dataset_path}")
                sys.exit(1)
        except ImportError:
            print("[Error] ais_bench not installed")
            sys.exit(1)

    def split_ceval_dataset(self, instance_id: int) -> str:
        """
        Split CEval dataset CSV files for this instance.

        Uses stride-based splitting:
        - Instance 0: samples 0, 4, 8, 12, ...
        - Instance 1: samples 1, 5, 9, 13, ...
        - Instance 2: samples 2, 6, 10, 14, ...
        - Instance 3: samples 3, 7, 11, 15, ...

        Returns path to split dataset directory.
        """
        original_dataset_path = self.find_dataset_path()

        # Create temporary split dataset directory
        split_dataset_dir = os.path.join(
            self.output_dir,
            f'ceval_split_{instance_id}'
        )
        os.makedirs(split_dataset_dir, exist_ok=True)

        print(f"[Dataset Split] Creating split dataset for instance {instance_id}")
        print(f"[Dataset Split] Original: {original_dataset_path}")
        print(f"[Dataset Split] Split: {split_dataset_dir}")

        # Split all CSV files in dev/val/test splits
        total_samples = 0
        split_samples = 0

        for split in ['dev', 'val', 'test']:
            split_dir = os.path.join(original_dataset_path, split)
            if not os.path.exists(split_dir):
                continue

            # Create split directory
            split_output_dir = os.path.join(split_dataset_dir, split)
            os.makedirs(split_output_dir, exist_ok=True)

            # Process each CSV file
            for csv_file in os.listdir(split_dir):
                if not csv_file.endswith('.csv'):
                    continue

                input_path = os.path.join(split_dir, csv_file)
                output_path = os.path.join(split_output_dir, csv_file)

                # Read and split CSV
                with open(input_path, 'r', encoding='utf-8') as f_in:
                    reader = csv.reader(f_in)
                    header = next(reader)
                    rows = list(reader)
                    total_samples += len(rows)

                    # Select rows for this instance (stride-based)
                    split_rows = [
                        row for idx, row in enumerate(rows)
                        if idx % self.num_instances == instance_id
                    ]
                    split_samples += len(split_rows)

                # Write split CSV
                with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(header)
                    writer.writerows(split_rows)

        print(f"[Dataset Split] Instance {instance_id}: {split_samples}/{total_samples} samples")

        return split_dataset_dir

    def create_split_dataset_config(self, instance_id: int, split_dataset_dir: str) -> str:
        """
        Create a custom dataset config file that points to the split dataset.

        The config file is placed in ais_bench's ceval config directory.

        Returns dataset config name (e.g., 'ceval/ceval_parallel_split_0_gen_0_shot_cot_chat_prompt').
        """
        try:
            import ais_bench
            ais_bench_root = os.path.dirname(ais_bench.__file__)
            original_config = os.path.join(
                ais_bench_root,
                'benchmark/configs/datasets/ceval/ceval_gen_0_shot_cot_chat_prompt.py'
            )

            # Use existing ceval config directory
            ceval_config_dir = os.path.join(
                ais_bench_root,
                'benchmark/configs/datasets/ceval'
            )

            # Custom config file name with unique identifier
            custom_config_filename = f'ceval_parallel_split_{instance_id}_gen_0_shot_cot_chat_prompt.py'
            custom_config_path = os.path.join(ceval_config_dir, custom_config_filename)

            # Read original config and modify path
            with open(original_config, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace dataset path with split dataset path (use absolute path)
            content = content.replace(
                "path='ais_bench/datasets/ceval/formal_ceval'",
                f"path='{os.path.abspath(split_dataset_dir)}'"
            )

            # Write custom config to ceval directory
            with open(custom_config_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"[Config] Created custom dataset config: {custom_config_path}")

            # Return the config name (without .py extension)
            dataset_config_name = f'ceval/{custom_config_filename[:-3]}'

            return dataset_config_name

        except Exception as e:
            print(f"[Error] Failed to create custom config: {e}")
            sys.exit(1)

    def run_ais_bench(self, instance_id: int, port: int) -> bool:
        """Run ais_bench for one instance with split dataset."""

        # Patch vllm_api config to use this port
        config_path = self.find_ais_bench_config()
        self.patch_config_port(config_path, port)

        # Split dataset for this instance
        split_dataset_dir = self.split_ceval_dataset(instance_id)

        # Create custom dataset config pointing to split dataset
        # Returns config name like 'ceval_parallel/ceval_split_0'
        dataset_config_name = self.create_split_dataset_config(instance_id, split_dataset_dir)

        # Build ais_bench command
        work_dir = os.path.join(self.output_dir, f'instance_{instance_id}')

        cmd = [
            'ais_bench',
            '--models', 'vllm_api_general_chat',
            '--datasets', dataset_config_name,  # Use config name, not path
            '--mode', 'all',
            '--work-dir', work_dir,
            '--max-num-workers', '1',
            '--merge-ds',
        ]

        log_file = os.path.join(self.output_dir, f'ais_bench_instance{instance_id}.log')
        print(f"[AISBench] Running instance {instance_id} (port {port})")
        print(f"[AISBench] Dataset config: {dataset_config_name}")
        print(f"[AISBench] Log: {log_file}")

        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True
            )

        if result.returncode == 0:
            print(f"[AISBench] ✓ Instance {instance_id} completed")
            return True
        else:
            print(f"[AISBench] ✗ Instance {instance_id} failed")
            return False

    def cleanup_custom_configs(self):
        """Clean up temporary dataset configs from ais_bench directory."""
        try:
            import ais_bench
            ais_bench_root = os.path.dirname(ais_bench.__file__)
            ceval_config_dir = os.path.join(
                ais_bench_root,
                'benchmark/configs/datasets/ceval'
            )

            # Remove all parallel split config files
            for i in range(self.num_instances):
                config_file = os.path.join(
                    ceval_config_dir,
                    f'ceval_parallel_split_{i}_gen_0_shot_cot_chat_prompt.py'
                )
                if os.path.exists(config_file):
                    os.remove(config_file)
                    print(f"[Cleanup] Removed temporary config: {config_file}")

        except Exception as e:
            print(f"[Cleanup] Warning: Failed to remove temporary configs: {e}")

    def shutdown_vllm(self):
        """Shutdown all vLLM processes."""
        print(f"\n[Cleanup] Shutting down {len(self.vllm_processes)} vLLM instances...")
        for i, process in enumerate(self.vllm_processes):
            if process.poll() is None:
                print(f"[Cleanup] Stopping vLLM instance {i}")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

    def aggregate_results(self) -> Dict:
        """
        Aggregate results from all instances.

        Since each instance processed a different portion of the dataset,
        we need to combine the results (not just return first one).
        """
        print(f"\n[Results] Aggregating results from {self.num_instances} instances...")

        all_results = []
        for i in range(self.num_instances):
            result_dir = os.path.join(self.output_dir, f'instance_{i}')

            # Find result files
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    if 'summary' in file.lower() and file.endswith('.json'):
                        result_path = os.path.join(root, file)
                        try:
                            with open(result_path, 'r') as f:
                                data = json.load(f)
                                all_results.append({
                                    'instance_id': i,
                                    'result_path': result_path,
                                    'data': data
                                })
                                print(f"[Results] Instance {i}: Found result at {result_path}")
                        except Exception as e:
                            print(f"[Results] Warning: Failed to load {result_path}: {e}")

        if not all_results:
            print("[Results] ⚠️  No results found")
            return {}

        print(f"[Results] ✓ Found {len(all_results)} result(s) from {self.num_instances} instances")

        # For split dataset evaluation, each instance has partial results
        # Return all results for manual inspection
        # (Proper aggregation would require understanding result structure)
        return {
            'instances': len(all_results),
            'results': [r['data'] for r in all_results],
            'note': 'Each instance processed a different portion of the dataset (stride-based split)'
        }

    def run(self):
        """Main execution."""
        config_path = self.find_ais_bench_config()

        try:
            # Backup config
            self.backup_config(config_path)

            # Step 0: Prepare dataset splits (show info)
            print(f"\n{'='*80}")
            print("STEP 0: Dataset splitting preparation")
            print(f"{'='*80}")
            print(f"[Info] Dataset will be split across {self.num_instances} instances")
            print(f"[Info] Each instance will process every {self.num_instances}th sample")
            print(f"[Info] Example: Instance 0 → samples 0,{self.num_instances},{ self.num_instances*2}...")

            # Launch all vLLM instances
            print(f"\n{'='*80}")
            print("STEP 1: Launching vLLM instances")
            print(f"{'='*80}")

            for i, rank in enumerate(self.ranks):
                port = self.base_port + i
                process = self.launch_vllm(rank, port)
                self.vllm_processes.append(process)
                time.sleep(2)  # Stagger starts

            # Wait for all to be ready
            print(f"\n{'='*80}")
            print("STEP 2: Waiting for vLLM instances to be ready")
            print(f"{'='*80}")

            all_ready = True
            for i, rank in enumerate(self.ranks):
                port = self.base_port + i
                if not self.wait_for_vllm_ready(port):
                    all_ready = False
                    break

            if not all_ready:
                print("[Error] Not all vLLM instances ready")
                return False

            # Run ais_bench in parallel (each will split dataset inside)
            print(f"\n{'='*80}")
            print("STEP 3: Running ais_bench in parallel (with dataset splitting)")
            print(f"{'='*80}")

            with ThreadPoolExecutor(max_workers=self.num_instances) as executor:
                futures = []
                for i, rank in enumerate(self.ranks):
                    port = self.base_port + i
                    future = executor.submit(self.run_ais_bench, i, port)
                    futures.append(future)

                # Wait for all to complete
                results = []
                for future in as_completed(futures):
                    results.append(future.result())

            # Aggregate results
            print(f"\n{'='*80}")
            print("STEP 4: Aggregating results")
            print(f"{'='*80}")

            final_result = self.aggregate_results()

            # Print summary
            print(f"\n{'='*80}")
            print("FINAL RESULTS")
            print(f"{'='*80}")
            if final_result:
                print(json.dumps(final_result, indent=2, ensure_ascii=False))

            print(f"\nAll results saved to: {self.output_dir}")

            return all(results)

        except KeyboardInterrupt:
            print("\n[Interrupted] Shutting down...")
            return False
        finally:
            self.shutdown_vllm()
            self.restore_config(config_path)
            self.cleanup_custom_configs()


def main():
    parser = argparse.ArgumentParser(
        description="Parallel benchmark evaluation using multiple vLLM instances"
    )
    parser.add_argument(
        '--rank',
        type=str,
        required=True,
        help='Comma-separated NPU ranks (e.g., 0,1,2,3)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ceval',
        help='Dataset name (default: ceval)'
    )
    parser.add_argument(
        '--base-port',
        type=int,
        default=8000,
        help='Base port number (default: 8000)'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        default=None,
        help='Quantization method (e.g., ascend for W4A4)'
    )

    args = parser.parse_args()

    # Parse ranks
    try:
        ranks = [int(r.strip()) for r in args.rank.split(',')]
    except:
        print("[Error] Invalid rank format. Use: --rank 0,1,2,3")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("PARALLEL BENCHMARK EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"NPU Ranks: {ranks}")
    print(f"Ports: {[args.base_port + i for i in range(len(ranks))]}")
    if args.quantization:
        print(f"Quantization: {args.quantization}")
    print(f"{'='*80}\n")

    evaluator = ParallelEvaluator(
        ranks=ranks,
        model_path=args.model_path,
        dataset=args.dataset,
        base_port=args.base_port,
        quantization=args.quantization
    )

    success = evaluator.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
