#!/usr/bin/env python3
"""
Parallel benchmark evaluation using multiple vLLM instances.

Run multiple TP=1 vLLM instances on different NPUs/GPUs in parallel to speed up evaluation.

Usage:
    python parallel_eval.py --rank 0,1,2,3 --model-path /path/to/model --dataset ceval
    python parallel_eval.py --rank 0,1,2,3 --model-path /path/to/model --custom-dataset-path datasets/custom_eval_mcq.jsonl

Features:
- Automatic port allocation (8000, 8001, 8002, ...)
- Dataset splitting across instances (supports CEval CSV and Custom JSONL)
- Auto-patch vllm_api_general_chat.py per instance (port + batch_size)
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
from typing import List, Dict, Optional, Tuple
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelEvaluator:
    def __init__(self, ranks: List[int], model_path: str, dataset: str,
                 base_port: int = 8000, quantization: str = None,
                 custom_dataset_path: str = None, batch_size: int = None):
        self.ranks = ranks
        self.model_path = model_path
        self.dataset = dataset
        self.base_port = base_port
        self.quantization = quantization
        
        # Support multiple custom datasets (comma separated or directory)
        if custom_dataset_path:
            path_obj = Path(custom_dataset_path)
            if path_obj.is_dir():
                # If directory, find all .jsonl files
                self.custom_datasets = sorted([str(p) for p in path_obj.glob('*.jsonl')])
                print(f"[Init] Found {len(self.custom_datasets)} JSONL files in {custom_dataset_path}")
            else:
                # Comma separated list of files
                self.custom_datasets = [p.strip() for p in custom_dataset_path.split(',')]
        else:
            self.custom_datasets = []
            
        self.batch_size = batch_size
        self.num_instances = len(ranks)

        self.vllm_processes = []
        
        # Output directory setup
        if self.custom_datasets:
            dataset_name = "custom_suite"
        else:
            dataset_name = dataset
            
        self.output_dir = f"outputs/parallel_{dataset_name}_{int(time.time())}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Track sample counts for aggregation: {dataset_path: {instance_id: count}}
        self.split_counts = {} 

        print(f"[Init] Running {self.num_instances} parallel instances")
        print(f"[Init] NPU ranks: {ranks}")
        print(f"[Init] Ports: {[self.base_port + i for i in range(self.num_instances)]}")
        print(f"[Init] Output: {self.output_dir}")
        if self.custom_datasets:
            print(f"[Init] Custom Datasets: {self.custom_datasets}")

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
            '--enforce-eager',  # Enforce eager mode for accuracy
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
        """
        original_dataset_path = self.find_dataset_path()

        # Create temporary split dataset directory
        split_dataset_dir = os.path.join(
            self.output_dir,
            f'ceval_split_{instance_id}'
        )
        os.makedirs(split_dataset_dir, exist_ok=True)

        print(f"[Dataset Split] Creating split dataset for instance {instance_id}")
        
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

    def split_custom_dataset(self, dataset_path_str: str, instance_id: int) -> Tuple[str, Optional[str], int]:
        """
        Split custom JSONL dataset for this instance.
        
        Returns:
            (split_jsonl_path, split_meta_path, sample_count)
        """
        input_path = Path(dataset_path_str)
        if not input_path.exists():
            print(f"[Error] Custom dataset not found: {input_path}")
            sys.exit(1)

        # Output file path: outputs/parallel_xxx/dataset_name/split_x.jsonl
        dataset_name = input_path.stem
        dataset_out_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)
        
        split_filename = f"{dataset_name}_split_{instance_id}{input_path.suffix}"
        split_path = os.path.join(dataset_out_dir, split_filename)

        print(f"[Dataset Split] Processing {dataset_name} for instance {instance_id}")
        
        # Read all lines
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_samples = len(lines)
        
        # Select lines for this instance (stride-based)
        split_lines = [
            line for idx, line in enumerate(lines)
            if idx % self.num_instances == instance_id
        ]
        
        # Write split file
        with open(split_path, 'w', encoding='utf-8') as f:
            f.writelines(split_lines)
            
        # Handle meta.json if exists
        meta_path = input_path.with_suffix(input_path.suffix + '.meta.json')
        split_meta_path = None
        
        if meta_path.exists():
            split_meta_path = split_path + '.meta.json'
            shutil.copy2(meta_path, split_meta_path)
            
        return split_path, split_meta_path, len(split_lines)

    def create_split_dataset_config(self, instance_id: int, split_dataset_dir: str) -> str:
        """
        Create a custom dataset config file that points to the split dataset (for CEval).
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

            # Return the config name (just filename, no directory prefix)
            dataset_config_name = custom_config_filename[:-3]  # Remove .py extension

            return dataset_config_name

        except Exception as e:
            print(f"[Error] Failed to create custom config: {e}")
            sys.exit(1)

    def create_model_config(self, instance_id: int, port: int) -> str:
        """
        Create a custom model config file with specific port and batch size for this instance.
        """
        try:
            import ais_bench
            ais_bench_root = os.path.dirname(ais_bench.__file__)
            original_config = os.path.join(
                ais_bench_root,
                'benchmark/configs/models/vllm_api/vllm_api_general_chat.py'
            )

            # Use existing vllm_api config directory
            vllm_api_config_dir = os.path.join(
                ais_bench_root,
                'benchmark/configs/models/vllm_api'
            )

            # Custom config file name with unique identifier
            custom_config_filename = f'vllm_api_parallel_{instance_id}_general_chat.py'
            custom_config_path = os.path.join(vllm_api_config_dir, custom_config_filename)

            # Read original config
            with open(original_config, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace port
            import re
            content = re.sub(
                r'host_port\s*=\s*\d+',
                f'host_port = {port}',
                content
            )
            
            # Replace batch size if specified
            if self.batch_size is not None:
                content = re.sub(
                    r'batch_size\s*=\s*\d+',
                    f'batch_size = {self.batch_size}',
                    content
                )

            # Write custom config
            with open(custom_config_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Return the config name (without .py extension)
            model_config_name = custom_config_filename[:-3]

            return model_config_name

        except Exception as e:
            print(f"[Error] Failed to create custom model config: {e}")
            sys.exit(1)

    def run_worker(self, instance_id: int, port: int) -> bool:
        """
        Worker function for a single parallel instance.
        Runs ais_bench for each dataset sequentially.
        """
        # Create custom model config with specific port and batch size
        model_config_name = self.create_model_config(instance_id, port)
        
        success = True
        
        # 1. Handle Custom Datasets (List)
        if self.custom_datasets:
            for dataset_path in self.custom_datasets:
                try:
                    # Split dataset
                    split_dataset_path, split_meta_path, sample_count = self.split_custom_dataset(dataset_path, instance_id)
                    
                    # Record sample count for aggregation
                    if dataset_path not in self.split_counts:
                        self.split_counts[dataset_path] = {}
                    self.split_counts[dataset_path][instance_id] = sample_count
                    
                    if sample_count == 0:
                        print(f"[Worker {instance_id}] Skipping empty split for {dataset_path}")
                        continue

                    # Prepare ais_bench args
                    dataset_args = ['--custom-dataset-path', split_dataset_path]
                    if split_meta_path:
                        dataset_args.extend(['--custom-dataset-meta-path', split_meta_path])
                    
                    # Output directory: outputs/parallel_xxx/dataset_name/instance_x
                    dataset_name = Path(dataset_path).stem
                    work_dir = os.path.join(self.output_dir, dataset_name, f'instance_{instance_id}')
                    
                    print(f"[Worker {instance_id}] Running {dataset_name} ({sample_count} samples)")
                    
                    if not self._run_ais_bench_cmd(instance_id, model_config_name, work_dir, dataset_args):
                        success = False
                        
                except Exception as e:
                    print(f"[Worker {instance_id}] Error processing {dataset_path}: {e}")
                    success = False

        # 2. Handle CEval (Standard)
        else:
            split_dataset_dir = self.split_ceval_dataset(instance_id)
            dataset_config_name = self.create_split_dataset_config(instance_id, split_dataset_dir)
            dataset_args = ['--datasets', dataset_config_name]
            
            work_dir = os.path.join(self.output_dir, 'ceval', f'instance_{instance_id}')
            print(f"[Worker {instance_id}] Running CEval")
            
            if not self._run_ais_bench_cmd(instance_id, model_config_name, work_dir, dataset_args):
                success = False

        return success

    def _run_ais_bench_cmd(self, instance_id: int, model_config: str, work_dir: str, dataset_args: List[str]) -> bool:
        """Helper to execute ais_bench command."""
        cmd = [
            'ais_bench',
            '--models', model_config,
            '--mode', 'all',
            '--work-dir', work_dir,
            '--max-num-workers', '1',
            '--merge-ds',
        ] + dataset_args

        log_file = os.path.join(work_dir, 'ais_bench.log')
        os.makedirs(work_dir, exist_ok=True)
        
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
        return result.returncode == 0

    def cleanup_custom_configs(self):
        """Clean up temporary dataset and model configs from ais_bench directory."""
        try:
            import ais_bench
            ais_bench_root = os.path.dirname(ais_bench.__file__)

            # Remove dataset config files (only for CEval mode)
            if not self.custom_datasets:
                ceval_config_dir = os.path.join(
                    ais_bench_root,
                    'benchmark/configs/datasets/ceval'
                )
                for i in range(self.num_instances):
                    config_file = os.path.join(
                        ceval_config_dir,
                        f'ceval_parallel_split_{i}_gen_0_shot_cot_chat_prompt.py'
                    )
                    if os.path.exists(config_file):
                        os.remove(config_file)

            # Remove model config files
            vllm_api_config_dir = os.path.join(
                ais_bench_root,
                'benchmark/configs/models/vllm_api'
            )
            for i in range(self.num_instances):
                config_file = os.path.join(
                    vllm_api_config_dir,
                    f'vllm_api_parallel_{i}_general_chat.py'
                )
                if os.path.exists(config_file):
                    os.remove(config_file)

        except Exception as e:
            print(f"[Cleanup] Warning: Failed to remove temporary configs: {e}")

    def shutdown_vllm(self):
        """Shutdown all vLLM processes."""
        print(f"\n[Cleanup] Shutting down {len(self.vllm_processes)} vLLM instances...")
        for i, process in enumerate(self.vllm_processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

    def aggregate_results(self) -> Dict:
        """
        Aggregate results from all instances and calculate global accuracy.
        """
        print(f"\n{'='*80}")
        print("AGGREGATED RESULTS")
        print(f"{'='*80}")

        final_summary = {}

        # List of datasets to aggregate
        datasets_to_process = self.custom_datasets if self.custom_datasets else ['ceval']

        for dataset_key in datasets_to_process:
            dataset_name = Path(dataset_key).stem if self.custom_datasets else 'ceval'
            print(f"\nDataset: {dataset_name}")
            
            total_correct = 0
            total_samples = 0
            instance_results = []
            
            for i in range(self.num_instances):
                # Locate result file
                if self.custom_datasets:
                    result_dir = os.path.join(self.output_dir, dataset_name, f'instance_{i}')
                else:
                    result_dir = os.path.join(self.output_dir, 'ceval', f'instance_{i}')
                
                # Find summary json
                summary_file = None
                if os.path.exists(result_dir):
                    for root, dirs, files in os.walk(result_dir):
                        for file in files:
                            if 'summary' in file.lower() and file.endswith('.json'):
                                summary_file = os.path.join(root, file)
                                break
                
                if summary_file:
                    try:
                        with open(summary_file, 'r') as f:
                            data = json.load(f)
                        
                        # Try to extract accuracy/correct count
                        # Logic: If we have correct_num, use it. 
                        # If not, use accuracy * split_sample_count
                        
                        correct = 0
                        count = 0
                        
                        # Strategy 1: Look for explicit counts (common in some evaluators)
                        if 'correct_num' in data:
                            correct = data['correct_num']
                            count = data.get('total_num', 0)
                        elif 'correct' in data:
                            correct = data['correct']
                            count = data.get('total', 0)
                        
                        # Strategy 2: Look for accuracy and use our tracked sample count
                        elif 'accuracy' in data or 'acc' in data or 'score' in data:
                            acc = data.get('accuracy', data.get('acc', data.get('score', 0)))
                            # Normalize to 0-1
                            if acc > 1.0: acc /= 100.0
                                
                            # Get tracked count
                            if self.custom_datasets:
                                count = self.split_counts.get(dataset_key, {}).get(i, 0)
                            else:
                                # For CEval, we don't strictly track count in this script version, 
                                # but we can infer or just accept the partial result.
                                # Fallback: assume equal weight if count missing? 
                                # Better: try to find count in data
                                count = data.get('num_samples', data.get('count', 0))
                            
                            correct = acc * count
                        
                        total_correct += correct
                        total_samples += count
                        
                        instance_results.append({
                            'instance': i,
                            'correct': correct,
                            'total': count,
                            'raw_data': data
                        })
                        
                    except Exception as e:
                        print(f"  Instance {i}: Failed to parse result: {e}")
                else:
                    print(f"  Instance {i}: No result found")

            # Calculate Global Accuracy
            if total_samples > 0:
                global_acc = (total_correct / total_samples) * 100
                print(f"  Total Samples: {total_samples}")
                print(f"  Total Correct: {total_correct:.2f}")
                print(f"  Global Accuracy: {global_acc:.2f}%")
                
                final_summary[dataset_name] = {
                    'accuracy': global_acc,
                    'total_samples': total_samples,
                    'total_correct': total_correct
                }
            else:
                print("  No valid samples found for aggregation.")

        return final_summary

    def run(self):
        """Main execution."""
        try:
            # Step 0: Prepare dataset splits (show info)
            print(f"\n{'='*80}")
            print("STEP 0: Preparation")
            print(f"{'='*80}")
            print(f"[Info] Instances: {self.num_instances}")
            if self.custom_datasets:
                print(f"[Info] Datasets: {len(self.custom_datasets)} files")
                for ds in self.custom_datasets:
                    print(f"  - {ds}")

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
            print("STEP 3: Running ais_bench in parallel")
            print(f"{'='*80}")

            with ThreadPoolExecutor(max_workers=self.num_instances) as executor:
                futures = []
                for i, rank in enumerate(self.ranks):
                    port = self.base_port + i
                    future = executor.submit(self.run_worker, i, port)
                    futures.append(future)

                # Wait for all to complete
                results = []
                for future in as_completed(futures):
                    results.append(future.result())

            # Aggregate results
            print(f"\n{'='*80}")
            print("STEP 4: Aggregating results")
            print(f"{'='*80}")

            self.aggregate_results()

            print(f"\nAll results saved to: {self.output_dir}")

            return all(results)

        except KeyboardInterrupt:
            print("\n[Interrupted] Shutting down...")
            return False
        finally:
            self.shutdown_vllm()
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
        help='Dataset name (default: ceval, ignored if --custom-dataset-path is set)'
    )
    parser.add_argument(
        '--custom-dataset-path',
        type=str,
        help='Path to custom JSONL dataset (overrides --dataset)'
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
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for ais_bench (overrides default in config)'
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
    if args.custom_dataset_path:
        print(f"Custom Dataset: {args.custom_dataset_path}")
    else:
        print(f"Dataset: {args.dataset}")
    print(f"NPU Ranks: {ranks}")
    print(f"Ports: {[args.base_port + i for i in range(len(ranks))]}")
    if args.quantization:
        print(f"Quantization: {args.quantization}")
    if args.batch_size:
        print(f"Batch Size: {args.batch_size}")
    print(f"{'='*80}\n")

    evaluator = ParallelEvaluator(
        ranks=ranks,
        model_path=args.model_path,
        dataset=args.dataset,
        base_port=args.base_port,
        quantization=args.quantization,
        custom_dataset_path=args.custom_dataset_path,
        batch_size=args.batch_size
    )

    success = evaluator.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
