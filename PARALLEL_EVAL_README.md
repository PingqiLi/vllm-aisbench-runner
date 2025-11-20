# Parallel Evaluation Script (parallel_eval.py)

Independent script for running multiple TP=1 vLLM instances in parallel to speed up W4A4 evaluation.

## Background

W4A4 quantization only supports TP=1, making single-instance evaluation slow. This script enables 4x speedup by:
- Launching multiple vLLM serve instances on different NPUs
- Splitting the CEval dataset across instances (stride-based)
- Running ais_bench in parallel
- Aggregating results

## Features

- ✅ Automatic NPU rank assignment via ASCEND_RT_VISIBLE_DEVICES
- ✅ Automatic port allocation (8000, 8001, 8002, ...)
- ✅ **Dataset splitting**: Stride-based CSV splitting (instance 0 → samples 0,4,8,12...)
- ✅ Config patching: Automatic vllm_api_general_chat.py port patching per instance
- ✅ Parallel execution: ThreadPoolExecutor for concurrent ais_bench runs
- ✅ Result aggregation: Collects results from all instances

## Usage

### Basic Example (4 NPUs)

```bash
python parallel_eval.py \
  --rank 0,1,2,3 \
  --model-path /path/to/Qwen3-30B-A3B-W4A4 \
  --dataset ceval \
  --quantization ascend
```

### Parameters

- `--rank`: Comma-separated NPU ranks (e.g., `0,1,2,3`)
- `--model-path`: Path to model
- `--dataset`: Dataset name (default: `ceval`)
- `--base-port`: Base port number (default: `8000`)
- `--quantization`: Quantization method (e.g., `ascend` for W4A4)

### How It Works

1. **Dataset Splitting**: Splits CEval CSV files using stride pattern:
   - Instance 0: samples 0, 4, 8, 12, ...
   - Instance 1: samples 1, 5, 9, 13, ...
   - Instance 2: samples 2, 6, 10, 14, ...
   - Instance 3: samples 3, 7, 11, 15, ...

2. **vLLM Launch**: Each instance runs:
   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 vllm serve /path/to/model --port 8000 --tensor-parallel-size 1
   ASCEND_RT_VISIBLE_DEVICES=1 vllm serve /path/to/model --port 8001 --tensor-parallel-size 1
   ...
   ```

3. **Config Patching**: Modifies `vllm_api_general_chat.py` to use correct port for each instance

4. **Parallel Execution**: Runs ais_bench concurrently on all instances

5. **Result Aggregation**: Collects partial results from all instances

## Output Structure

```
outputs/parallel_ceval_<timestamp>/
├── ceval_split_0/          # Split dataset for instance 0
├── ceval_split_1/          # Split dataset for instance 1
├── ...
├── config_instance_0/      # Custom dataset config for instance 0
├── config_instance_1/      # Custom dataset config for instance 1
├── ...
├── instance_0/             # AISBench results for instance 0
├── instance_1/             # AISBench results for instance 1
├── ...
├── vllm_rank0_port8000.log
├── vllm_rank1_port8001.log
├── ...
├── ais_bench_instance0.log
├── ais_bench_instance1.log
└── ...
```

## Notes

- **W4A4 Only**: This script is specifically designed for W4A4 (TP=1 limitation)
- **CEval Focused**: Currently hardcoded for CEval dataset (can be extended)
- **Independent**: Standalone script, doesn't use the main vllm-aisbench-runner framework
- **Cleanup**: Automatically restores original configs and shuts down vLLM on exit

## Limitations

- Currently only supports CEval dataset (hardcoded dataset structure)
- Result aggregation is simple (just collects all partial results)
- No automatic accuracy recalculation across splits
- Requires manual interpretation of aggregated results

## Example Output

```
================================================================================
PARALLEL BENCHMARK EVALUATION
================================================================================
Model: /workspace/weights/Qwen3-30B-A3B-W4A4
Dataset: ceval
NPU Ranks: [0, 1, 2, 3]
Ports: [8000, 8001, 8002, 8003]
Quantization: ascend
================================================================================

[Init] Running 4 parallel instances
[Init] NPU ranks: [0, 1, 2, 3]
[Init] Ports: [8000, 8001, 8002, 8003]
[Init] Output: outputs/parallel_ceval_1700000000

================================================================================
STEP 0: Dataset splitting preparation
================================================================================
[Info] Dataset will be split across 4 instances
[Info] Each instance will process every 4th sample
[Info] Example: Instance 0 → samples 0,4,8...

================================================================================
STEP 1: Launching vLLM instances
================================================================================
[vLLM] Launching on NPU 0, port 8000
...
```
