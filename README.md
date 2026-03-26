# vLLM AISBench Runner

一体化 vLLM + AISBench 评测工具，基于 task-based 架构实现自动化评测流程。

## 1. 功能

- **精度评测**：通过 AISBench 运行 CEval、MMLU、AIME2024、GPQA、MATH500、LiveCodeBench、LongBenchV2 等数据集
- **性能评测**：合成数据集吞吐/延迟测试
- **PPL 评测**：基于 wikitext-2 的 Perplexity 评估，支持量化模型与 BF16 基线对比
- **重复实验**：suite 配置 `repeat: N` 支持同一任务多次运行，用于可靠性测试
- **自动化流程**：自动启停 vLLM 服务、注入 AISBench 配置、保存实验快照

## 2. 环境准备

```bash
pip install -r requirements.txt

# vLLM (Ascend)
pip install vllm vllm-ascend

# AISBench
git clone -b quant_eval https://github.com/PingqiLi/ais-bench.git
cd ais-bench && pip install -e ./ --use-pep517
```

准备数据集：
```bash
./prepare_datasets.sh /path/to/ais-bench
```

## 3. 使用

### 3.1 精度/性能评测（`run.py`）

```bash
# 精度评测
python run.py --config-file configs/suites/qwen3-30b-bf16-acc.yaml

# 性能评测
python run.py --config-file configs/suites/qwen3-30b-bf16-perf.yaml

# 自定义模型路径
python run.py --config-file configs/suites/qwen3-30b-w4a4-acc.yaml --model-path /path/to/model

# 调试（限制数据量）
python run.py --config-file configs/suites/qwen3-30b-bf16-acc.yaml --num-prompts 10 --debug
```

也可以直接运行单个 task 配置：
```bash
python run.py --config-file configs/tasks/qwen3-30b-bf16/ceval.yaml
```

### 3.2 PPL 评测（`tools/eval_ppl.py`）

```bash
# 单模型 PPL
python tools/eval_ppl.py --model-path /path/to/model

# 量化模型 + BF16 基线对比
python tools/eval_ppl.py \
    --model-path /path/to/quantized_model \
    --quantization ascend --enforce-eager \
    --baseline-model-path /path/to/bf16_model

# 使用缓存的基线 PPL（跳过 BF16 重新评测）
python tools/eval_ppl.py \
    --model-path /path/to/quantized_model \
    --quantization ascend --enforce-eager \
    --baseline-ppl 7.52
```

### 3.3 常用 CLI 参数

| 参数 | 说明 |
|------|------|
| `--config-file` | suite 或 task 配置文件路径 |
| `--model-path` | 覆盖配置中的模型路径 |
| `--tensor-parallel-size` | 覆盖 TP 数 |
| `--port` | vLLM 服务端口 |
| `--num-prompts` | 限制评测数据条数 |
| `--debug` | 调试模式 |
| `--vllm-timeout` | vLLM 启动超时（秒） |

## 4. 配置体系

### 4.1 目录结构

```
configs/
├── suites/              # 评测套件（入口配置）
│   ├── qwen3-30b-bf16-acc.yaml
│   ├── qwen3-30b-w4a4-perf.yaml
│   └── ...
├── tasks/               # 任务配置（模型 + 数据集 + 参数）
│   ├── qwen3-30b-bf16/
│   │   ├── ceval.yaml
│   │   ├── mmlu.yaml
│   │   └── ...
│   └── qwen3-30b-w4a4/
│       └── ...
└── ais_bench_patches/   # AISBench 数据集配置补丁
    ├── longbenchv2/
    └── livecodebench/
```

### 4.2 Suite 配置

Suite 是评测入口，定义一组要运行的 task：

```yaml
suite:
  name: "qwen3-30b-bf16-acc"
  description: "Full accuracy evaluation for Qwen3-30B-A3B (BF16)"
  type: "accuracy"      # accuracy | performance | probe

tasks:
  - "configs/tasks/qwen3-30b-bf16/ceval.yaml"
  - "configs/tasks/qwen3-30b-bf16/mmlu.yaml"
  # ...

output:
  work_dir: "outputs/qwen3_30b_bf16_acc"

# 可选：重复运行（可靠性测试）
# suite:
#   repeat: 5
```

### 4.3 Task 配置

每个 task 是一个完整的、自包含的评测单元：

```yaml
task:
  name: qwen3-30b-bf16-ceval
  model: Qwen3-30B-A3B
  precision: bf16
  dataset: ceval

vllm:
  model_path: Qwen/Qwen3-30B-A3B
  host: localhost
  port: 8000
  tensor_parallel_size: 2
  max_model_len: 32768
  timeout: 600
  # 量化模型额外参数:
  # quantization: ascend
  # enforce_eager: true

aisbench:
  dataset: ceval_gen_0_shot_cot_chat_prompt
  model: vllm_api_general_chat
  mode: all              # all (精度) | perf (性能)
  batch_size: 64
  max_out_len: 32000
  max_num_workers: 16
  merge_ds: true
  dump_eval_details: true

sampling_params:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  min_p: 0
  seed: 42
  repetition_penalty: 1.0
```

## 5. 定义新的评测 Suite

### 5.1 添加新数据集到已有模型

1. 在对应模型目录下创建 task 配置：

```bash
# 以 humaneval 为例
vim configs/tasks/qwen3-30b-bf16/humaneval.yaml
```

```yaml
task:
  name: qwen3-30b-bf16-humaneval
  model: Qwen3-30B-A3B
  precision: bf16
  dataset: humaneval

vllm:
  model_path: Qwen/Qwen3-30B-A3B
  host: localhost
  port: 8000
  tensor_parallel_size: 2
  max_model_len: 32768
  timeout: 600

aisbench:
  dataset: humaneval_gen         # AISBench 中注册的数据集名
  model: vllm_api_general_chat
  mode: all
  batch_size: 64
  max_out_len: 32000
  max_num_workers: 16
  merge_ds: true
  dump_eval_details: true

sampling_params:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  seed: 42
```

2. 将 task 添加到 suite：

```yaml
# configs/suites/qwen3-30b-bf16-acc.yaml
tasks:
  - "configs/tasks/qwen3-30b-bf16/ceval.yaml"
  - "configs/tasks/qwen3-30b-bf16/humaneval.yaml"  # 新增
```

### 5.2 添加新模型的评测 Suite

1. 创建 task 目录和各数据集配置：

```bash
mkdir configs/tasks/qwen3-32b-resq/
# 复制已有配置作为模板，修改 model_path、precision、quantization 等
cp configs/tasks/qwen3-30b-w4a4/ceval.yaml configs/tasks/qwen3-32b-resq/ceval.yaml
```

修改关键字段：
```yaml
task:
  name: qwen3-32b-resq-ceval
  model: Qwen3-32B
  precision: resq

vllm:
  model_path: /path/to/resq-quantized-model
  tensor_parallel_size: 2
  quantization: ascend
  enforce_eager: true
```

2. 创建 suite 配置：

```yaml
# configs/suites/qwen3-32b-resq-acc.yaml
suite:
  name: "qwen3-32b-resq-acc"
  description: "Accuracy evaluation for Qwen3-32B (ResQ W4A8)"
  type: "accuracy"

tasks:
  - "configs/tasks/qwen3-32b-resq/ceval.yaml"
  - "configs/tasks/qwen3-32b-resq/mmlu.yaml"

output:
  work_dir: "outputs/qwen3_32b_resq_acc"
```

3. 运行：

```bash
python run.py --config-file configs/suites/qwen3-32b-resq-acc.yaml
```

### 5.3 AISBench 补丁

部分数据集需要自定义 AISBench 配置（如修改 prompt 模板、version_tag 等）。补丁文件放在 `configs/ais_bench_patches/` 下，手动复制到 AISBench 安装目录：

```bash
cp configs/ais_bench_patches/longbenchv2/*.py /path/to/ais_bench/benchmark/configs/datasets/longbenchv2/
```

## 6. 输出结构

```
outputs/qwen3_30b_bf16_acc/
└── 2025-01-15_14-30-45/          # 实验时间戳
    ├── configs/                   # 配置快照（可复现）
    ├── vllm_logs/                 # vLLM 服务日志
    ├── ceval/                     # 各数据集评测结果
    ├── mmlu/
    └── ...
```

配置快照（`configs/config_snapshot.yaml`）包含完整的 suite + 所有 task 内联配置，可直接用于复现：
```bash
python run.py --config-file outputs/.../configs/config_snapshot.yaml
```

## 7. 常见问题

| 问题 | 解决 |
|------|------|
| MATH500 报 `ModuleNotFoundError` | `pip install latex2sympy2_extended math_verify` |
| OOM | 降低 task 配置中 `gpu_memory_utilization`、`max_num_seqs`、`max_model_len` |
| 端口占用 | `--port 8080` |
| vLLM 启动超时 | `--vllm-timeout 900` |
