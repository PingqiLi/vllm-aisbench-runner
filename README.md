# vLLM AISBench Runner

一体化 vLLM + AISBench 评测工具，基于 task-based 架构实现自动化评测流程。

## 1. 环境准备

### 1.1 安装依赖

#### Python 依赖
```bash
pip install -r requirements.txt
```

#### vLLM
根据硬件平台选择：
```bash
# 默认使用vllm-ascend镜像
pip install vllm
pip install vllm-ascend
```

#### AISBench
```bash
git clone -b quant_eval https://github.com/PingqiLi/ais-bench.git
cd ais-bench
pip install -e ./ --use-pep517
```

### 1.2 准备数据集

```bash
# 下载数据集到 AISBench 目录
./prepare_datasets.sh /path/to/ais-bench

# 示例
./prepare_datasets.sh ~/ais-bench
```

## 2. 标准评测 (`run.py`)

适用于 BF16/W8A8/W4A4 等标准精度评测，以及性能测试。

### 快速开始

```bash
# 精度评测 (BF16)
python run.py --config-file configs/suites/qwen3-30b-bf16-acc.yaml

# 精度评测 (W4A4 量化)
python run.py --config-file configs/suites/qwen3-30b-w4a4-acc.yaml

# 性能测试 (BF16)
python run.py --config-file configs/suites/qwen3-30b-bf16-perf.yaml

```

### 常用命令

#### 自定义模型路径
```bash
python run.py \
    --config-file configs/suites/qwen3-30b-bf16-acc.yaml \
    --model-path /path/to/custom/model
```

#### 自定义 vLLM 参数
```bash
python run.py \
    --config-file configs/suites/qwen3-30b-bf16-acc.yaml \
    --tensor-parallel-size 4 \
    --port 8080
```

#### 调试模式（限制数据量）
```bash
python run.py \
    --config-file configs/suites/qwen3-30b-bf16-acc.yaml \
    --num-prompts 10 \
    --debug
```

### 配置文件结构

```
configs/
├── suites/              # 评测套件 (入口配置)
│   ├── qwen3-30b-bf16-acc.yaml
│   └── ...
├── tasks/               # 任务配置 (模型+数据集组合)
│   ├── qwen3-30b-bf16/
│   │   ├── ceval.yaml
│   │   └── ...
└── ais_bench_patches/   # AISBench 自定义配置补丁
```

### 输出目录结构

```
outputs/qwen3_30b_bf16_acc/
└── 2025-01-15_14-30-45/          # 实验时间戳
    ├── configs/                   # 配置快照 (可复现)
    ├── vllm_logs/                 # vLLM 服务日志
    ├── ceval/                     # 评测结果
    └── ...
```

## 3. 并行评测 (`parallel_eval.py`)

`parallel_eval.py` 是一个独立的评测脚本，专门用于在多卡环境下并行运行多个 TP=1 的 vLLM 实例，从而显著加速 W4A4 量化模型的评测过程。

### 核心特性

- **自动资源分配**: 自动分配 NPU (ASCEND_RT_VISIBLE_DEVICES) 和端口。
- **数据自动切分**: 支持 CEval (CSV) 和自定义数据集 (JSONL) 的 Stride 切分。
- **配置自动注入**: 自动生成专属配置文件。
- **结果聚合**: 自动收集所有实例结果。

### 使用方法

#### 3.1 使用标准数据集 (如 CEval)

```bash
python parallel_eval.py \
  --rank 0,1,2,3 \
  --model-path /path/to/Qwen3-30B-A3B-Instruct \
  --dataset ceval \
  --quantization ascend  # 如果是W4A4权重
```

#### 3.2 使用自定义采样数据集 (推荐)

结合 `tools/create_sampled_dataset.py` 生成的数据集使用：

```bash
# 1. 生成采样数据集 (如 MCQ 类型)
python tools/create_sampled_dataset.py --output datasets/custom_eval ...

# 2. 并行评测
python parallel_eval.py \
  --rank 0,1,2,3 \
  --model-path /path/to/Qwen3-30B-A3B-Instruct \
  --custom-dataset-path datasets/custom_eval_mcq.jsonl \
  --batch-size 16 \
  --quantization ascend
```

**参数说明**:
- `--rank`: 使用的NPU卡号列表 (如 `0,1,2,3`)
- `--model-path`: 模型路径
- `--custom-dataset-path`: 自定义数据集路径 (支持 JSONL)
- `--batch-size`: 指定 ais_bench 的 batch size
- `--quantization`: 量化方式 (如 `ascend` 用于 W4A4)
- `--enforce-eager`: 脚本默认开启，强制使用 eager 模式以保证精度

### 输出目录结构

```
outputs/parallel_ceval_<timestamp>/
├── ceval_split_0/          # 切分数据集
├── instance_0/             # AISBench 评测结果
├── vllm_rank0_port8000.log # vLLM 日志
└── ais_bench_instance0.log # AISBench 日志
```

### 注意事项

- **仅限 W4A4/TP=1**: 如果模型支持 TP>1 (如 BF16 TP=8)，请直接使用 `run.py`。
- **结果聚合**: 脚本会收集所有实例的 Partial Result，但不会自动重新计算整体准确率。

## 4. 常见问题

### 1. MATH500 评测报错：ModuleNotFoundError

```bash
pip install latex2sympy2_extended math_verify
```

### 2. 显存不足 (OOM)

降低配置（修改 task 配置文件中的 vllm 参数）：
```yaml
vllm:
  gpu_memory_utilization: 0.85
  max_num_seqs: 128
  max_model_len: 16384

aisbench:
  max_num_workers: 8
```

### 3. 端口占用

```bash
# 使用不同端口
python run.py --config-file config.yaml --port 8080
```

### 4. vLLM 启动超时

```bash
python run.py --config-file config.yaml --vllm-timeout 900
```
