# vLLM AISBench Runner

一体化 vLLM + AISBench 评测工具，支持模块化配置架构和自动化评测流程。

## 特性

- 自动管理 vLLM 服务生命周期
- 模块化配置：分离模型、数据集和评测组合
- 支持精度评测和性能测试
- 零代码扩展：添加新模型/数据集只需创建配置文件

## 快速开始

### 1. 安装依赖

#### 安装 Python 依赖
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

#### 安装 AISBench
```bash
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e ./ --use-pep517
```

### 2. 准备数据集

```bash
# 下载数据集到 AISBench 目录
./prepare_datasets.sh /path/to/benchmark

# 示例
./prepare_datasets.sh ~/benchmark
```

### 3. 运行评测

```bash
# 精度评测 (BF16)
python run.py --config-file configs/benchmarks/qwen3-30b-acc.yaml

# 精度评测 (W4A4 量化)
python run.py --config-file configs/benchmarks/qwen3-30b-w4a4-acc.yaml

# 性能测试
python run.py --config-file configs/benchmarks/qwen3-30b-perf.yaml
```

## 配置文件结构

项目采用三层模块化配置：

```
configs/
├── models/          # 模型配置 (vLLM 参数、硬件需求)
│   ├── qwen3-30b-a3b-bf16.yaml
│   ├── qwen3-30b-a3b-w4a4.yaml
│   └── qwen3-30b-a3b-perf.yaml
├── datasets/        # 数据集配置 (评测参数、max_out_len)
│   ├── ceval.yaml
│   ├── mmlu.yaml
│   ├── gpqa.yaml
│   ├── aime2024.yaml
│   ├── math500.yaml
│   └── ...
└── benchmarks/      # 评测组合 (模型 + 数据集列表)
    ├── qwen3-30b-acc.yaml
    ├── qwen3-30b-w4a4-acc.yaml
    └── qwen3-30b-perf.yaml
```

### 配置文件示例

#### Dataset 配置 (`configs/datasets/gpqa.yaml`)
```yaml
dataset:
  name: "gpqa_gen_0_shot_cot_chat_prompt.py"
  description: "Graduate-level scientific reasoning benchmark"

model_config:
  max_out_len: 4096  # 针对该数据集的输出长度
  generation_kwargs:
    temperature: 0.0
    seed: 42
```

#### Model 配置 (`configs/models/qwen3-30b-a3b-bf16.yaml`)
```yaml
model:
  name: "Qwen3-30B-A3B"
  precision: "bf16"

vllm:
  model_path: "Qwen/Qwen3-30B-A3B"
  tensor_parallel_size: 2
  dtype: "bfloat16"
  max_model_len: 32768
  # ...其他 vLLM 参数

aisbench:
  model: "vllm_api_general_chat"
  max_num_workers: 8
```

#### Benchmark 配置 (`configs/benchmarks/qwen3-30b-accuracy.yaml`)
```yaml
benchmark:
  name: "qwen3-30b-accuracy"
  description: "Full accuracy evaluation for Qwen3-30B-A3B"
  type: "accuracy"

model_config: "configs/models/qwen3-30b-a3b-bf16.yaml"

datasets:
  - "configs/datasets/ceval.yaml"
  - "configs/datasets/mmlu.yaml"
  - "configs/datasets/gpqa.yaml"
  - "configs/datasets/math500.yaml"
  # ...

output:
  work_dir: "outputs/qwen3_30b_bf16_accuracy"
```

## 常用命令

### 选择特定数据集
```bash
python run.py \
    --config-file configs/benchmarks/qwen3-30b-acc.yaml \
    --datasets ceval_gen_0_shot_cot_chat_prompt.py mmlu_gen_0_shot_cot_chat_prompt.py
```

### 自定义 vLLM 参数
```bash
python run.py \
    --config-file configs/benchmarks/qwen3-30b-acc.yaml \
    --tensor-parallel-size 4 \
    --port 8080
```

### 快速验证配置（推荐）
使用 debug 模式快速验证配置正确性：
```bash
# 使用快速测试脚本（只跑 5 条数据）
./quick_test.sh configs/benchmarks/qwen3-30b-acc.yaml

# 或手动指定
python run.py \
    --config-file configs/benchmarks/qwen3-30b-acc.yaml \
    --debug \
    --num-prompts 5
```

## 添加新模型/数据集

### 添加新数据集

创建 `configs/datasets/新数据集.yaml`:
```yaml
dataset:
  name: "新数据集_gen_0_shot_cot_chat_prompt.py"
  description: "数据集说明"

model_config:
  max_out_len: 2048
  generation_kwargs:
    temperature: 0.0
    seed: 42
```

### 添加新模型

1. 创建 `configs/models/新模型.yaml`:
```yaml
model:
  name: "新模型名称"
  precision: "bf16"

vllm:
  model_path: "HuggingFace/Model-Name"
  tensor_parallel_size: 2
  dtype: "bfloat16"
  # ...

aisbench:
  model: "vllm_api_general_chat"
  max_num_workers: 8
```

2. 创建 `configs/benchmarks/新模型-accuracy.yaml`:
```yaml
benchmark:
  name: "新模型-accuracy"

model_config: "configs/models/新模型.yaml"

datasets:
  - "configs/datasets/ceval.yaml"
  - "configs/datasets/mmlu.yaml"
  # ...

output:
  work_dir: "outputs/新模型_accuracy"
```

3. 运行评测：
```bash
python run.py --config-file configs/benchmarks/新模型-accuracy.yaml
```

**无需修改任何代码！**

## 支持的数据集

| 数据集 | 类型 | 说明 |
|--------|------|------|
| ceval | 精度 | 中文综合评测 |
| mmlu | 精度 | 英文知识评测 |
| gpqa | 精度 | 科学问答 |
| aime2024 | 精度 | 数学推理 |
| math500 | 精度 | 数学问题 (需安装 `latex2sympy2_extended`) |
| livecodebench | 精度 | 代码生成 |
| longbenchv2 | 精度 | 长文本评测 |
| gsm8k | 性能 | 数学推理性能测试 |
| synthetic | 性能 | 合成数据性能测试 |

## 常见问题

### 1. MATH500 评测报错：ModuleNotFoundError

```bash
pip install latex2sympy2_extended math_verify
```

### 2. 显存不足 (OOM)

降低配置：
```yaml
vllm:
  gpu_memory_utilization: 0.85
  max_num_seqs: 128
  max_model_len: 16384

aisbench:
  max_num_workers: 4
```

### 3. 端口占用

```bash
# 使用不同端口
python run.py --config-file config.yaml --port 8080

# 或手动 kill
pkill -f vllm
```

### 4. vLLM 启动超时

```bash
python run.py --config-file config.yaml --vllm-timeout 600
```

## 硬件配置

| 精度 | 模型显存 | 推荐 TP | 说明 |
|------|---------|---------|------|
| BF16 | ~60GB | 2 | 两卡并行 |
| W4A4 | ~15GB | 1 | 单卡运行 |

## 高级配置

详细的配置架构说明、扩展指南和最佳实践，请参考 [CONFIGURATION.md](CONFIGURATION.md)。

## License

Apache License 2.0

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [AISBench](https://gitee.com/ascend/ais-bench) - Ascend AI 评测工具
- [OpenCompass](https://github.com/open-compass/opencompass) - 大模型评测框架
