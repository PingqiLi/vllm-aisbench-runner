# vLLM AISBench Runner

一体化 vLLM + AISBench 评测工具，基于 task-based 架构实现自动化评测流程。

## 特性

- 自动管理 vLLM 服务生命周期
- Task-based 架构：每个任务是完整的模型+数据集配置
- 支持精度评测和性能测试
- 配置快照：自动保存完整配置以确保实验可复现
- 零代码扩展：添加新评测只需创建配置文件

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
python run.py --config-file configs/suites/qwen3-30b-bf16-acc.yaml

# 精度评测 (W4A4 量化)
python run.py --config-file configs/suites/qwen3-30b-w4a4-acc.yaml

# 精度评测 (W8A8 量化)
python run.py --config-file configs/suites/qwen3-30b-w8a8-acc.yaml

# 性能测试 (BF16)
python run.py --config-file configs/suites/qwen3-30b-bf16-perf.yaml

# 性能测试 (W4A4)
python run.py --config-file configs/suites/qwen3-30b-w4a4-perf.yaml
```

## 配置文件结构

项目采用 task-based 配置架构：

```
configs/
├── suites/              # 评测套件 (入口配置)
│   ├── qwen3-30b-bf16-acc.yaml
│   ├── qwen3-30b-w4a4-acc.yaml
│   ├── qwen3-30b-w8a8-acc.yaml
│   ├── qwen3-30b-bf16-perf.yaml
│   ├── qwen3-30b-w4a4-perf.yaml
│   └── qwen3-30b-w8a8-perf.yaml
├── tasks/               # 任务配置 (模型+数据集组合)
│   ├── qwen3-30b-bf16/
│   │   ├── ceval.yaml
│   │   ├── mmlu.yaml
│   │   └── ...
│   ├── qwen3-30b-w4a4/
│   └── qwen3-30b-w8a8/
└── ais_bench_patches/   # AISBench 自定义配置补丁
    ├── longbenchv2/
    └── livecodebench/
```

详细配置说明请参考：[configs/README.md](configs/README.md)

### 配置文件示例

#### Suite 配置 (`configs/suites/qwen3-30b-bf16-acc.yaml`)
```yaml
suite:
  name: "qwen3-30b-bf16-acc"
  description: "Full accuracy evaluation for Qwen3-30B-A3B (BF16)"
  type: "accuracy"

tasks:
  - "configs/tasks/qwen3-30b-bf16/ceval.yaml"
  - "configs/tasks/qwen3-30b-bf16/mmlu.yaml"
  - "configs/tasks/qwen3-30b-bf16/aime2024.yaml"
  - "configs/tasks/qwen3-30b-bf16/gpqa.yaml"
  - "configs/tasks/qwen3-30b-bf16/math500.yaml"
  - "configs/tasks/qwen3-30b-bf16/livecodebench.yaml"
  - "configs/tasks/qwen3-30b-bf16/longbenchv2.yaml"

output:
  work_dir: "outputs/qwen3_30b_bf16_acc"
```

#### Task 配置 (`configs/tasks/qwen3-30b-bf16/ceval.yaml`)
```yaml
task:
  name: "qwen3-30b-bf16-ceval"
  model: "Qwen3-30B-A3B"
  precision: "bf16"
  dataset: "ceval"

vllm:
  model_path: "Qwen/Qwen3-30B-A3B"
  tensor_parallel_size: 8
  dtype: "bfloat16"
  max_model_len: 32768
  gpu_memory_utilization: 0.95
  trust_remote_code: true
  max_num_seqs: 256
  enable_prefix_caching: true
  disable_log_requests: true

aisbench:
  dataset: "ceval_gen_0_shot_cot_chat_prompt"
  model: "vllm_api_general_chat"
  mode: "all"
  max_num_workers: 16
  max_out_len: 2048
  generation_kwargs:
    temperature: 0.0
    seed: 42
```

## 常用命令

### 自定义模型路径
```bash
python run.py \
    --config-file configs/suites/qwen3-30b-bf16-acc.yaml \
    --model-path /path/to/custom/model
```

### 自定义 vLLM 参数
```bash
python run.py \
    --config-file configs/suites/qwen3-30b-bf16-acc.yaml \
    --tensor-parallel-size 4 \
    --port 8080
```

### 调试模式（限制数据量）
```bash
python run.py \
    --config-file configs/suites/qwen3-30b-bf16-acc.yaml \
    --num-prompts 10 \
    --debug
```

## 支持的数据集

### 精度评测
- **ceval** - 中文综合评测
- **mmlu** - 英文知识理解
- **aime2024** - 高级数学推理
- **gpqa** - 研究生级别科学问答
- **math500** - 高级数学
- **livecodebench** - 代码生成
- **longbenchv2** - 长文本理解 (128k context)

### 性能评测
- **synthetic-perf** - 合成数据性能测试 (mode=perf)

## 输出目录结构

每次运行会创建带时间戳的实验目录：

```
outputs/qwen3_30b_bf16_acc/
└── 2025-01-15_14-30-45/          # 实验时间戳
    ├── configs/                   # 配置快照
    │   ├── config_snapshot.yaml   # 完整可复现配置
    │   ├── config_original.yaml   # 原始配置文件
    │   ├── metadata.yaml          # 运行环境信息
    │   └── per_dataset/           # 每个数据集的配置
    │       ├── ceval.yaml
    │       └── ...
    ├── vllm_logs/                 # vLLM 服务日志
    │   ├── vllm_ceval.log
    │   ├── vllm_mmlu.log
    │   └── ...
    ├── ceval/                     # 评测结果
    ├── mmlu/
    └── ...
```

## 实验复现

配置快照保证了实验的完全可复现性：

```bash
# 使用保存的配置快照复现实验
python run.py --config-file outputs/qwen3_30b_bf16_acc/2025-01-15_14-30-45/configs/config_snapshot.yaml
```

## 添加新配置

### 添加新数据集任务

1. 为每个模型精度创建任务配置：
```bash
configs/tasks/qwen3-30b-bf16/new_dataset.yaml
configs/tasks/qwen3-30b-w4a4/new_dataset.yaml
configs/tasks/qwen3-30b-w8a8/new_dataset.yaml
```

2. 将任务添加到对应的 suite 配置中：
```yaml
tasks:
  - "configs/tasks/qwen3-30b-bf16/new_dataset.yaml"
```

### 添加新模型配置

1. 创建新的任务目录：
```bash
mkdir configs/tasks/new-model/
```

2. 为每个数据集创建任务配置

3. 创建对应的 suite 配置：
```bash
configs/suites/new-model-acc.yaml
```

**无需修改任何代码！**

## 常见问题

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

# 或手动 kill
pkill -f vllm
```

### 4. vLLM 启动超时

```bash
python run.py --config-file config.yaml --vllm-timeout 900
```

### 5. LiveCodeBench 或 LongBench v2 配置问题

需要应用 AISBench 补丁：

```bash
# 复制补丁到 AISBench 安装目录
cp configs/ais_bench_patches/longbenchv2/*.py \
   /path/to/ais_bench/benchmark/configs/datasets/longbenchv2/

cp configs/ais_bench_patches/livecodebench/livecodebench.py \
   /path/to/ais_bench/benchmark/datasets/livecodebench/
```

## 硬件配置建议

| 精度 | 模型显存 | TP | Workers |
|------|---------|-----|---------|
| BF16 | ~60GB | 8 | 16 |
| W4A4 | ~15GB | 8 | 16 |
| W8A8 | ~30GB | 8 | 16 |

## License

Apache License 2.0

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [AISBench](https://gitee.com/ascend/ais-bench) - Ascend AI 评测工具
- [OpenCompass](https://github.com/open-compass/opencompass) - 大模型评测框架
