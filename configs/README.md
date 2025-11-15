# Configuration Directory Structure

这个目录包含了benchmark_runner项目的所有配置文件，采用**task-based架构**。

## 目录结构

```
configs/
├── suites/              # 评测套件 (入口配置)
├── tasks/               # 任务配置 (模型+数据集组合)
└── ais_bench_patches/   # AISBench自定义配置补丁
```

## 使用方式

### 1. 运行完整评测套件

**精度评测（Accuracy）：**
```bash
# BF16完整精度评测
python run.py --config-file configs/suites/qwen3-30b-bf16-acc.yaml

# W4A4量化评测
python run.py --config-file configs/suites/qwen3-30b-w4a4-acc.yaml

# W8A8量化评测
python run.py --config-file configs/suites/qwen3-30b-w8a8-acc.yaml
```

**性能评测（Performance）：**
```bash
# BF16性能测试
python run.py --config-file configs/suites/qwen3-30b-bf16-perf.yaml

# W4A4性能测试
python run.py --config-file configs/suites/qwen3-30b-w4a4-perf.yaml

# W8A8性能测试
python run.py --config-file configs/suites/qwen3-30b-w8a8-perf.yaml
```

### 2. 运行单个任务

```bash
# 单独运行某个数据集评测
python run.py --config-file configs/tasks/qwen3-30b-bf16/ceval.yaml
```

## 配置文件说明

### suites/ - 评测套件

评测套件是**入口配置**，包含一组相关的任务。

**文件列表：**

精度评测（Accuracy）：
- `qwen3-30b-bf16-acc.yaml` - BF16精度评测
- `qwen3-30b-w4a4-acc.yaml` - W4A4量化评测
- `qwen3-30b-w8a8-acc.yaml` - W8A8量化评测

性能评测（Performance）：
- `qwen3-30b-bf16-perf.yaml` - BF16性能测试
- `qwen3-30b-w4a4-perf.yaml` - W4A4性能测试
- `qwen3-30b-w8a8-perf.yaml` - W8A8性能测试

**配置格式：**
```yaml
suite:
  name: "qwen3-30b-bf16-acc"
  description: "Full accuracy evaluation"
  type: "accuracy"

tasks:
  - "configs/tasks/qwen3-30b-bf16/ceval.yaml"
  - "configs/tasks/qwen3-30b-bf16/mmlu.yaml"
  # ... 更多任务

output:
  work_dir: "outputs/qwen3_30b_bf16_acc"
```

### tasks/ - 任务配置

每个任务是一个**完整的模型+数据集配置**，包含所有vLLM和AISBench参数。

**目录结构：**
- `qwen3-30b-bf16/` - BF16精度任务
- `qwen3-30b-w4a4/` - W4A4量化任务
- `qwen3-30b-w8a8/` - W8A8量化任务

**支持的数据集：**

精度评测任务：
- `ceval.yaml` - 中文综合评测
- `mmlu.yaml` - 英文知识理解
- `aime2024.yaml` - 高级数学推理
- `gpqa.yaml` - 研究生级别科学问答
- `math500.yaml` - 高级数学
- `livecodebench.yaml` - 代码生成
- `longbenchv2.yaml` - 长文本理解 (128k context)

性能评测任务：
- `synthetic-perf.yaml` - 合成数据性能测试 (mode=perf)

**配置格式：**
```yaml
task:
  name: "qwen3-30b-bf16-ceval"
  model: "Qwen3-30B-A3B"
  precision: "bf16"
  dataset: "ceval"

vllm:
  model_path: "Qwen/Qwen3-30B-A3B"
  tensor_parallel_size: 2
  dtype: "bfloat16"
  # ... 完整的vLLM参数

aisbench:
  dataset: "ceval_gen_0_shot_cot_chat_prompt"
  model: "vllm_api_general_chat"
  # ... 完整的AISBench参数
```

### ais_bench_patches/ - AISBench补丁

某些数据集需要自定义配置来覆盖AISBench默认行为。

**当前补丁：**
- `longbenchv2/` - LongBench v2的without CoT配置
- `livecodebench/` - LiveCodeBench的version_tag修复

**使用方式：**
```bash
# 手动复制到AISBench安装目录
cp configs/ais_bench_patches/longbenchv2/*.py \
   /path/to/ais_bench/benchmark/configs/datasets/longbenchv2/

cp configs/ais_bench_patches/livecodebench/livecodebench.py \
   /path/to/ais_bench/benchmark/datasets/livecodebench/
```

## 添加新配置

### 添加新的数据集任务

1. 为每个模型精度创建任务配置：
   ```bash
   configs/tasks/qwen3-30b-bf16/new_dataset.yaml
   configs/tasks/qwen3-30b-w4a4/new_dataset.yaml
   configs/tasks/qwen3-30b-w8a8/new_dataset.yaml
   ```

2. 将任务添加到对应的suite配置中：
   ```yaml
   tasks:
     - "configs/tasks/qwen3-30b-bf16/new_dataset.yaml"
   ```

### 添加新的模型配置

1. 创建新的任务目录：
   ```bash
   mkdir configs/tasks/new-model/
   ```

2. 为每个数据集创建任务配置

3. 创建对应的suite配置：
   ```bash
   configs/suites/new-model-acc.yaml
   ```

## 架构优势

✅ **清晰的入口**：suites/目录是唯一的入口
✅ **完全自包含**：每个task包含所有运行参数
✅ **易于复现**：可以直接运行单个task或完整suite
✅ **参数隔离**：不同精度的配置完全独立，避免冲突
✅ **易于扩展**：添加新数据集或新模型配置都很简单
