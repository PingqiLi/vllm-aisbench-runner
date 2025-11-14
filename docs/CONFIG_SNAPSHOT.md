# Configuration Snapshot和实验复现

## 概述

从现在开始，每次运行benchmark时，系统会自动保存完整的配置快照，确保实验可复现。

## 保存的文件结构

运行benchmark后，会在实验目录下生成以下文件：

```
outputs/qwen3_30b_bf16_acc/
└── configs/
    ├── config_original.yaml       # 原始benchmark配置文件（引用）
    ├── config_snapshot.yaml       # 完整展开的配置（包含所有参数）
    ├── metadata.yaml              # 运行时元数据（时间、环境、版本等）
    ├── reproduce.sh               # 一键复现脚本（可执行）
    └── per_dataset/               # 每个数据集的实际配置
        ├── ceval.yaml
        ├── mmlu.yaml
        ├── longbenchv2.yaml       # 包含应用的128k override
        └── ...
```

## 各文件说明

### 1. config_original.yaml

原始的benchmark配置文件副本，仅供参考。

示例：
```yaml
benchmark:
  name: "qwen3-30b-acc"

model_config: "configs/models/qwen3-30b-a3b-bf16.yaml"

datasets:
  - "configs/datasets/ceval.yaml"
  - "configs/datasets/longbenchv2.yaml"
```

### 2. config_snapshot.yaml

**最重要的文件** - 完整展开的配置，包含：
- 所有vLLM参数的实际值
- 所有AISBench参数
- 每个数据集的配置（包括vllm_config_override）

示例：
```yaml
benchmark:
  name: qwen3-30b-acc
  timestamp: '2025-11-15T10:30:00'

vllm:
  model_path: Qwen/Qwen3-30B-A3B
  host: localhost
  port: 8000
  tensor_parallel_size: 2
  max_model_len: 32768
  dtype: bfloat16
  gpu_memory_utilization: 0.9
  trust_remote_code: true

aisbench:
  datasets:
    - ceval_gen_0_shot_cot_chat_prompt
    - longbenchv2_gen_0_shot_chat_prompt
  mode: all
  max_num_workers: 8

datasets:
  ceval_gen_0_shot_cot_chat_prompt:
    description: "Chinese comprehensive evaluation"
    model_config:
      max_out_len: 2048

  longbenchv2_gen_0_shot_chat_prompt:
    description: "Long context understanding (without CoT)"
    model_config:
      max_out_len: 256
    vllm_config_override:
      max_model_len: 131072
      rope_scaling: '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
```

### 3. metadata.yaml

运行时环境信息，用于排查环境差异：

```yaml
runtime:
  timestamp: '2025-11-15T10:30:00'
  hostname: gpu-server-01
  platform: Linux-5.15.0-x86_64
  python_version: 3.10.12
  working_directory: /home/user/benchmark_runner

versions:
  python: 3.10.12
  vllm: 0.6.0
  torch: 2.1.0
  transformers: 4.36.0

command:
  executable: run.py
  args: ['--config-file', 'configs/benchmarks/qwen3-30b-acc.yaml']
  full_command: python run.py --config-file configs/benchmarks/qwen3-30b-acc.yaml
```

### 4. reproduce.sh

**可执行的复现脚本**，直接运行即可复现实验：

```bash
#!/bin/bash
# Reproduction script for benchmark run
# Generated: 2025-11-15T10:30:00

set -e  # Exit on error

# Check if running from correct directory
if [ ! -f "run.py" ]; then
    echo "Error: Please run this script from the benchmark_runner directory"
    exit 1
fi

# Run the benchmark
python run.py \
    --config-file configs/benchmarks/qwen3-30b-acc.yaml
```

使用方法：
```bash
cd /path/to/benchmark_runner
./outputs/qwen3_30b_bf16_acc/configs/reproduce.sh
```

### 5. per_dataset/*.yaml

**每个数据集实际使用的配置**，包含所有override后的参数。

示例 - `per_dataset/longbenchv2.yaml`：
```yaml
dataset:
  name: longbenchv2_gen_0_shot_chat_prompt

vllm:
  model_path: Qwen/Qwen3-30B-A3B
  host: localhost
  port: 8000
  tensor_parallel_size: 2
  max_model_len: 131072                    # 已应用128k override
  rope_scaling: '{"rope_type":"yarn",...}' # 已应用YaRN scaling
  dtype: bfloat16
  gpu_memory_utilization: 0.9

aisbench:
  model: vllm_api_general_chat
  num_prompts: null

model_config:
  max_out_len: 256
  generation_kwargs:
    temperature: 0.0
    seed: 42

vllm_config_override_applied:
  max_model_len: 131072
  rope_scaling: '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
```

**对比** - `per_dataset/ceval.yaml`（标准32k配置）：
```yaml
dataset:
  name: ceval_gen_0_shot_cot_chat_prompt

vllm:
  model_path: Qwen/Qwen3-30B-A3B
  max_model_len: 32768  # 标准配置，无override
  ...
```

## 如何复现实验

### 方法1：使用reproduce.sh（推荐）

```bash
cd /path/to/benchmark_runner
./outputs/qwen3_30b_bf16_acc/configs/reproduce.sh
```

### 方法2：使用config_snapshot.yaml

虽然config_snapshot.yaml包含完整信息，但无法直接作为--config-file使用（因为它是展开后的格式）。
它的主要用途是：
- 查看实际使用的所有参数
- 了解每个数据集的配置差异
- 调试配置问题

### 方法3：复现单个数据集

使用per_dataset下的配置文件：

```bash
# 复现longbenchv2评测（128k context）
python run.py \
    --model-path Qwen/Qwen3-30B-A3B \
    --datasets longbenchv2_gen_0_shot_chat_prompt \
    --max-model-len 131072 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --num-prompts 100
```

## 配置验证

查看某次运行实际使用的配置：

```bash
# 查看完整配置
cat outputs/qwen3_30b_bf16_acc/configs/config_snapshot.yaml

# 查看longbenchv2的实际配置
cat outputs/qwen3_30b_bf16_acc/configs/per_dataset/longbenchv2.yaml

# 对比不同数据集的vLLM配置
diff outputs/qwen3_30b_bf16_acc/configs/per_dataset/ceval.yaml \
     outputs/qwen3_30b_bf16_acc/configs/per_dataset/longbenchv2.yaml
```

## 最佳实践

1. **每次重要实验后检查config_snapshot.yaml**
   - 确认所有参数符合预期
   - 特别注意vllm_config_override是否正确应用

2. **保存实验结果时包含configs目录**
   ```bash
   tar -czf experiment_results.tar.gz \
       outputs/qwen3_30b_bf16_acc/configs/ \
       outputs/qwen3_30b_bf16_acc/*/results/
   ```

3. **比较不同实验的配置差异**
   ```bash
   diff experiment1/configs/config_snapshot.yaml \
        experiment2/configs/config_snapshot.yaml
   ```

4. **使用reproduce.sh快速复现**
   - 适合验证结果
   - 适合在不同环境中测试

## 故障排查

### Q: reproduce.sh运行失败？

检查：
1. 是否在正确的目录运行（需要在benchmark_runner目录）
2. 原始配置文件是否还存在（config_original.yaml中引用的路径）
3. 环境是否与metadata.yaml中记录的一致

### Q: 如何知道longbenchv2用的是128k还是32k？

查看：
```bash
cat outputs/.../configs/per_dataset/longbenchv2.yaml | grep max_model_len
```

如果显示131072，说明使用了128k配置。

### Q: 想修改某个参数重新运行？

1. 复制reproduce.sh
2. 编辑脚本，添加或修改参数
3. 运行修改后的脚本

## 总结

新的配置保存机制提供：
- ✅ 完整的参数记录（包括override）
- ✅ 每个数据集的实际配置
- ✅ 一键复现能力
- ✅ 环境信息追踪
- ✅ 配置验证和对比工具
