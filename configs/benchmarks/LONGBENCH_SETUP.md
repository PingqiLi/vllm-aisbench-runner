# LongBench v2 Evaluation Setup

## Overview

LongBench v2是一个长文本理解评测基准，用于评估大语言模型的长上下文能力（8k-2M words）。

本配置与Qwen3官方评测设置对齐：
- **Context Length**: 128k tokens（使用YaRN RoPE scaling factor 4）
- **Evaluation Mode**: Without CoT（非思考模式）
- **Model**: Qwen3-30B-A3B BF16

## 配置文件说明

### 1. 模型配置: `configs/models/qwen3-30b-a3b-bf16-128k.yaml`

关键配置：
```yaml
vllm:
  max_model_len: 131072  # 128k context
  rope_scaling: '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
```

**YaRN RoPE Scaling说明**:
- 原始context length: 32768 (32k)
- Scaling factor: 4.0
- 扩展后context length: 131072 (128k)
- 与LongBench v2官方评测对齐

### 2. 数据集配置: `configs/datasets/longbenchv2.yaml`

```yaml
dataset:
  name: "longbenchv2_gen_0_shot_chat_prompt"  # Without CoT version
  description: "Long context understanding (without CoT)"
```

**Without CoT说明**:
- 使用非思考模式（non-thinking mode）
- Prompt中不包含 "Let's think step by step"
- 直接要求模型给出答案

### 3. Benchmark配置: `configs/benchmarks/qwen3-30b-longbench.yaml`

完整的LongBench v2评测配置，组合了上述模型和数据集配置。

## 使用方法

### 1. 准备数据集

```bash
# 安装huggingface_hub
pip install huggingface_hub

# 下载数据集（会自动下载到ais_bench/datasets/LongBench-v2/）
cd /path/to/ais_bench
huggingface-cli download zai-org/LongBench-v2 --repo-type dataset --local-dir datasets/LongBench-v2

# 验证数据集结构
tree datasets/LongBench-v2/
# 应该看到：
# LongBench-v2/
# └── data.json
```

### 2. 运行评测

```bash
# 运行完整的LongBench v2评测
python run.py --config-file configs/benchmarks/qwen3-30b-longbench.yaml

# 快速测试（只评测5个样本）
python run.py --config-file configs/benchmarks/qwen3-30b-longbench.yaml --num-prompts 5
```

### 3. 结果输出

评测结果会保存在：
```
outputs/qwen3_30b_longbench/
└── longbenchv2/           # 简化后的目录名
    ├── predictions/       # 模型预测结果
    ├── results/          # 评测分数
    └── configs/          # 运行配置快照
```

## 官方Qwen3评测设置

根据LongBench v2官方文档，Qwen3模型的评测设置：

1. **Context Length**: 128k
   - 使用YaRN扩展，scaling factor为4
   - 原始32k → 扩展至128k

2. **Evaluation Mode**:
   - w/o CoT: 非思考模式（本配置）
   - w/ CoT: 思考模式，16K token thinking budget（如需使用需要另外配置）

3. **支持Hybrid Thinking**:
   - Qwen3支持混合思考模式
   - 本配置使用w/o CoT结果，对应非思考模式

## 硬件要求

- **最小GPU数量**: 2
- **每卡显存**: 60GB
- **推荐配置**: 2x A100 80GB 或 2x NPU

**注意**: LongBench v2包含超长上下文（最长2M words），建议：
- 设置足够的`max_model_len`（推荐131072+）
- 监控显存使用情况
- 如果输入超过上下文限制，预测结果将为空

## AISBench配置文件位置

### 安装自定义配置

自定义的without CoT配置保存在benchmark_runner项目中，需要手动复制到AISBench：

```bash
# 从benchmark_runner复制到ais_bench
cp configs/ais_bench_patches/longbenchv2/*.py \
   /path/to/ais_bench/benchmark/configs/datasets/longbenchv2/
```

**配置文件说明：**
```
benchmark_runner/configs/ais_bench_patches/longbenchv2/
├── longbenchv2_gen_0_shot_chat_prompt.py      # 入口文件
└── longbenchv2_gen_wo_cot.py                  # 实现文件（without CoT）
```

复制后，AISBench目录结构：
```
ais_bench/benchmark/configs/datasets/longbenchv2/
├── longbenchv2_gen.py                          # 原始配置（with CoT）
├── longbenchv2_gen_0_shot_chat_prompt.py      # 新增：without CoT入口
└── longbenchv2_gen_wo_cot.py                  # 新增：without CoT实现
```

## 参考资料

- LongBench v2官方主页: https://longbench2.github.io/
- LongBench v2数据集: https://huggingface.co/datasets/zai-org/LongBench-v2
- Qwen3官方文档: https://qwenlm.github.io/
- YaRN论文: NeurIPS 2023
