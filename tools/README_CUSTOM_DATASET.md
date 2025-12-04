# 自定义采样数据集使用指南

## 快速开始

### 步骤1: 设置环境与生成数据集

首先需要设置 `AIS_BENCH_PATH` 环境变量指向 ais_bench 的安装/源码目录，以便脚本能找到原始数据集。

```bash
export AIS_BENCH_PATH=/path/to/ais_bench
```

**方案 A: 快速验证 (小规模采样)**
*AIME: 10, MATH500: 30, CEval: 30, GPQA: 20, MMLU: 30*

```bash
python3 tools/create_sampled_dataset.py \
  --aime-count 10 \
  --math-count 30 \
  --ceval-count 30 \
  --mmlu-count 30 \
  --gpqa-count 20 \
  --output datasets/custom_eval_small \
  --seed 42
```

**方案 B: 标准验证 (中等规模采样)**
*AIME: 20, MATH500: 80, CEval: 80, GPQA: 80, MMLU: 40*

```bash
python3 tools/create_sampled_dataset.py \
  --aime-count 20 \
  --math-count 80 \
  --ceval-count 80 \
  --mmlu-count 40 \
  --gpqa-count 80 \
  --output datasets/custom_eval_medium \
  --seed 42
```

**生成文件说明**：
每个命令会生成 2 个数据文件（共 4 个文件）以及对应的 meta 信息文件：
```
# 方案 A 生成:
datasets/custom_eval_small_mcq.jsonl
datasets/custom_eval_small_math_qa.jsonl
datasets/custom_eval_small_math_qa.jsonl.meta.json  (自动生成)

# 方案 B 生成:
datasets/custom_eval_medium_mcq.jsonl
datasets/custom_eval_medium_math_qa.jsonl
datasets/custom_eval_medium_math_qa.jsonl.meta.json (自动生成)
```

### 步骤2: 运行评测

**注意**: MATH-QA 的 `.meta.json` 文件已由脚本自动生成，无需手动创建。

```bash
# 1. MCQ (不需要meta.json)
ais_bench \
  --models vllm_api_general_chat \
  --custom-dataset-path datasets/custom_eval_small_mcq.jsonl \
  --mode all \
  --work-dir outputs/custom_eval_small_mcq

# 2. MATH-QA (自动使用生成的meta.json)
ais_bench \
  --models vllm_api_general_chat \
  --custom-dataset-path datasets/custom_eval_small_math_qa.jsonl \
  --custom-dataset-meta-path datasets/custom_eval_small_math_qa.jsonl.meta.json \
  --mode all \
  --work-dir outputs/custom_eval_small_math_qa
```

## 支持的数据集

| 数据集 | 类型 | 说明 |
|--------|------|------|
| CEval | MCQ | 中文知识评测 (A/B/C/D选项) |
| MMLU | MCQ | 大规模多任务语言理解 (A/B/C/D选项) |
| GPQA | MCQ | 研究生级科学问题 (A/B/C/D选项) |
| AIME2024 | MATH-QA | 美国数学竞赛 (需要MATHEvaluator + pred_postprocessor) |
| MATH500 | MATH-QA | 数学问题求解 (需要MATHEvaluator + pred_postprocessor) |

⚠️ **LiveCodeBench 不支持自定义采样**：代码评测需要完整的测试用例和特殊字段，请使用官方完整数据集。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--aime-count` | AIME2024采样数量 | 20 |
| `--math-count` | MATH500采样数量 | 20 |
| `--ceval-count` | CEval采样数量 | 20 |
| `--mmlu-count` | MMLU采样数量 | 20 |
| `--gpqa-count` | GPQA采样数量 | 0 |
| `--output` | 输出文件路径前缀 | `datasets/custom_sampled_eval` |
| `--seed` | 随机种子 | None |
| `--no-shuffle` | 不打乱数据顺序 | False |

## 常见问题

### 为什么要分成2个文件？

不同数据集需要不同的evaluator：
- **MCQ**: 自动使用 `OptionSimAccEvaluator`（选项匹配）
- **MATH-QA**: 需要 `MATHEvaluator`（提取并标准化数学答案）+ `math_postprocessor_v2`（从推理过程中提取 `\boxed{}` 答案）

如果混在一起，无法为不同题目指定不同evaluator。

### 为什么 MATH-QA 准确率很低（<10%）？

**原因**: 缺少 `pred_postprocessor` 配置。

模型输出包含完整推理过程：
```
Let's solve step by step... The final answer is \boxed{1024}.
```

如果没有 `pred_postprocessor`，整个输出会被送给 evaluator，无法匹配 gold answer "1024"。

**解决**: 使用 meta.json 模板，它包含：
```json
{
  "pred_postprocessor": "ais_bench.benchmark.datasets.math.math_postprocess_v2"
}
```

这个 postprocessor 会提取 `\boxed{1024}` → "1024"，然后送给 evaluator 匹配。

### 路径说明

- **相对路径**: 相对于项目根目录（ais_bench 安装位置）
- **绝对路径**: 也支持，直接使用完整路径
