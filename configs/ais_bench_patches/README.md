# AISBench Custom Configurations

这个目录包含了benchmark_runner项目中使用的自定义AISBench配置文件。

## 为什么需要这些配置？

某些评测场景需要对AISBench的默认配置进行定制，例如：
- LongBench v2的without CoT版本（官方默认是with CoT）
- 自定义的prompt模板
- 特殊的评测参数设置

## 如何使用

### 方法1：手动复制（推荐）

将配置文件复制到对应的AISBench目录中：

```bash
# 以LongBench v2 without CoT为例
cp configs/ais_bench_patches/longbenchv2/*.py \
   /path/to/ais_bench/benchmark/configs/datasets/longbenchv2/
```

### 方法2：使用脚本（待实现）

未来可能会添加自动注入机制。

## 当前包含的配置

### longbenchv2/

LongBench v2的without CoT配置，与Qwen3官方评测设置对齐。

**文件列表：**
- `longbenchv2_gen_wo_cot.py` - 主配置文件（去掉了"Let's think step by step"）
- `longbenchv2_gen_0_shot_chat_prompt.py` - 入口文件

**对比：**
- **With CoT** (原始): "Let's think step by step. Based on the above..."
- **Without CoT** (本配置): "Based on the above..."

**使用场景：**
- Qwen3官方LongBench v2评测（w/o CoT模式）
- 需要非思考模式的长文本评测

## 维护说明

1. 配置文件与AISBench版本对应，升级AISBench时需要检查兼容性
2. 修改配置后需要重新复制到AISBench目录
3. 建议在配置文件中添加详细注释说明定制原因
