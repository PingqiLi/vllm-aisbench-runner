"""
LongBench v2 - Without CoT Entry Point
This file should be copied to: ais_bench/benchmark/configs/datasets/longbenchv2/
"""

from mmengine.config import read_base

with read_base():
    from .longbenchv2_gen_wo_cot import LongBenchv2_datasets
