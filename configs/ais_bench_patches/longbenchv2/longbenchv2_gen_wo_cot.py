"""
LongBench v2 - Without CoT Configuration
This is a custom configuration for LongBench v2 evaluation without Chain-of-Thought.

To use this configuration:
1. Copy this file to: ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_wo_cot.py
2. Copy the entry file to: ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_0_shot_chat_prompt.py

Or use the official longbenchv2_gen configuration if you prefer the CoT version.
"""

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import LongBenchv2Dataset, LongBenchv2Evaluator
from ais_bench.benchmark.utils.text_postprocessors import first_option_postprocess

LongBenchv2_reader_cfg = dict(
    input_columns=['context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'difficulty', 'length'],
    output_column='answer',
)

LongBenchv2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    # Official LongBench v2 w/o CoT prompt (from prompts/0shot.txt)
                    prompt='Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\nWhat is the correct answer to this question: {question}\nChoices:\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\n\nFormat your response as follows: "The correct answer is (insert answer here)".',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBenchv2_eval_cfg = dict(
    evaluator=dict(type=LongBenchv2Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD')
)

LongBenchv2_datasets = [
    dict(
        type=LongBenchv2Dataset,
        abbr='LongBenchv2',
        path='ais_bench/datasets/LongBench-v2/data.json',
        reader_cfg=LongBenchv2_reader_cfg,
        infer_cfg=LongBenchv2_infer_cfg,
        eval_cfg=LongBenchv2_eval_cfg,
    )
]
