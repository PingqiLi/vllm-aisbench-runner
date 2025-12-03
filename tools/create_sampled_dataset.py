#!/usr/bin/env python3
"""
Create custom sampled datasets from multiple existing datasets.

This script samples data from aime2024, math500, ceval, mmlu, gpqa, and livecodebench
to create evaluation datasets for quick testing and validation.

Since different QA datasets require different evaluators, this tool separates samples
into THREE files based on evaluation requirements:
  - *_mcq.jsonl:      CEval, MMLU, GPQA (Multiple Choice, use AccEvaluator)
  - *_math_qa.jsonl:  AIME, MATH (Math problems, use MATHEvaluator)
  - *_code_qa.jsonl:  LiveCodeBench (Code generation, use code execution)

Usage:
    # 1. Create the custom datasets (generates 3 files)
    python3 tools/create_sampled_dataset.py \
        --aime-count 30 \
        --math-count 40 \
        --ceval-count 50 \
        --mmlu-count 50 \
        --gpqa-count 30 \
        --livecodebench-count 30 \
        --output datasets/custom_eval \
        --seed 42

    # 2. Run evaluation on MCQ dataset
    ais_bench \
        --models vllm_api_general_chat \
        --custom-dataset-path datasets/custom_eval_mcq.jsonl \
        --mode all \
        --work-dir outputs/custom_eval_mcq

    # 3. Run evaluation on MATH-QA dataset (IMPORTANT: needs MATHEvaluator)
    # This requires creating a .meta.json file - see README for details
    ais_bench \
        --models vllm_api_general_chat \
        --custom-dataset-path datasets/custom_eval_math_qa.jsonl \
        --mode all \
        --work-dir outputs/custom_eval_math_qa
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any


class DatasetSampler:
    """Sample data from multiple datasets and merge into a single dataset."""

    def __init__(self, seed: int = None):
        """Initialize sampler with optional random seed."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # Get ais_bench root directory
        script_dir = Path(__file__).parent
        self.ais_bench_root = script_dir.parent

    def load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        full_path = self.ais_bench_root / path

        if not full_path.exists():
            print(f"Warning: {full_path} not found, skipping")
            return []

        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        return data

    def sample_aime2024(self, count: int) -> List[Dict[str, Any]]:
        """Sample from AIME2024 dataset."""
        print(f"Sampling {count} from AIME2024...")
        data = self.load_jsonl('datasets/aime/aime.jsonl')

        if not data:
            print(f"  ⚠ AIME2024 dataset not found, skipping")
            return []

        samples = random.sample(data, min(count, len(data)))

        # Convert to standardized format - keep ONLY standard fields
        standardized = []
        for item in samples:
            std_item = {
                'question': item.get('origin_prompt', item.get('question', '')),
                'answer': item.get('gold_answer', item.get('answer', '')),
                'source_dataset': 'aime2024'
            }
            standardized.append(std_item)

        print(f"  ✓ Sampled {len(standardized)} / {len(data)} items")
        return standardized

    def sample_math500(self, count: int) -> List[Dict[str, Any]]:
        """Sample from MATH500 dataset."""
        print(f"Sampling {count} from MATH500...")
        # Try multiple possible paths for MATH dataset
        possible_paths = [
            'datasets/math/test_prm800k_500.jsonl',
            'datasets/math/test.jsonl',
            'datasets/math/prm800k_500.jsonl',
        ]

        data = []
        for path in possible_paths:
            data = self.load_jsonl(path)
            if data:
                break

        if not data:
            print(f"  ⚠ MATH500 dataset not found, skipping")
            return []

        samples = random.sample(data, min(count, len(data)))

        # Convert to standardized format - keep ONLY standard fields
        standardized = []
        for item in samples:
            std_item = {
                'question': item.get('problem', item.get('question', '')),
                'answer': item.get('solution', item.get('answer', '')),
                'source_dataset': 'math500'
            }
            standardized.append(std_item)

        print(f"  ✓ Sampled {len(standardized)} / {len(data)} items")
        return standardized

    def sample_ceval(self, count: int) -> List[Dict[str, Any]]:
        """Sample from CEval dataset."""
        print(f"Sampling {count} from CEval...")

        # CEval has multiple subjects, sample from val split
        ceval_path = self.ais_bench_root / 'datasets' / 'ceval' / 'formal_ceval' / 'val'

        if not ceval_path.exists():
            print(f"  ⚠ CEval dataset not found at {ceval_path}, skipping")
            return []

        # Load all CSV files and combine
        all_data = []
        import csv

        for csv_file in ceval_path.glob('*.csv'):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['subject'] = csv_file.stem.replace('_val', '')
                    all_data.append(row)

        if not all_data:
            print(f"  ⚠ No CEval data loaded, skipping")
            return []

        samples = random.sample(all_data, min(count, len(all_data)))

        # Convert to standardized MCQ format - keep ONLY standard fields
        standardized = []
        for item in samples:
            std_item = {
                'question': item.get('question', ''),
                'A': item.get('A', ''),
                'B': item.get('B', ''),
                'C': item.get('C', ''),
                'D': item.get('D', ''),
                'answer': item.get('answer', ''),
                'source_dataset': 'ceval'
            }
            standardized.append(std_item)

        print(f"  ✓ Sampled {len(standardized)} / {len(all_data)} items")
        return standardized

    def sample_mmlu(self, count: int) -> List[Dict[str, Any]]:
        """Sample from MMLU dataset."""
        print(f"Sampling {count} from MMLU...")

        # MMLU has multiple subjects
        mmlu_path = self.ais_bench_root / 'datasets' / 'mmlu'

        if not mmlu_path.exists():
            print(f"  ⚠ MMLU dataset not found at {mmlu_path}, skipping")
            return []

        # Try to load from test or val
        all_data = []
        import csv

        for split in ['test', 'val']:
            split_path = mmlu_path / split
            if not split_path.exists():
                continue

            for csv_file in split_path.glob('*.csv'):
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 6:  # input, A, B, C, D, target
                            item = {
                                'input': row[0],  # MMLU uses 'input' not 'question'
                                'A': row[1],
                                'B': row[2],
                                'C': row[3],
                                'D': row[4],
                                'target': row[5],  # MMLU uses 'target' not 'answer'
                                'subject': csv_file.stem.replace('_test', '').replace('_val', '')
                            }
                            all_data.append(item)

        if not all_data:
            print(f"  ⚠ No MMLU data loaded, skipping")
            return []

        samples = random.sample(all_data, min(count, len(all_data)))

        # Convert to standardized MCQ format - keep ONLY standard fields
        standardized = []
        for item in samples:
            std_item = {
                'question': item.get('input', item.get('question', '')),
                'A': item.get('A', ''),
                'B': item.get('B', ''),
                'C': item.get('C', ''),
                'D': item.get('D', ''),
                'answer': item.get('target', item.get('answer', '')),
                'source_dataset': 'mmlu'
            }
            standardized.append(std_item)

        print(f"  ✓ Sampled {len(standardized)} / {len(all_data)} items")
        return standardized

    def sample_gpqa(self, count: int) -> List[Dict[str, Any]]:
        """Sample from GPQA dataset."""
        print(f"Sampling {count} from GPQA...")

        gpqa_path = self.ais_bench_root / 'datasets' / 'gpqa' / 'gpqa_diamond.csv'

        if not gpqa_path.exists():
            print(f"  ⚠ GPQA dataset not found at {gpqa_path}, skipping")
            return []

        import csv
        all_data = []
        with open(gpqa_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                # Skip header row
                if row_idx == 0 or (len(row) > 7 and row[7] == 'Question'):
                    continue

                if len(row) >= 12:  # Need at least 12 columns for question + 4 options
                    # GPQA format: row[7]=question, row[8-11]=options (first is correct)
                    question = row[7]
                    options = [row[8], row[9], row[10], row[11]]

                    # Create shuffled options like the official loader does
                    shuffle_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']
                    pattern = shuffle_patterns[row_idx % 4]

                    item = {'question': question}
                    ground_truth = options[0]  # First option is always correct

                    # Shuffle options
                    for i in range(4):
                        item['ABCD'[i]] = options[ord(pattern[i]) - ord('A')]

                    # Find which letter corresponds to correct answer
                    for i in range(4):
                        if item['ABCD'[i]] == ground_truth:
                            item['answer'] = 'ABCD'[i]
                            break

                    all_data.append(item)

        if not all_data:
            print(f"  ⚠ No GPQA data loaded, skipping")
            return []

        samples = random.sample(all_data, min(count, len(all_data)))

        # Convert to standardized MCQ format - keep ONLY standard fields
        standardized = []
        for item in samples:
            std_item = {
                'question': item.get('question', ''),
                'A': item.get('A', ''),
                'B': item.get('B', ''),
                'C': item.get('C', ''),
                'D': item.get('D', ''),
                'answer': item.get('answer', ''),
                'source_dataset': 'gpqa'
            }
            standardized.append(std_item)

        print(f"  ✓ Sampled {len(standardized)} / {len(all_data)} items")
        return standardized

    def sample_livecodebench(self, count: int) -> List[Dict[str, Any]]:
        """Sample from LiveCodeBench dataset.

        LiveCodeBench dataset should be downloaded from HuggingFace:
        https://huggingface.co/datasets/livecodebench/code_generation_lite

        Deploy to: ais_bench/datasets/code_generation_lite/
        """
        print(f"Sampling {count} from LiveCodeBench...")

        # LiveCodeBench is in code_generation_lite directory with multiple JSONL files
        lcb_dir = self.ais_bench_root / 'datasets' / 'code_generation_lite'

        if not lcb_dir.exists():
            print(f"  ⚠ LiveCodeBench dataset not found at {lcb_dir}")
            print(f"  ℹ Please download from: https://huggingface.co/datasets/livecodebench/code_generation_lite")
            print(f"  ℹ Deploy to: ais_bench/datasets/code_generation_lite/")
            return []

        # Try to load from test.jsonl (main test file)
        test_file = lcb_dir / 'test.jsonl'
        if not test_file.exists():
            print(f"  ⚠ LiveCodeBench test.jsonl not found in {lcb_dir}")
            return []

        data = self.load_jsonl(str(test_file))

        if not data:
            print(f"  ⚠ No LiveCodeBench data loaded, skipping")
            return []

        samples = random.sample(data, min(count, len(data)))

        # Convert to standardized QA format - keep ONLY standard fields
        standardized = []
        for item in samples:
            std_item = {
                'question': item.get('question_content', item.get('question', '')),
                'answer': '',  # Empty for code generation tasks (evaluated by execution)
                'source_dataset': 'livecodebench'
            }
            standardized.append(std_item)

        print(f"  ✓ Sampled {len(standardized)} / {len(data)} items")
        return standardized

    def create_dataset(self,
                      aime_count: int = 20,
                      math_count: int = 20,
                      ceval_count: int = 20,
                      mmlu_count: int = 20,
                      gpqa_count: int = 0,
                      livecodebench_count: int = 20,
                      shuffle: bool = True) -> tuple:
        """Create combined dataset from multiple sources.

        Returns:
            (mcq_samples, math_qa_samples, code_qa_samples): Three lists
            - mcq_samples: CEval, MMLU, GPQA (choice questions)
            - math_qa_samples: AIME, MATH (math problems, need MATHEvaluator)
            - code_qa_samples: LiveCodeBench (code generation)
        """

        print("\n" + "="*80)
        print("Creating Custom Sampled Dataset")
        print("="*80 + "\n")

        mcq_samples = []       # CEval, MMLU, GPQA
        math_qa_samples = []   # AIME, MATH (use MATHEvaluator)
        code_qa_samples = []   # LiveCodeBench (use code execution)

        # Sample MATH-type QA datasets (AIME + MATH)
        if aime_count > 0:
            math_qa_samples.extend(self.sample_aime2024(aime_count))

        if math_count > 0:
            math_qa_samples.extend(self.sample_math500(math_count))

        # Sample code generation QA dataset
        if livecodebench_count > 0:
            code_qa_samples.extend(self.sample_livecodebench(livecodebench_count))

        # Sample MCQ datasets
        if ceval_count > 0:
            mcq_samples.extend(self.sample_ceval(ceval_count))

        if mmlu_count > 0:
            mcq_samples.extend(self.sample_mmlu(mmlu_count))

        if gpqa_count > 0:
            mcq_samples.extend(self.sample_gpqa(gpqa_count))

        # Shuffle if requested
        if shuffle:
            random.shuffle(mcq_samples)
            random.shuffle(math_qa_samples)
            random.shuffle(code_qa_samples)
            print(f"\n✓ Shuffled MCQ, MATH-QA, and Code-QA datasets separately")

        return mcq_samples, math_qa_samples, code_qa_samples

    def save_datasets(self, mcq_data: List[Dict[str, Any]],
                      math_qa_data: List[Dict[str, Any]],
                      code_qa_data: List[Dict[str, Any]],
                      output_dir: str):
        """Save MCQ and QA datasets to separate JSONL files in a directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine output file paths
        mcq_file = output_path / "mcq.jsonl"
        math_qa_file = output_path / "math_qa.jsonl"
        code_qa_file = output_path / "code_qa.jsonl"

        print(f"\n{'='*80}")
        print(f"Saving datasets to directory: {output_path}")
        print(f"{'='*80}\n")

        # Save MCQ dataset
        if mcq_data:
            with open(mcq_file, 'w', encoding='utf-8') as f:
                for item in mcq_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✓ MCQ dataset:     {len(mcq_data)} samples → {mcq_file}")

            # Print MCQ statistics
            mcq_counts = {}
            for item in mcq_data:
                source = item.get('source_dataset', 'unknown')
                mcq_counts[source] = mcq_counts.get(source, 0) + 1
            print("  MCQ composition:")
            for source, count in sorted(mcq_counts.items()):
                percentage = count / len(mcq_data) * 100
                print(f"    {source:15s}: {count:4d} samples ({percentage:5.1f}%)")
        else:
            print("⚠ No MCQ samples, skipping MCQ file")

        print()

        # Save MATH-QA dataset
        if math_qa_data:
            with open(math_qa_file, 'w', encoding='utf-8') as f:
                for item in math_qa_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✓ MATH-QA dataset: {len(math_qa_data)} samples → {math_qa_file}")

            # Copy meta template
            import shutil
            template_path = Path(__file__).parent / 'math_qa_meta_template.json'
            if template_path.exists():
                meta_dest = output_path / "math_qa.jsonl.meta.json"
                shutil.copy2(template_path, meta_dest)
                print(f"✓ Copied meta file: {meta_dest}")
            else:
                print(f"⚠ Warning: Meta template not found at {template_path}")

            # Print MATH-QA statistics
            math_qa_counts = {}
            for item in math_qa_data:
                source = item.get('source_dataset', 'unknown')
                math_qa_counts[source] = math_qa_counts.get(source, 0) + 1
            print("  MATH-QA composition:")
            for source, count in sorted(math_qa_counts.items()):
                percentage = count / len(math_qa_data) * 100
                print(f"    {source:15s}: {count:4d} samples ({percentage:5.1f}%)")
        else:
            print("⚠ No MATH-QA samples, skipping MATH-QA file")

        print()

        # Save Code-QA dataset
        if code_qa_data:
            with open(code_qa_file, 'w', encoding='utf-8') as f:
                for item in code_qa_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✓ Code-QA dataset: {len(code_qa_data)} samples → {code_qa_file}")

            # Print Code-QA statistics
            code_qa_counts = {}
            for item in code_qa_data:
                source = item.get('source_dataset', 'unknown')
                code_qa_counts[source] = code_qa_counts.get(source, 0) + 1
            print("  Code-QA composition:")
            for source, count in sorted(code_qa_counts.items()):
                percentage = count / len(code_qa_data) * 100
                print(f"    {source:15s}: {count:4d} samples ({percentage:5.1f}%)")
        else:
            print("⚠ No Code-QA samples, skipping Code-QA file")

        print(f"\n{'='*80}")
        total = len(mcq_data) + len(math_qa_data) + len(code_qa_data)
        print(f"✓ Total: {total} samples ({len(mcq_data)} MCQ + {len(math_qa_data)} MATH-QA + {len(code_qa_data)} Code-QA)")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create custom sampled dataset from multiple sources"
    )

    parser.add_argument(
        '--aime-count',
        type=int,
        default=20,
        help='Number of samples from AIME2024 (default: 20)'
    )
    parser.add_argument(
        '--math-count',
        type=int,
        default=20,
        help='Number of samples from MATH500 (default: 20)'
    )
    parser.add_argument(
        '--ceval-count',
        type=int,
        default=20,
        help='Number of samples from CEval (default: 20)'
    )
    parser.add_argument(
        '--mmlu-count',
        type=int,
        default=20,
        help='Number of samples from MMLU (default: 20)'
    )
    parser.add_argument(
        '--livecodebench-count',
        type=int,
        default=0,
        help='LiveCodeBench NOT supported for custom sampling (default: 0, use official dataset)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets/custom_sampled_eval',
        help='Output directory path (default: datasets/custom_sampled_eval). '
             'Will generate mcq.jsonl, math_qa.jsonl, etc. inside this directory.'
    )
    parser.add_argument(
        '--gpqa-count',
        type=int,
        default=0,
        help='Number of samples from GPQA (default: 0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the final dataset'
    )

    args = parser.parse_args()

    # Create sampler
    sampler = DatasetSampler(seed=args.seed)

    # Create datasets (returns MCQ, MATH-QA, Code-QA separately)
    mcq_dataset, math_qa_dataset, code_qa_dataset = sampler.create_dataset(
        aime_count=args.aime_count,
        math_count=args.math_count,
        ceval_count=args.ceval_count,
        mmlu_count=args.mmlu_count,
        gpqa_count=args.gpqa_count,
        livecodebench_count=args.livecodebench_count,
        shuffle=not args.no_shuffle
    )

    # Save datasets
    if mcq_dataset or math_qa_dataset or code_qa_dataset:
        sampler.save_datasets(mcq_dataset, math_qa_dataset, code_qa_dataset, args.output)
        return 0
    else:
        print("\n❌ No data was sampled. Please check dataset paths.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
