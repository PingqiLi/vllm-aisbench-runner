#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
AISBENCH_ROOT=""
if [ $# -ge 1 ]; then
    AISBENCH_ROOT="$1"
fi

# If no argument provided, prompt user
if [ -z "$AISBENCH_ROOT" ]; then
    echo "Usage: $0 <aisbench_root_path>"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/benchmark"
    echo "  $0 ~/benchmark"
    echo ""
    echo "This will create datasets in: <aisbench_root_path>/ais_bench/datasets/"
    exit 1
fi

# Validate AISBench root directory
if [ ! -d "$AISBENCH_ROOT" ]; then
    echo -e "${RED}Error: AISBench root directory does not exist: $AISBENCH_ROOT${NC}"
    exit 1
fi

# Check if ais_bench exists
if [ ! -d "$AISBENCH_ROOT/ais_bench" ]; then
    echo -e "${YELLOW}Warning: ais_bench directory not found in $AISBENCH_ROOT${NC}"
    echo -e "${YELLOW}Make sure this is the correct AISBench repository root${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

DATASETS_DIR="$AISBENCH_ROOT/ais_bench/datasets"
OPENCOMPASS_OSS="http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data"

echo "======================================"
echo "AISBench Dataset Preparation Script"
echo "======================================"
echo ""
echo "AISBench root: $AISBENCH_ROOT"
echo "Datasets dir:  $DATASETS_DIR"
echo ""

# Create base directory
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        return 1
    fi
    return 0
}

# Check required commands
echo "Checking dependencies..."
check_command wget || exit 1
check_command unzip || exit 1
check_command git || exit 1

# Check for huggingface-cli (optional, for datasets 6-7)
if command -v huggingface-cli &> /dev/null; then
    echo -e "${GREEN}huggingface-cli found${NC}"
    HF_CLI_AVAILABLE=true
else
    echo -e "${YELLOW}huggingface-cli not found (optional, needed for LiveCodeBench and LongBenchV2)${NC}"
    echo -e "${YELLOW}To install: pip install huggingface_hub${NC}"
    HF_CLI_AVAILABLE=false
fi

echo ""
echo "======================================"
echo "Dataset 1/7: CEVAL"
echo "======================================"
if [ -d "ceval" ]; then
    echo -e "${YELLOW}CEVAL already exists, skipping...${NC}"
else
    echo "Downloading CEVAL from ModelScope..."
    wget --no-check-certificate https://modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip
    unzip -q ceval-exam.zip
    rm ceval-exam.zip

    # CEVAL needs structure: ceval/formal_ceval/dev/, val/, test/
    # The zip extracts to dev/, val/, test/ folders directly
    mkdir -p ceval/formal_ceval
    mv dev val test ceval/formal_ceval/ 2>/dev/null || true

    echo -e "${GREEN}CEVAL downloaded successfully${NC}"
fi

echo ""
echo "======================================"
echo "Dataset 2/7: MMLU"
echo "======================================"
if [ -d "mmlu" ]; then
    echo -e "${YELLOW}MMLU already exists, skipping...${NC}"
else
    echo "Downloading MMLU..."
    wget --no-check-certificate ${OPENCOMPASS_OSS}/mmlu.zip
    unzip -q mmlu.zip
    rm mmlu.zip
    # mmlu.zip extracts to mmlu/ directory - correct structure
    echo -e "${GREEN}MMLU downloaded successfully${NC}"
fi

echo ""
echo "======================================"
echo "Dataset 3/7: AIME2024"
echo "======================================"
if [ -d "aime" ]; then
    echo -e "${YELLOW}AIME2024 already exists, skipping...${NC}"
else
    echo "Downloading AIME2024..."
    wget --no-check-certificate ${OPENCOMPASS_OSS}/aime.zip
    unzip -q aime.zip
    rm aime.zip
    # aime.zip extracts to aime.jsonl file - need to create directory
    mkdir -p aime
    mv aime.jsonl aime/
    echo -e "${GREEN}AIME2024 downloaded successfully${NC}"
fi

echo ""
echo "======================================"
echo "Dataset 4/7: GPQA"
echo "======================================"
if [ -d "gpqa" ]; then
    echo -e "${YELLOW}GPQA already exists, skipping...${NC}"
else
    echo "Downloading GPQA..."
    wget --no-check-certificate ${OPENCOMPASS_OSS}/gpqa.zip
    unzip -q gpqa.zip
    rm gpqa.zip
    # gpqa.zip extracts to gpqa/ directory - correct structure
    echo -e "${GREEN}GPQA downloaded successfully${NC}"
fi

echo ""
echo "======================================"
echo "Dataset 5/7: MATH500"
echo "======================================"
if [ -d "math" ]; then
    echo -e "${YELLOW}MATH500 already exists, skipping...${NC}"
else
    echo "Downloading MATH500..."
    wget --no-check-certificate ${OPENCOMPASS_OSS}/math.zip
    unzip -q math.zip
    rm math.zip
    # math.zip extracts to math/ directory - correct structure
    echo -e "${GREEN}MATH500 downloaded successfully${NC}"

    # Install extra requirements for MATH500
    echo "Installing extra dependencies for MATH500..."
    EXTRA_REQ="$AISBENCH_ROOT/requirements/extra.txt"
    if [ -f "$EXTRA_REQ" ]; then
        pip3 install -r "$EXTRA_REQ"
        echo -e "${GREEN}Extra dependencies installed${NC}"
    else
        echo -e "${YELLOW}Warning: $EXTRA_REQ not found, skipping dependency installation${NC}"
    fi
fi

echo ""
echo "======================================"
echo "Dataset 6/7: LiveCodeBench"
echo "======================================"
if [ -d "code_generation_lite" ]; then
    echo -e "${YELLOW}LiveCodeBench already exists, skipping...${NC}"
else
    if [ "$HF_CLI_AVAILABLE" = true ]; then
        echo "Downloading LiveCodeBench using huggingface-cli..."
        huggingface-cli download \
            --repo-type dataset \
            --local-dir code_generation_lite \
            livecodebench/code_generation_lite
        echo -e "${GREEN}LiveCodeBench downloaded successfully${NC}"
    else
        echo -e "${YELLOW}Skipping LiveCodeBench (huggingface-cli not available)${NC}"
        echo -e "${YELLOW}To download manually:${NC}"
        echo -e "${YELLOW}  pip install huggingface_hub${NC}"
        echo -e "${YELLOW}  cd $DATASETS_DIR${NC}"
        echo -e "${YELLOW}  huggingface-cli download --repo-type dataset --local-dir code_generation_lite livecodebench/code_generation_lite${NC}"
    fi
fi

echo ""
echo "======================================"
echo "Dataset 7/7: LongBenchV2"
echo "======================================"
if [ -d "LongBench-v2" ]; then
    echo -e "${YELLOW}LongBenchV2 already exists, skipping...${NC}"
else
    if [ "$HF_CLI_AVAILABLE" = true ]; then
        echo "Downloading LongBenchV2 using huggingface-cli..."
        huggingface-cli download \
            --repo-type dataset \
            --local-dir LongBench-v2 \
            zai-org/LongBench-v2
        echo -e "${GREEN}LongBenchV2 downloaded successfully${NC}"
    else
        echo -e "${YELLOW}Skipping LongBenchV2 (huggingface-cli not available)${NC}"
        echo -e "${YELLOW}To download manually:${NC}"
        echo -e "${YELLOW}  pip install huggingface_hub${NC}"
        echo -e "${YELLOW}  cd $DATASETS_DIR${NC}"
        echo -e "${YELLOW}  huggingface-cli download --repo-type dataset --local-dir LongBench-v2 zai-org/LongBench-v2${NC}"
    fi
fi

echo ""
echo "======================================"
echo -e "${GREEN}Dataset preparation completed!${NC}"
echo "======================================"
echo ""
echo "Dataset locations:"
echo "  1. CEVAL:         $DATASETS_DIR/ceval/formal_ceval/{dev,val,test}/"
echo "  2. MMLU:          $DATASETS_DIR/mmlu/{dev,val,test}/"
echo "  3. AIME2024:      $DATASETS_DIR/aime/aime.jsonl"
echo "  4. GPQA:          $DATASETS_DIR/gpqa/*.csv"
echo "  5. MATH500:       $DATASETS_DIR/math/math.json"
if [ "$HF_CLI_AVAILABLE" = true ]; then
    echo "  6. LiveCodeBench: $DATASETS_DIR/code_generation_lite/"
    echo "  7. LongBenchV2:   $DATASETS_DIR/LongBench-v2/"
else
    echo "  6. LiveCodeBench: (skipped - install huggingface-cli to enable)"
    echo "  7. LongBenchV2:   (skipped - install huggingface-cli to enable)"
fi
echo ""
echo "Next steps:"
echo "  1. Run benchmarks: python run.py --config-file configs/benchmarks/qwen3-30b-acc.yaml"
echo ""
