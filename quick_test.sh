#!/bin/bash
# Quick verification script using AISBench debug mode
# Tests configuration correctness without running full benchmarks

set -e

# Use quick-test.yaml by default, or accept custom config
CONFIG_FILE="${1:-configs/benchmarks/quick-test.yaml}"
NUM_PROMPTS="${2:-5}"

echo "========================================="
echo "Quick Configuration Verification"
echo "========================================="
echo "Config: $CONFIG_FILE"
echo "Prompts: $NUM_PROMPTS (debug mode)"
echo "========================================="
echo ""

# Run with debug mode and limited prompts
python run.py \
    --config-file "$CONFIG_FILE" \
    --num-prompts "$NUM_PROMPTS"

echo ""
echo "========================================="
echo "✓ Verification completed successfully!"
echo "========================================="
echo ""
echo "To run full benchmark:"
echo "  python run.py --config-file configs/benchmarks/qwen3-30b-acc.yaml"
echo ""
