#!/bin/bash 

# Load environment here

MODEL="deepseek-ai/DeepSeek-V4-Flash"

# Use only the part after the last "/" in MODEL, if any
MODEL_SHORT="${MODEL##*/}"

# Execution
python3 taxonomy_extraction.py \
    --system_prompt prompts/taxonomy_prompt.txt \
    --label_folder evaluation/results/label_taxonomy/"${MODEL_SHORT}" \
    --model "${MODEL}" \
    --scripts evaluation/evaluation_scripts \
