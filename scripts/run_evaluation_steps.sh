#!/bin/bash 

# Load environment here

MODEL="deepseek-ai/DeepSeek-V4-Flash"

# Use only the part after the last "/" in MODEL, if any
MODEL_SHORT="${MODEL##*/}"

# Execution
python3 steps_extraction.py \
    --system_prompt prompts/steps_extraction_prompt.txt \
    --label_folder evaluation/results/steps_extraction/"${MODEL_SHORT}" \
    --model "${MODEL}" \
    --scripts evaluation/evaluation_scripts \
