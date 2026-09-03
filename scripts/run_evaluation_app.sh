#!/bin/bash

# Example usage of app_extraction.py on a single script

# Load environment here

# Paths
SCRIPTS_FOLDER=evaluation/evaluation_scripts
SYSTEM_PROMPT=prompts/app_classification_prompt.txt
MODEL="deepseek-ai/DeepSeek-V4-Flash"

# Use only the part after the last "/" in MODEL, if any
MODEL_SHORT="${MODEL##*/}"

# Execution
python3 app_extraction.py \
    --scripts "${SCRIPTS_FOLDER}" \
    --model "${MODEL} \
    --system_prompt "$SYSTEM_PROMPT" \
    --label_file evaluation/results/application_classification/"${MODEL_SHORT}"
