from dotenv import load_dotenv
import os
import time
import argparse
import multiprocessing as mp

import yaml

from clients.llm_client import LLMClient
from models.script import Script
from dtos.llm_request import ScriptTaxonomyLLMRequest
from dtos.llm_response import ScriptTaxonomyLLMResponse


def taxonomy_job(job, endpoint, api_key, model, system_prompt):

    llm_client = LLMClient(endpoint=endpoint, api_key=api_key, model=model)
    if not job or not job.body:
        print(f"The script cannot be empty or none")
        return None
    try:
        # Execute the request
        request = ScriptTaxonomyLLMRequest(script=job, system_message=system_prompt)
        # Time the response
        t0_req = time.time()
        response = llm_client.query_model(request=request, response_type=ScriptTaxonomyLLMResponse)
        t1_req = time.time()

        # Insert here time reporting operations, e.g.:
        # print(t1_req - t0_req)

        return response.content
    except Exception as e:
        print(f"Call failed with error: {e}")
        return None


if __name__ == "__main__":
    # Load dotenv before reading env vars in argparse defaults
    load_dotenv(".env")

    parser = argparse.ArgumentParser(description="Extract taxonomy characterizations from scripts using LLM")

    # Input: either a single script or a folder of scripts (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--script", type=str,
                             help="Path to a single script to analyze")
    input_group.add_argument("--scripts_folder", type=str,
                             help="Path to a folder whose files are all read as scripts")

    # System prompt path
    parser.add_argument("--system_prompt", default="prompts/label_taxonomy_prompt.txt",
                        help="Path to system prompt file (default: prompts/label_taxonomy_prompt.txt)")

    # Output folder, one yaml per script
    parser.add_argument("--label_folder", default="results/taxonomy_labels",
                        help="Output folder for yaml labels (default: results/taxonomy_labels)")

    # LLM config - keep as env vars by default, but allow override
    parser.add_argument("--endpoint", default=os.getenv("ENDPOINT"),
                        help="LLM endpoint URL (default: LOADS FROM ENDPOINT env var)")
    parser.add_argument("--api_key", default=os.getenv("API_KEY"),
                        help="API key for LLM (default: LOADS FROM API_KEY env var)")
    parser.add_argument("--model", default=os.getenv("MODEL"),
                        help="Model name (default: LOADS FROM MODEL env var)")

    args = parser.parse_args()

    # Load scripts
    if args.script:
        raw_scripts = [open(args.script).read()]
    else:
        folder = args.scripts_folder
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Scripts folder not found: {folder}")
        raw_scripts = [open(os.path.join(folder, name)).read()
                       for name in sorted(os.listdir(folder))
                       if os.path.isfile(os.path.join(folder, name))]
    # Parse scripts
    scripts = [Script(jid=i, body=script, job_directives_keyword="SBATCH") for i, script in enumerate(raw_scripts)]

    # System prompt path
    system_prompt_path = args.system_prompt
    if not os.path.exists(system_prompt_path):
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
    system_prompt = open(system_prompt_path).read()

    # Create the output folder if missing
    label_folder = args.label_folder
    os.makedirs(label_folder, exist_ok=True)

    # Initialize the llm client config
    endpoint = args.endpoint
    api_key = args.api_key
    model = args.model

    # Perform calls in parallel
    pool_args = [(s, endpoint, api_key, model, system_prompt) for s in scripts]
    with mp.Pool(os.cpu_count()) as p:
        res_sub = p.starmap_async(taxonomy_job, pool_args).get()

    # Counter for the failures
    fails = 0

    # Save characterizations, one yaml file per script id
    for s, r in zip(scripts, res_sub):
        if r:
            with open(os.path.join(label_folder, f"{s.jid}.yaml"), "w") as f:
                yaml.safe_dump(r.model_dump(), f, sort_keys=False)
        else:
            fails += 1

    print(f"Total failures: {fails} over {len(scripts)} jobs.")
