from dotenv import load_dotenv
import os
import time
import argparse
import multiprocessing as mp

from clients.llm_client import LLMClient
from models.script import Script
from dtos.llm_request import ScriptCharacterizerLLMRequest
from dtos.llm_response import ScriptCharacterizerLLMResponse


def characterize_job(job, endpoint, api_key, model, system_prompt):

    llm_client = LLMClient(endpoint=endpoint, api_key=api_key, model=model)
    if not job or not job.body:
        print(f"The script cannot be empty or none")
        return None
    try:
        # Execute the request
        request = ScriptCharacterizerLLMRequest(script=job, system_message=system_prompt)
        # Time the response
        t0_req = time.time()
        response = llm_client.query_model(request=request, response_type=ScriptCharacterizerLLMResponse)
        t1_req = time.time()

        # Insert here time reporting operations, e.g.:
        # print(t1_req - t0_req)

        return response.content.application.replace(',', '')
    except Exception as e:
        print(f"Call failed with error: {e}")
        return None


if __name__ == "__main__":
    # Load dotenv before reading env vars in argparse defaults
    load_dotenv(".env")

    parser = argparse.ArgumentParser(description="Extract app labels from scripts using LLM")

    # Input: either a single script or a folder of scripts (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--script", type=str,
                             help="Path to a single script to analyze")
    input_group.add_argument("--scripts_folder", type=str,
                             help="Path to a folder whose files are all read as scripts")

    # System prompt path
    parser.add_argument("--system_prompt", default="prompts/app_classification_prompt.txt",
                        help="Path to system prompt file (default: prompts/app_classification_prompt.txt)")

    # Label output file
    parser.add_argument("--label_file", default="results/labels.csv",
                        help="Output CSV file for labels (default: results/labels.csv)")

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

    # Define output file
    label_file = args.label_file
    label_dir = os.path.dirname(label_file)
    if label_dir:
        os.makedirs(label_dir, exist_ok=True)

    # Initialize the llm client config
    endpoint = args.endpoint
    api_key = args.api_key
    model = args.model

    # Perform calls in parallel
    pool_args = [(s, endpoint, api_key, model, system_prompt) for s in scripts]
    with mp.Pool(os.cpu_count()) as p:
        res_sub = p.starmap_async(characterize_job, pool_args).get()

    # Counter for the failures
    fails = 0

    # Save labels
    labels = []

    # Parse results
    for r in res_sub:
        if r:
            labels.append(r)
        else:
            fails += 1

    print(f"Total failures: {fails} over {len(scripts)} jobs.")

    # Save labels to file
    with open(label_file, "w") as f:
        f.write("job_id,app_label\n")
        for i, e in enumerate(labels):
            f.write(f"{i},{e}\n")
