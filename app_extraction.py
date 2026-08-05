from dotenv import load_dotenv
import os
import time
import argparse
import multiprocessing as mp

from clients.llm_client import LLMClient
from models.script import Script
from dtos.llm_request import ScriptCharacterizerLLMRequest
from dtos.llm_response import ScriptCharacterizerLLMResponse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract app labels from scripts using LLM")

    # System prompt path
    parser.add_argument("--system_prompt", default="prompts/system_prompt.txt",
                        help="Path to system prompt file (default: prompts/system_prompt.txt)")

    # Label output file
    parser.add_argument("--label_file", default="results/labels.csv",
                        help="Output CSV file for labels (default: results/labels.csv)")

    # LLM config - keep as env vars by default, but allow override
    parser.add_argument("--endpoint", default=None,
                        help="LLM endpoint URL (default: LOADS FROM ENDPOINT env var)")
    parser.add_argument("--api_key", default=None,
                        help="API key for LLM (default: LOADS FROM API_KEY env var)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: LOADS FROM MODEL env var)")

    args = parser.parse_args()

    # Load dotenv
    load_dotenv(".env")

    # Load scripts 
    raw_scripts = [] # Replace with function to load a list of scripts
    # Parse scripts
    scripts = [Script(jid=i, body=script, job_directives_keyword="SBATCH") for i, script in enumerate(raw_scripts)]

    # System prompt path
    system_prompt_path = args.system_prompt
    if not os.path.exists(system_prompt_path):
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
    system_prompt = open(system_prompt_path).read()

    # Define output file
    label_file = args.label_file

    # Initialize the llm client
    endpoint = args.endpoint if args.endpoint else os.getenv("ENDPOINT")
    api_key = args.api_key if args.api_key else os.getenv("API_KEY")
    model = args.model if args.model else os.getenv("MODEL")
                
    def characterize_job(script:str):

        llm_client = LLMClient(endpoint=endpoint, api_key=api_key, model=model)
        if not(script):
            print(f"The script cannot be empty or none")
            return None
        try:
            # Execute the request
            request = ScriptCharacterizerLLMRequest(script = script, system_message=system_prompt)
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
        
    # Perform calls in parallel
    with mp.Pool(os.cpu_count()) as p:
        res_sub = p.map_async(characterize_job, scripts).get()
    
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
        f.write("job_id,app_label")
        for i, e in enumerate(labels):
            f.write(f"{i},{e}\n")
    
    
                    
            
            
            
            
    
    
    
    
    
    
    