from dotenv import load_dotenv
import os 
import time
import multiprocessing as mp

from clients.llm_client import LLMClient
from models.script import Script
from dtos.llm_request import ScriptCharacterizerLLMRequest
from dtos.llm_response import ScriptCharacterizerLLMResponse

if __name__ == "__main__":
    # Load dotenv 
    load_dotenv(".env")

    # Load scripts 
    raw_scripts = [] # Replace with function to load a list of scripts
    # Parse scripts
    scripts = [Script(jid=i, body=script, job_directives_keyword="SBATCH") for i, script in enumerate(raw_scripts)]

    # System prompt path 
    system_prompt = open("prompts/system_prompt.txt").read()
            
    # Define output file
    label_file = f"results/labels.csv"
                
    def characterize_job(script:str):
        # Initialize the llm client
        llm_client = LLMClient(endpoint=os.getenv("ENDPOINT"), api_key=os.getenv("API_KEY"), model = os.getenv("MODEL"))
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
    
    
                    
            
            
            
            
    
    
    
    
    
    
    