from dotenv import load_dotenv
import gc
import math
import multiprocessing as mp
import os
import time
import sys

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from dtos.llm_request import ScriptCharacterizerLLMRequest
from dtos.llm_response import ScriptCharacterizerLLMResponse
from models.script import Script
from models.script_characterization import ScriptCharacterization


def show_usage():
    print("Usage: python app_extraction.py <job_script_parquet> <chunk_size> <start_chunk_index> <end_chunk_index>")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        show_usage()

    # parse args
    chunk_size = int(sys.argv[2])
    start_chunk_index = int(sys.argv[3])
    end_chunk_index = int(sys.argv[4])

    # Load dotenv
    load_dotenv(".env")

    # setup vllm model
    llm = LLM(
        os.environ["MODEL"],
        tensor_parallel_size=torch.cuda.device_count(),
        enable_expert_parallel=True,
        distributed_executor_backend="mp"
    )

    # Load scripts
    tic = time.perf_counter()
    df = pd.read_parquet(
        sys.argv[1],
        columns=['content']
    )
    print(f"Data loading time: {time.perf_counter() - tic:.4f} seconds")

    # System prompt path
    system_prompt = open("prompts/system_prompt.txt").read()

    # calculate number of chunks
    num_chunks = math.ceil(df.shape[0] / chunk_size)

    for chunk_index in range(start_chunk_index, min(end_chunk_index+1, num_chunks)):
        print(f"Chunk {chunk_index:05} of {num_chunks:05}")

        # get the chunk
        raw_scripts = df.iloc[chunk_size*chunk_index:chunk_size*(chunk_index+1)]

        # Parse scripts
        scripts = [
            Script(jid=row.Index, body=row.content, job_directives_keyword="PBS")
            for row in raw_scripts.itertuples(index=True)
        ]

        # Compiles messages for chat
        script_idx = []
        requests = []
        for script in scripts:
            cmd = script.script_commands.strip()
            if cmd:
                script_idx.append(script.jid)
                requests.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the job script to analyze:\n{cmd}"}
                ])

        tic = time.perf_counter()
        guided_decoding_params = GuidedDecodingParams(
            json=ScriptCharacterization.model_json_schema()
        )
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
            max_tokens=1024
            #guided_decoding=guided_decoding_params
        )
        responses = llm.chat(
            requests,
            sampling_params=sampling_params
        )
        print(f"Inference time: {time.perf_counter() - tic:.4f} seconds")

        # Save labels
        labels = []
        for r in responses:
            labels.append('\n'.join([l.text for l in r.outputs]))

        # Define output file
        label_file = f"results_offline/labels-{chunk_index:05}-of-{num_chunks:05}.csv"

        # Save labels to file
        with open(label_file, "w") as f:
            for i, e in zip(script_idx, labels):
                f.write(f"{i},{e}\n")

    # shutdown vllm
    del llm
    gc.collect()
    torch.cuda.empty_cache()
