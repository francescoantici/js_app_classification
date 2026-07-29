# Job Script Appllication Classification with AI

This repo contains the code to execute application classification of job scripts via LLMs. The LLM inference is handled through API calls to an OpenAI-compatible inference backend.

**Repository Structure**
- **app_extraction.py:** Example to run the extraction/classification workflow. To be filled with the needed data.
- **requirements.txt:** Python dependencies required to run the project.
- **clients/**: Contains client wrappers used to call external services.
	- **llm_client.py:** Lightweight wrapper around the LLM OpenAI API used for sending requests and receiving responses.
- **dtos/**: Data Transfer Objects — small classes that represent request/response payloads.
	- **llm_request.py:** DTO representing the structure of a request sent to the LLM client.
	- **llm_response.py:** DTO for responses returned from the LLM client.
- **models/**: Core domain logic and model adapters.
	- **script.py:** Model for a generic job script.
	- **script_characterization.py:** Data structure to parse the LLM output. It is used as the structure output for the LLM calls.
- **prompts/**: Text prompt templates and system prompt for the LLM.
	- **system_prompt.txt:** System-level instructions used when querying the LLM.
- **evaluation/**: Contains the evaluation scripts and results for a series of models.

**How to run (local, minimal)**
- **Install dependencies:**

	```
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	```

Add API keys and configuration to `.env` file before running (not checked into repo). To add new prompt templates, create files under `prompts/` and reference them from `clients/llm_client.py` or the model logic. Make sure to modify the `app_extraction.py` file to load the needed raw job script dataset. If needed, the file `script.py` should be configured, or extended by a new class, to describe the job script structure of the target dataset.

- **Run extraction:**

	```
	python app_extraction.py
	```


