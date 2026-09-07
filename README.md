# Job Script Appllication Classification with AI

This repo contains the code to execute application classification of job scripts via LLMs. The LLM inference is handled through API calls to an OpenAI-compatible inference backend.

**Repository Structure**
- **app_extraction.py:** Example to run the extraction/classification workflow. To be filled with the needed data.
- **app_clustering.py:** DBSCAN clustering script for grouping similar app labels.
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
	- **app_classification_prompt.txt:** System-level instructions used when querying the LLM.
- **evaluation/**: Contains the evaluation scripts and results for a series of models.

## app_extraction.py - LLM-based Application Extraction

Extracts application labels from job scripts using an LLM.

**Required setup:**
- Add API keys and configuration to `.env` file:
  ```
  ENDPOINT=https://your-endpoint.com/v1
  API_KEY=your-api-key
  MODEL=your-model-name
  ```

**Usage:**
```bash
python app_extraction.py [OPTIONS]
```

**Options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--system_prompt` | `prompts/app_classification_prompt.txt` | Path to system prompt file |
| `--label_file` | `results/labels.csv` | Output CSV file for labels |
| `--endpoint` | env variable | LLM endpoint URL (overrides `.env`) |
| `--api_key` | env variable | API key for LLM (overrides `.env`) |
| `--model` | env variable | Model name (overrides `.env`) |

**Example:**
```bash
python app_extraction.py --system_prompt prompts/custom_prompt.txt --label_file results/output.csv
```

## app_clustering.py - DBSCAN Clustering

Clusters similar app labels using embedding vectors and DBSCAN.

**Output from `app_extraction.py` (labels.csv) is required as input.**

**Usage:**
```bash
python app_clustering.py [OPTIONS]
```

**Options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_csv` | `results/labels.csv` | Input CSV file with app labels |
| `--output_csv` | `results/labels_clustered.csv` | Output clustered CSV file |
| `--label_column` | `app_label` | Column name containing labels to cluster |
| `--min_cluster_size` | `15` | Minimum number of samples in a cluster (DBSCAN min_samples) |
| `--eps` | `0.01` | Maximum distance between samples in the same cluster (DBSCAN eps) |
| `--metric` | `cosine` | Distance metric for DBSCAN |
| `--embedding_model` | `google/embeddinggemma-300m` | Sentence-transformers model for embeddings |
| `--truncate_dim` | `256` | Dimension to truncate embeddings |
| `--batch_size` | `64` | Batch size for embedding inference |
| `--device` | auto-detect | Device for embedding (e.g., cpu, cuda) |

**Example:**
```bash
python app_clustering.py --eps 0.05 --min_cluster_size 10
```

## Full Workflow

1. **Extract** application labels from scripts:
   ```bash
   python app_extraction.py
   ```

2. **Cluster** the extracted labels:
   ```bash
   python app_clustering.py
   ```

3. Results are saved to `results/labels.csv` (extraction) and `results/labels_clustered.csv` (clustering).

