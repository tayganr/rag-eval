# rag-eval
Python-based evaluation suite for assessing the quality of various Retrieval Augmented Generation (RAG) configurations.

# Pre-requisites
- Azure subscription
- Azure resources
  - Azure OpenAI Service
  - Azure AI Search Service
- Model deployments in the Azure OpenAI Service:
  - Embedding model (e.g. `text-embedding-ada-002`)
  - Generative model (e.g. `gpt-4o`)

# Setup
1. Clone this repo locally
2. create a virtual environment, for example with venv:
   - Linux: `python3 -m venv .venv`
3. activate it
   - Linux: `source .venv/bin/activate`
4. install python requirements
   - `pip install -r requirements.txt`
5. copy and adapt your .env file
   - `cp config/.sample.env config/.env`
6. if you have different model deployment names than those suggested above, review and update `config/evals/eval_1.json` with your model deployment names before a first run
  
# Usage
```
python main.py --config ./config/evals/eval_1.json
```

> Note: The `--config` argument should point to a valid configuration file. The sample eval_1.json file is configured to use a small setup test source document folder with only one text document - update the source_document_folder value to run a full evaluation