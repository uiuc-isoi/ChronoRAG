# ChronoRAG
Detecting events from emails using Time-based Retrieval for RAG

## Steps to run Qdrant Vector Store 
- We will be using [Qdrant](https://qdrant.tech/) Vector Store to create a vector store with the email dataset. The dataset needs to be cleaned to include only the information required for the LLM to extract and column values should be in string format. 
- To run Qdrant in local as a docker container, do
```
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```
Point your browser to `localhost:6333` and you will see a UI dashbaord for Qdrant.
- The vectore store is created and stored in memory of the local machine.
- To upload files (csv, pdf, word are currently supported), change the env `DATA_DIR_PATH` where all files to be uploaded resides, change `llm.py` [main script](https://github.com/uiuc-isoi/COIN/blob/main/llm.py) to run `qdrant_upload` function. Run `llm.py` using `python llm.py`.

## Steps to run llm.py
This file is the main script for this repository. Other files can be called from `llm.py`. `llm.py` has functions to create a simple hugging Face pipeline and to create a RAG agent.
- If creating a RAG agent, in the `main` object call `rag`.
- For executing a HF pipeline with HF models, in the `main` object call `hf_pipeline`.

### Instructions for HF pipeline
- The HF pipeline uses Llama2 model. To change this, modify the `model_id`.
- The user input query is taken from the "QUERY" environment variable.
### Instructions for RAG agent
- The RAG agent uses a vectore store and takes user input queries from .txt files in the QUERY_DIR_PATH.
- The vector store should already be defined and the collection should exists.
- Define the env variables in `.env` file
- Add multiple user queries in multiple `.txt` files under the QUERY_DIR_PATH

### Steps to run llm.py on Delta
- Ssh to delta `ssh user@login.delta.ncsa.illinois.edu`
- Find out our account tag (it's four letters) with command `accounts`. Note this down in the slurm file (llm.slurm).
- Copy files `llm.py llm.slurm .example-env requirements.txt` to `/scratch/account_tag/username/`.
- Load gcc and anaconda modules `module load gcc anaconda3_gpu`
- Activate conda shell `source /sw/external/python/anaconda3/etc/profile.d/conda.sh`
- Create a conda virtual environment using the requirements file. `conda create --name coin --file requirements.txt`
- Activate conda environment `conda activate coin`
- Rename `.example-env` to `.env`
- You can change the question given to the LLM in the .env file or in the llm.py `example_question` variable.
- The prompt template can be changed in llm.py `template` variable.
- Run slurm file as `sbatch llm.slurm`
- Logs can be found at `llm.log` and slurm outputs at `slurm-jobid.out`
  
