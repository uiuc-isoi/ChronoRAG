# Main file to run

import torch
import pandas as pd
from datetime import datetime
from typing import List
from ast import literal_eval
import sys
import logging
import os
import json
from dotenv import load_dotenv

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate

from rag import rag_results, rag_search_results, rag_weekly_sequential
from utils.utils import find_unique_id, write_rag_results
from vector_db import Qdrantdb

# Load .env file
load_dotenv(dotenv_path=".env", override=True)


# create and configure logger
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%dT%H:%M:%S',
                    format='%(asctime)-15s.%(msecs)03dZ %(levelname)-7s : %(name)s - %(message)s',
                    handlers=[logging.FileHandler("llm.log"), logging.StreamHandler(sys.stdout)])
# create log object with current module name
log = logging.getLogger(__name__)

# set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    log.info("MPS device selected")

elif torch.cuda.is_available():
    device = torch.device("cuda")
    log.info("CUDA device selected")
else:
    device = torch.device("cpu")
    log.info("MPS device not found, and CUDA device not available. Setting device as CPU")

# HF pipeline with Llama2 model
def hf_pipeline():
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id, device=device)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    hf = HuggingFacePipeline(pipeline=pipe)


    template = """You are an AI-assistant for the business operations team at the University of Illinois. Answer the question based on your business acumen.
    Try to answer the question in detail. If the question cannot be answered, answer with "I don't know". 
    Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    chain = prompt | hf

    example_question = """What is prospective monetization?"""
    question = os.getenv('QUERY', example_question)

    log.info(chain.invoke({"question": question}))


def read_user_query() -> List[str]:
    """
    Read user query from the QUERY_DIR_PATH directory
    All text files are read and returned as a list of queries(str)
    Args:
        None
    Returns:
        List[str]: List of queries
    """
    query_dir = os.getenv('QUERY_DIR_PATH', "query")
    queries = []
    if os.path.isdir(query_dir):
        for file_name in os.listdir(query_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(query_dir, file_name)
                with open(file_path, 'r') as file:
                    query = file.read().strip()
                    queries.extend([{"query": query}])
    else:
        log.error("%s is not a valid directory", query_dir)

    return queries


def query_pipeline() -> List[str]:
    """
    Read user query from the QUERY_DIR_PATH directory. All text files are read and stored as list of strings.
    Matches with the TARGET_QUERY_NAME and returns the query that matches
    All text files are read and returned as a list of queries(str)
    Args:
        None
    Returns:
        List[str]: List of queries
    """
    query_dir = os.getenv('QUERY_DIR_PATH', "query")
    queries = []
    ids = []
    names = []
    timestamps = []

    # Read text files in query directory
    if os.path.isdir(query_dir):
        for file_name in os.listdir(query_dir):
            if file_name.endswith('.txt'):
                base_name = os.path.splitext(file_name)[0]
                names.append(base_name)
                file_path = os.path.join(query_dir, file_name)
                with open(file_path, 'r') as file:
                    query = file.read().strip()
                    queries.append({"query": query})
                    timestamps.append(datetime.now())
    else:
        log.error("%s is not a valid directory", query_dir)

    for query in queries:
        ids.append(find_unique_id(query['query']))

    data = {
        'ID': ids,
        'NAME': names,
        'QUERY': [q['query'] for q in queries],
        'TIMESTAMP': timestamps
    }

    # Create the DataFrame
    df_update = pd.DataFrame(data)

    # Write source documents to file
    QUERY_FILE = os.getenv('QUERY_FILE', "query.csv")
    if not os.path.exists(QUERY_FILE):
        os.makedirs(os.path.dirname(QUERY_FILE), exist_ok=True)
    
    try:
        df = pd.read_csv(QUERY_FILE, parse_dates=['TIMESTAMP'])
    except FileNotFoundError:
        df = pd.DataFrame(columns=['ID', 'NAME', 'QUERY', 'TIMESTAMP'])
    
    # Concatenate the new data with the existing data
    df = pd.concat([df, df_update], ignore_index=True)

    # Remove older entries based on the NAME column
    df = df.sort_values(by='TIMESTAMP').drop_duplicates(subset='NAME', keep='last')

    # Write the updated DataFrame to the CSV file
    df.to_csv(QUERY_FILE, index=False)


    if not QUERY_FILE.endswith('.csv'):
        raise ValueError("Sources file must be a CSV file.")
    
    target_query = os.getenv('TARGET_QUERY_NAME', "test")
    names_to_match = target_query.split(',')

    matched_queries = df[df['NAME'].isin(names_to_match)]['QUERY'].tolist()
    
    return matched_queries


def rag():
    """
    Create a RAG agent that has knowledge from data directory and responds to user queries from query directory
    Args:
        None
    Returns:
        None
    """
    # check if qdrantdb collection exists
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', "email-collection")
    qdrantdb = Qdrantdb(collection_name=collection_name)
    collection_exists = qdrantdb.collection_exists()
    if not collection_exists:
        log.error("Qdrantdb collection %s not found", collection_name)
        return
    
    # fetch user queries from query directory. 
    # Use query_pipeline() to read all files from query directory and match to TARGET_QUERY_NAME.
    # Use read_user_query() to read all files from query directory
    queries = read_user_query()   # query_pipeline()
    
    # Qdrant db as a vector store for RAG. Simplest form.
    # vector_store = qdrantdb.vector_store()
    # responses = rag_results(vector_store=vector_store, queries=queries)

    # Use a custom retriever for RAG, enables filtering. A bit more complex
    rag_responses = rag_search_results(queries=queries, collection_name=collection_name)
    # write to results file
    results_file = os.getenv('RESULTS_FILE', "./results.csv")
     # write source documents to file
    source_file = os.getenv('SOURCES_FILE', "./sources.csv")
    filters = os.getenv('FILTERS', []) 
    write_rag_results(rag_responses=rag_responses, filters=filters, results_file=results_file, source_file=source_file)
    
    pass


def rag_chain_collection():
    """
    Create a RAG agent that has knowledge from data directory and responds to user queries from query directory
    RAG is done for two collections - the collections will be chained in the order they are in the list.
    User query is first searched through first collection and the results from that is passed to RAG with the second collection.
    """
    email_collection_name = os.getenv('EMAIL_COLLECTION_NAME', "email-collection")
    doc_collection_name = os.getenv('DOC_COLLECTION_NAME', "doc-collection")
    rag_two_collection(collections=[email_collection_name, doc_collection_name])  # change the collection names as needed
    pass


def rag_two_collection(collections: List[str]):
    """
    Create a RAG agent that has knowledge from data directory and responds to user queries from query directory
    RAG is done for two collections - the collections will be chained in the order they are in the list.
    User query is first searched through first collection and the results from that is passed to RAG with the second collection.
    """
    num_collections = len(collections)
    if num_collections != 2:
        log.error("Exactly two collections are required to be passed")
        return
    first_collection = collections[0]
    second_collection = collections[1]

    user_queries = query_pipeline()

    response_list = []
    source_data_list = []

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # get RAG responses from first collection
    first_responses = rag_search_results(queries=user_queries, collection_name=first_collection)
    first_responses_list = []
    second_queries = []  # modified queries for RAG in doc-collection
    second_query_suffix = "Some initial findings are as follows. \n"
    unique_id = find_unique_id(timestamp)

    for response in first_responses:
        query = response['query']
        result = response['result']
        source_documents = response['source_documents']
        source_data = [{'source': doc.metadata['source'], 'row_page': doc.metadata['row_page'], 'content': doc.page_content} for doc in source_documents]
        source_data_str = " ".join([f"({data['source']}, {data['row_page']}, {data['content']})" for data in source_data])
        for doc in source_documents:
                source_data_list.append({
                    'id': str(unique_id)+'_eml',
                    'source': doc.metadata.get('source', ''),
                    'row_page': doc.metadata.get('row_page', ''),
                    'content': doc.page_content
                })
                first_responses_list.append({
                    'id':str(unique_id)+'_eml',
                    'timestamp': timestamp, 
                    'query': query, 
                    'result': result
                })        # for each user query and first collection RAG response, use it for second RAG pass
        second_queries.append(" ".join([query, second_query_suffix, result]))

    # get RAG responses from doc collection
    second_responses_list = []
    second_responses = rag_search_results(queries=second_queries, collection_name=second_collection)
    for response in second_responses:
        query = response['query']
        result = response['result']
        source_documents = response['source_documents']
        for doc in source_documents:
            source_data_list.append({
                'id': str(unique_id)+'_doc',
                'source': doc.metadata.get('source', ''),
                'row_page': doc.metadata.get('row_page', ''),
                'content': doc.page_content
            })
        second_responses_list.append({
            'id':str(unique_id)+'_doc', 
            'timestamp': timestamp, 
            'query': query, 
            'result': result
        })

    sources_updated_df = pd.DataFrame(source_data_list)

    # write to results file
    source_file = os.getenv('SOURCES_FILE', "sources.csv")
    if not os.path.exists(source_file):
        os.makedirs(os.path.dirname(source_file), exist_ok=True)
        with open(source_file, 'w') as file:
            file.write('id,source,row_page,content\n')
    if not source_file.endswith('.csv'):
        raise ValueError("Sources file must be a CSV file.")
    
    sources_df = pd.read_csv(source_file, usecols=['id', 'source', 'row_page', 'content'])
    
    df = pd.concat([sources_updated_df, sources_df])
    df.to_csv(source_file, index=False)


    responses_df = pd.DataFrame(second_responses_list)

    # write to results file
    results_file = os.getenv('RESULTS_FILE', "results.csv")
    if not os.path.exists(results_file):
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as file:
            file.write('id,timestamp,query,result\n')
    if not results_file.endswith('.csv'):
        raise ValueError("Results file must be a CSV file.")
    
    results_df = pd.read_csv(results_file, usecols=['id', 'timestamp', 'query', 'result'])

    df = pd.concat([responses_df, results_df])
    df.to_csv(results_file, index=False)


def rag_sequential():
    """
    Create a RAG agent that has knowledge from email qdrant collection and responds to user queries.
    RAG is chained on a week to week basis. User provides a start week and end week.
    For each week, the RAG is done on the email collection for that week and the results are passed to the email collection for the next week, till the end week.
    Weekly results are stored in results file
    Args:
        None
    Returns:
        None
    """
    # check if qdrantdb collection exists
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', "email-collection")
    qdrantdb = Qdrantdb(collection_name=collection_name)
    collection_exists = qdrantdb.collection_exists()
    if not collection_exists:
        log.error("Qdrantdb collection %s not found", collection_name)
        return
    
    # fetch user queries from query directory. 
    # Use query_pipeline() to read all files from query directory and match to TARGET_QUERY_NAME.
    # Use read_user_query() to read all files from query directory
    queries = read_user_query()   # query_pipeline()

    # Use a custom retriever for RAG,with weekly filtering with sequential chaining.
    start_week = os.getenv('START_WEEK', "1")
    end_week = os.getenv('END_WEEK', "5")
    filters = os.getenv('FILTERS', [])  
    combined_filters, rag_responses = rag_weekly_sequential(queries=queries, collection_name=collection_name, start_week=start_week, end_week=end_week, filters=filters)
    
    # write to results file
    results_file = os.getenv('RESULTS_FILE', "./results.csv")
     # write source documents to file
    source_file = os.getenv('SOURCES_FILE', "./sources.csv")
    write_rag_results(rag_responses=rag_responses, filters=combined_filters, results_file=results_file, source_file=source_file)
    
    pass


def qdrant_upload():
    """
    Create a qdrant collection and upload data in a given directory
    This is a generic function that accomodates csv, pdf and word files
    All file content are treated as payload. Metadata is source file path and row number / page number
    """
    data_dir = os.getenv('DATA_DIR_PATH', "data")
    qdrantdb = Qdrantdb()
    try:
        collection_exists = qdrantdb.collection_exists()
        log.info("Collection exsists: %s", collection_exists)
        if not collection_exists:
            collection_create_status = qdrantdb.create_collection()
            log.info("Collection create status: %s", collection_create_status)
    except Exception as e:
        log.error("Execption in qdrant : Trying recreating collection")
        collection_create_status = qdrantdb.create_collection()
        log.info("Collection create status: %s", collection_create_status)
    
    upsert_data_status = qdrantdb.upsert_data_generic(data_dir_path=data_dir)
    log.info("Data upload completed: %s", upsert_data_status)


def qdrant_upload_csv_fine_control():
    """
    Create a qdrant collection and upload csv data in given directory. 
    The uploaded csv data is formatted for filtering, with metadata columns of thread_id, year_month, communication_channel, week, period, source, row
    Payload will be the rest of the content in the csv file.
    It is recommended to have the rest of the contents that can be used for RAG (Eg: only the conversation column)
    Metadata columns are used for exact-match filtering
    """
    data_dir = os.getenv('DATA_DIR_PATH', "data")
    qdrantdb = Qdrantdb()
    try:
        collection_exists = qdrantdb.collection_exists()
        log.info("Collection exsists: %s", collection_exists)
        if not collection_exists:
            collection_create_status = qdrantdb.create_collection()
            log.info("Collection create status: %s", collection_create_status)
    except Exception as e:
        log.error("Execption in qdrant : Trying recreating collection")
        collection_create_status = qdrantdb.create_collection()
        log.info("Collection create status: %s", collection_create_status)

    upsert_data_status = qdrantdb.upsert_csvdata_fine_control(data_dir_path=data_dir)
    log.info("CSV Data upload with fine control completed: %s", upsert_data_status)


if __name__ == '__main__':
    #qdrant_upload_csv_fine_control() # to create qdrant collection and upload csv data with fine controls for filtering
    #qdrant_upload() # to create qdrant collection and upload data in a generic fashion. Accomodates csv, pdf, word and text files
    #rag()  # to do RAG on one qdrant collection
    rag_sequential()  # to do RAG on one qdrant collection with weekly sequential chaining
    #rag_chain_collection()  # to do RAG on two collections
