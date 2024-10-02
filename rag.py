# Create a RAG LLM using Qdrant vector store. Read multiple user queries and return agent output

from typing import Any, List, Dict
from pydantic import Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain import hub
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from vector_db import Qdrantdb
from utils.utils import get_weekly_filters

import sys
import logging
import os
import json
from dotenv import load_dotenv

# create and configure logger
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%dT%H:%M:%S',
                    format='%(asctime)-15s.%(msecs)03dZ %(levelname)-7s : %(name)s - %(message)s',
                    handlers=[logging.FileHandler("llm.log"), logging.StreamHandler(sys.stdout)])
# create log object with current module name
log = logging.getLogger(__name__)

# Get the logger for 'httpx'
httpx_logger = logging.getLogger("httpx")

# Set the logging level to WARNING to ignore INFO and DEBUG logs
httpx_logger.setLevel(logging.WARNING)

# Load .env file
load_dotenv()


class CustomRetriever(BaseRetriever) :
    """
    Custom Retriver class for the user defined collection
    """
    collection_name: str = Field(...)

    def __init__(self, collection_name:str):
        super().__init__()
        self.collection_name = collection_name
        

    def get_similar_docs(self, user_query:str, filters:List[Any]=None):
        """
        Method to get top similar docs from vector store
        Args:
            user_query: str
            filters: List[Any]
        Returns:
            List[Document] : list of semantically similar documents retrieved from vector store
        """
        qdrantdb = Qdrantdb(collection_name=self.collection_name)
        try:
            filters = json.loads(os.getenv("FILTERS")) if not filters else filters
            # Retrieve similar documents based on the user input vector
            results = qdrantdb.vector_search(user_query=user_query, collection_name=self.collection_name, filters=filters)
            log.info("Retrieved similar documents successfully.")
            return results
        except Exception as e:
            log.error(f"Failed to retrieve similar documents: {str(e)}")
            return []

    def _get_relevant_documents(self, user_query: str, filters:List[Any]=None, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Method to get relevant documents
        """
        log.info("Get relevent docs for query: %s . Query filters: %s", user_query, filters)
        search_docs = self.get_similar_docs(user_query=user_query, filters=filters) 
        found_docs = []
        for doc in search_docs:
            try:
                payload = doc.payload  # {'content': 'str', 'row': 69, 'source': 'communication_channel_2_thread_multi_3.csv'}
                score = doc.score
                doc_content = payload["content"]
                doc_metadata = {key: value for key, value in payload.items() if key != "content"}
                doc_metadata["score"] = score
                found_docs.append(Document(page_content=doc_content, metadata=doc_metadata))
            except Exception as e:
                log.error(f"Error in rag search : {str(e)}")            
        return found_docs



def create_retriever(vector_store):
    """
    Create a RAG QA retriver
    """

    llm: ChatOpenAI = ChatOpenAI(
            temperature=0,
            model="gpt-4-0125-preview",
            max_retries=500,
            )
    retriever = vector_store.as_retriever()  # search_type="mmr"
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever)
    
    return qa


def create_custom_retriever(collection_name):
    """
    Create a custom RAG QA retriever
    """
    qa_prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If the question has some initial findings, use that as context.
    2. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer but you may want to check the following sourcess** and add the source documents as a list.
    3. If you find the answer, write the answer in a concise way and add the list of sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.

    {context}

    Question: {question}
    Helpful Answer:"""
    
    qa_chain_prompt = hub.pull("rlm/rag-prompt") #PromptTemplate.from_template(qa_prompt_template)

    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="Context:\n{page_content}",
    )

    retriever = CustomRetriever(collection_name=collection_name)
    llm: ChatOpenAI = ChatOpenAI(
            temperature=0,
            model="gpt-4o", 
            max_retries=500,
        )
    llm_chain = LLMChain(llm=llm, prompt=qa_chain_prompt, callbacks=None, verbose=False)
    combine_documents_chain = StuffDocumentsChain(
                                llm_chain=llm_chain,
                                document_variable_name="context",
                                document_prompt=document_prompt,
                                callbacks=None,
                            )
    qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            callbacks=None,
            verbose=False,
            retriever=retriever,
            return_source_documents=True,
        )
    return qa


def create_weekly_custom_retriever(collection_name:str, previous_week_context:str):
    """
    Create a custom RAG QA retriever with weekly filters
    """
    
    qa_weekly_prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context of the current week and the previous week to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.
    Question: {question} 
    Context: {context} 
    Previous week context: {previous_week_context}
    Answer:
        """
    
    qa_weekly_prompt = PromptTemplate(
        input_variables=["question", "context"], 
        template=qa_weekly_prompt_template,
        partial_variables={
            "previous_week_context": previous_week_context
        }
    )


    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="Context:\n{page_content}",
    )

    retriever = CustomRetriever(collection_name=collection_name)
    llm: ChatOpenAI = ChatOpenAI(
            temperature=0,
            model="gpt-4o", 
            max_retries=500,
        )
    llm_chain = LLMChain(llm=llm, prompt=qa_weekly_prompt, callbacks=None, verbose=False, )
    combine_documents_chain = StuffDocumentsChain(
                                llm_chain=llm_chain,
                                document_variable_name="context",
                                document_prompt=document_prompt,
                                callbacks=None, 
                            )
    qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            callbacks=None,
            verbose=False,
            retriever=retriever,
            return_source_documents=True,
        )
    return qa

    
def rag_results(vector_store, queries: List[str]):
    """
    Call retriever for user queries
    """
    retriever = create_retriever(vector_store=vector_store)
    log.info("Calling llm with batch queries")
    responses = retriever.batch(queries)
    log.info("QA retrieval completed")
    return responses


def rag_search_results(queries: List[Any], collection_name:str):
    """
    RAG method
    Get similar documents form vector search
    User queries can be a list of str. Utilize filters to get the relevant documents using the env variable
    Give the documents and user query as context to LLM
    Return answers with source citations
    """
    retriever = create_custom_retriever(collection_name=collection_name)
    log.info("Calling llm with batch queries")
    responses = retriever.batch(queries)
    log.info("QA retrieval completed")
    return responses


def rag_weekly_sequential(queries: List[str], collection_name:str, start_week:str, end_week:str, filters:List[Any]=None):
    """
    RAG method for weekly sequential RAG
    """
    # get the weekly filter
    weekly_filters = get_weekly_filters(start_week, end_week)
    combined_filters = []
    if filters and len(filters) > 0:
        # Ensure filters is a list of dictionaries
        if isinstance(filters, dict):
            filters = [filters]  # Convert to list if it's a single dictionary
        elif isinstance(filters, str):
            filters = eval(filters)  # Convert string representation of list/dict to actual list/dict
        combined_filters = [{**f, **w} for f in filters for w in weekly_filters]
    else:
        combined_filters = weekly_filters
    log.info("Query filters: %s", combined_filters)

    responses = []
    previous_week_context = ""
    for query in queries:
        # TODO : check if creating retriever for each query is efficient
        retriever = create_weekly_custom_retriever(collection_name=collection_name, previous_week_context=previous_week_context)
        log.info("Invoking llm for query: %s", query)
        for filter in combined_filters:
            input_dict = {
                "query": query,
                "filters": filter,
                "previous_week_context": previous_week_context
            }
            response = retriever.invoke(input_dict)
            responses.append(response)
            previous_week_context = response
    log.info("QA retrieval completed")
    return combined_filters, responses


