# Vectore Store using Qdrant

import uuid
import sys
import logging
import os
from ast import literal_eval
from dotenv import load_dotenv
from typing import List
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.utils import create_filter_conditions
from utils.data_loaders import get_csv_data_fine_control, get_data_generic


# Load .env file
load_dotenv(override=True)

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


class Qdrantdb:
    """
    Qdrant client for Qdrant server hosted in QDRANT_URL
    """
    def __init__(self, collection_name:str) -> None:
        self.collection_name = collection_name
        self.qdrant_url = os.getenv('QDRANT_URL', "localhost")
        self.qdrant_port = os.getenv('QDRANT_PORT', "6333")
        self.qdrant_client = QdrantClient(url=self.qdrant_url, port=self.qdrant_port, timeout=300) 
        self.embeddings = OpenAIEmbeddings()

    def get_qdrant_client(self):
        return self.qdrant_client
    
    def collection_exists(self) -> bool:
        return self.qdrant_client.collection_exists(self.collection_name)
    
    def create_collection(self) -> bool:
        status = self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(distance=Distance.COSINE, size=1536),
            )
        return status
    
    def get_collection_count(self) -> int:
        return self.qdrant_client.count(collection_name=self.collection_name)

    
    def upsert_data_generic(self, data_dir_path: str) -> bool:
        """
        Method to upload data to qdrant collection
        Filtering can be applied only to metadata - not to content of file
        Metadata - row:int, source:str
        Payload format : {"source":"/path/to/file", "content":"file content"}
        """
        texts , metadatas = get_data_generic(data_dir_path)
        return self.upsert_data(texts, metadatas)
        

    def upsert_csvdata_fine_control(self, data_dir_path: str) -> bool:
        """
        Method to upload data to qdrant collection with metadata columns as filter
        """
        texts , metadatas = get_csv_data_fine_control(data_dir_path)
        return self.upsert_data(texts, metadatas)
        

    def upsert_data(self, texts, metadatas):
        """
        Split and upload data to qdrant collection
        Return True if successful, False otherwise
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=150,
                separators=[
                    "\n\n", "\n", ". ", " ", ""
                ]  # try to split on paragraphs... fallback to sentences, then chars, ensure we always fit in context window
            )
        log.info("Creating documents from text splitter")
        docs: List[Document] = text_splitter.create_documents(texts=texts, metadatas=metadatas)  # gives a Document class with attributes page_content and metadata
        log.info("Creating text embeddings")
        text_embeddings = self.embeddings.embed_documents([doc.page_content for doc in docs])
        #text_embeddings_dict = [{texts[i]: text_embeddings[i]} for i in range(len(texts))]
        payload = [{**doc.metadata, "content": doc.page_content} for doc in docs]

        # Qdrant bulk upsert
        vectors: list[PointStruct] = []
        log.info("Convert to vector")
        vectors = [PointStruct(id=str(uuid.uuid4()), vector=text_embeddings[i], payload=payload[i]) for i in range(len(docs))]
        
        try:
            log.info("Qdrant bulk upload of vector points")
            self.qdrant_client.upload_points(collection_name=self.collection_name, points=vectors)
            return True
        except Exception as e:
            log.error(f"Failed to upsert data into collection: {self.collection_name}. Error: {e}")
            if hasattr(e, 'message'):
                log.error(f"Message: {e.message} ")  
            elif hasattr(e, 'response'):
                log.error(f"Status code: {e.response.status_code}, Body: {e.response.body}")  # 
            return False
        pass


    def vector_store(self):
        """
        Method to return a qdrant vector store
        """
        # create vector Store
        vector_store = Qdrant(client=self.qdrant_client,
                        embeddings=self.embeddings,
                        collection_name=self.collection_name,
                        metadata_payload_key="metadata",
                        content_payload_key="content"
                    )
        return vector_store


    def vector_search(self, user_query, collection_name=None, filters=[]):
        """
        Method to search for user query in qdrant collection
        """
        if isinstance(user_query, str):
            user_query_embedding = self.embeddings.embed_query(user_query)
        elif isinstance(user_query, dict):
            if 'query' in user_query:
                user_query_embedding = self.embeddings.embed_query(user_query['query'])
            else:
                log.error("Invalid user query format. Expected 'query' key in dictionary.")
                return None
            if 'filters' in user_query:
                filters = user_query['filters']

        must_filters = None
        if filters:
            # create must filter conditions
            filter_conditions = create_filter_conditions(filters)
            must_filters = models.Filter(must=filter_conditions)

        top_n = 30 # return n closest points
        score_threshold = 0.6
        log.info("Search filters: %s", must_filters)
        if not collection_name:
            collection_name = self.collection_name

        search_results = self.qdrant_client.search(collection_name=collection_name,
                                            query_vector=user_query_embedding,
                                            query_filter=must_filters,
                                            limit=top_n,
                                            score_threshold=score_threshold
                                            )
        log.info("Search results count: %s", len(search_results))
        return search_results
