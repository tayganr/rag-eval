import os  
import json  
import uuid  
import logging  
from dotenv import load_dotenv  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.indexes.models import (  
    SimpleField, SearchFieldDataType, SearchableField, SearchField,  
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,  
    SemanticConfiguration, SemanticPrioritizedFields, SemanticField,  
    SemanticSearch, SearchIndex  
)  
from azure.search.documents import SearchClient  
  
# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)   
  
# Load environment variables from .env file  
load_dotenv(dotenv_path=os.path.join('config', '.env'))  
  
def initialize_search_clients():  
    service_name = os.getenv('AZURE_AI_SEARCH_SERVICE_NAME')  
    admin_key = os.getenv('AZURE_AI_SEARCH_ADMIN_KEY')  
    endpoint = f"https://{service_name}.search.windows.net"  
    credential = AzureKeyCredential(admin_key)  
      
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)  
    return index_client, endpoint, credential  
  
def create_or_update_index(index_client, index_name):  
    # Define the index schema  
    fields = [  
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),  
        SimpleField(name="title", type=SearchFieldDataType.String),  
        SimpleField(name="chunk_id", type=SearchFieldDataType.String),  
        SearchableField(name="content", type=SearchFieldDataType.String),  
        SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")  
    ]  
  
    # Configure the vector search configuration  
    vector_search = VectorSearch(
        algorithms=[  
            HnswAlgorithmConfiguration(  
                name="myHnsw"  
            )  
        ],  
        profiles=[  
            VectorSearchProfile(  
                name="myHnswProfile",  
                algorithm_configuration_name="myHnsw",  
            )  
        ]  
    )  
  
    # Define the semantic configuration  
    semantic_config = SemanticConfiguration(  
        name="my-semantic-config",  
        prioritized_fields=SemanticPrioritizedFields(  
            title_field=SemanticField(field_name="title"),  
            content_fields=[SemanticField(field_name="content")]  
        )  
    )  
  
    # Create the semantic settings with the configuration  
    semantic_search = SemanticSearch(configurations=[semantic_config])  
  
    try:  
        index_client.delete_index(index_name)  
        logging.info(f"Index '{index_name}' deleted successfully.")  
    except Exception as e:  
        logging.warning(f"Index '{index_name}' does not exist or could not be deleted: {e}")  
  
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)  
    result = index_client.create_or_update_index(index)  
    logging.info(f"Index '{index_name}' created successfully with vector search and semantic search configurations.")  
  
def load_and_upload_chunked_data(search_client, embeddings_folder):  
    documents = []  
    for file_name in os.listdir(embeddings_folder):  
        if file_name.endswith('.json'):  
            with open(os.path.join(embeddings_folder, file_name), 'r', encoding='utf-8') as file:  
                chunked_data = json.load(file)  
                for chunk in chunked_data:  
                    document_id = str(uuid.uuid4())  
                    documents.append({  
                        "id": document_id,  
                        "title": file_name.replace('.json', ''),  
                        "chunk_id": str(chunk['chunk_id']),  
                        "content": chunk['chunk_text'],  
                        "contentVector": chunk['chunk_embedding']  
                    })  
      
    result = search_client.upload_documents(documents=documents)  
    
    # Summarize the upload results  
    successful_uploads = sum(1 for r in result if r.succeeded)  
    failed_uploads = len(result) - successful_uploads  
  
    logging.info(f"Documents uploaded successfully: {successful_uploads}")  
    if failed_uploads > 0:  
        logging.warning(f"Documents failed to upload: {failed_uploads}")  
  
    for r in result:  
        if not r.succeeded:  
            logging.error(f"Document ID {r.key} failed to upload with error: {r.error_message}")  
  
def process_indexing(eval_output_folder): 
    print("########################\nINDEXING\n########################")   
    index_name = "example-index"  
      
    index_client, endpoint, credential = initialize_search_clients()  
    create_or_update_index(index_client, index_name)  
      
    embeddings_folder = os.path.join(eval_output_folder, 'embeddings')  
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)  
    load_and_upload_chunked_data(search_client, embeddings_folder)  
  
if __name__ == "__main__":  
    import sys  
    if len(sys.argv) != 2:  
        raise ValueError("Usage: indexing.py <eval_output_folder>")  
      
    eval_output_folder = sys.argv[1]  
    process_indexing(eval_output_folder)  
