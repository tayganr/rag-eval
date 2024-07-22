import os  
import json  
import logging  
import time  
from dotenv import load_dotenv  
from openai import AzureOpenAI  
  
# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
httpx_logger = logging.getLogger("httpx")  
httpx_logger.setLevel(logging.WARNING)  
  
# Load environment variables from .env file  
load_dotenv(dotenv_path=os.path.join('config', '.env'))  
  
# Initialize Azure OpenAI client  
def initialize_openai_client(embedding_strategy):  
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')  
    azure_openai_key = os.getenv('AZURE_OPENAI_KEY')  
    azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')  
    azure_openai_embedding_model = embedding_strategy['embedding_model_name']  
    azure_openai_embedding_deployment = embedding_strategy['embedding_deployment_name']  
  
    client = AzureOpenAI(  
        azure_deployment=azure_openai_embedding_deployment,  
        api_version=azure_openai_api_version,  
        azure_endpoint=azure_openai_endpoint,  
        api_key=azure_openai_key  
    )  
    return client, azure_openai_embedding_model  
  
def read_chunked_texts(chunked_texts_folder):  
    chunked_texts = []  
    for file_name in os.listdir(chunked_texts_folder):  
        if file_name.endswith('.json'):  
            with open(os.path.join(chunked_texts_folder, file_name), 'r', encoding='utf-8') as file:  
                chunked_texts.append((file_name, json.load(file)))  
    return chunked_texts  
  
def generate_embeddings(client, embedding_model, chunked_texts, embeddings_output_folder):  
    total_chunks = sum(len(doc_chunks) for _, doc_chunks in chunked_texts)  
    processed_chunks = 0  
  
    for file_idx, (file_name, doc_chunks) in enumerate(chunked_texts):  
        for chunk in doc_chunks:  
            chunk_id = chunk['chunk_id']  
            chunk_text = chunk['chunk_text']  
            response = client.embeddings.create(input=[chunk_text], model=embedding_model)  
            chunk['chunk_embedding'] = response.data[0].embedding  
            processed_chunks += 1  
  
            # Log progress percentage at the chunk level  
            percentage = (processed_chunks / total_chunks) * 100  
            print(f"\rProcessing chunk {processed_chunks} of {total_chunks} ({percentage:.2f}%)", end='')  
  
        # Save updated document with embeddings  
        output_path = os.path.join(embeddings_output_folder, file_name)  
        with open(output_path, 'w', encoding='utf-8') as file:  
            json.dump(doc_chunks, file, ensure_ascii=False, indent=4)  
  
    print()  # Move to the next line after progress is complete  
  
def process_embeddings(eval_folder, embedding_strategy):  
    chunked_texts_folder = os.path.join(eval_folder, 'chunked_texts')  
    embeddings_output_folder = os.path.join(eval_folder, 'embeddings')  
    os.makedirs(embeddings_output_folder, exist_ok=True)  # Ensure the output directory exists  
    print("########################\nEMBEDDING\n########################")  
    logging.info(f"Using embedding model: {embedding_strategy['embedding_model_name']}")  
  
    # Initialize OpenAI client  
    client, embedding_model = initialize_openai_client(embedding_strategy)  
    
    # Read chunked texts  
    chunked_texts = read_chunked_texts(chunked_texts_folder)  
    total_chunks = sum(len(doc_chunks) for _, doc_chunks in chunked_texts)  
    logging.info(f"Found {total_chunks} chunks to process across {len(chunked_texts)} documents.")
  
    # Generate embeddings  
    start_time = time.time()  
    generate_embeddings(client, embedding_model, chunked_texts, embeddings_output_folder)  
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    logging.info(f"Generated embeddings for all chunks in {elapsed_time:.2f} seconds.")  
  
if __name__ == "__main__":  
    import sys  
  
    if len(sys.argv) != 3:  
        raise ValueError("Usage: embedding.py <eval_folder> <embedding_strategy>")  
  
    eval_folder = sys.argv[1]  
    embedding_strategy = json.loads(sys.argv[2])  
  
    process_embeddings(eval_folder, embedding_strategy)  
