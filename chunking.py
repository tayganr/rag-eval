import os  
import json  
import logging  
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter  
  
# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
def read_text_files(folder_path):  
    text_files = []  
    for file_name in os.listdir(folder_path):  
        if file_name.endswith('.txt'):  
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:  
                text_files.append((file_name, file.read()))  
    return text_files  
  
def get_text_splitter(chunking_strategy):  
    method = chunking_strategy["chunking_method"]  
    params = chunking_strategy["chunking_parameters"]  
  
    if method == "split_by_characters":  
        return CharacterTextSplitter(  
            separator=params.get("separator", '\n\n'),  
            chunk_size=params["max_chunk_size"],  
            chunk_overlap=params["overlap_size"],  
            length_function=len,  
            is_separator_regex=params.get("is_separator_regex", False)  
        )  
    elif method == "recursively_split_by_characters":  
        return RecursiveCharacterTextSplitter(  
            separators=params.get("separators", [  
                "\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""  
            ]),  
            chunk_size=params["max_chunk_size"],  
            chunk_overlap=params["overlap_size"],  
            length_function=len,  
            is_separator_regex=params.get("is_separator_regex", False)  
        )  
    elif method == "split_by_tokens":  
        return TokenTextSplitter(  
            chunk_size=params["max_chunk_size"],  
            chunk_overlap=params["overlap_size"]  
        )  
    else:  
        raise ValueError(f"Unsupported chunking method: {method}")  
  
def chunk_text(text, chunking_strategy):  
    text_splitter = get_text_splitter(chunking_strategy)  
    return text_splitter.split_text(text)  
  
def save_chunked_texts(chunked_texts, output_dir, file_name):  
    os.makedirs(output_dir, exist_ok=True)  
    output_path = os.path.join(output_dir, f'{file_name}.json')  
    chunks = [{'chunk_id': idx, 'chunk_text': chunk} for idx, chunk in enumerate(chunked_texts)]  
    with open(output_path, 'w', encoding='utf-8') as file:  
        json.dump(chunks, file, ensure_ascii=False, indent=4)  
  
def process_documents(documents_folder, chunking_strategy, eval_folder):  
    chunked_texts_folder = os.path.join(eval_folder, 'chunked_texts')  
    print("########################\nCHUNKING\n########################")  
  
    text_files = read_text_files(documents_folder)  
    logging.info(f"Found {len(text_files)} documents to process.")  
    logging.info(f"Using chunking method: {chunking_strategy['chunking_method']}")  
    logging.info(f"Chunks will be saved to: {chunked_texts_folder}")  
  
    total_chunks = 0  
  
    for file_idx, (file_name, text) in enumerate(text_files):  
        logging.info(f"Chunking document {file_idx + 1} of {len(text_files)}: {file_name}")  
        chunked_texts = chunk_text(text, chunking_strategy)  
        logging.info(f"Created {len(chunked_texts)} chunks for document {file_name}")  
  
        save_chunked_texts(chunked_texts, chunked_texts_folder, file_name)  
        total_chunks += len(chunked_texts)  
  
    logging.info(f"Total number of chunks created: {total_chunks}")  
