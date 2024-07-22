import os  
import json  
import time  
import argparse  
from datetime import datetime  
from chunking import process_documents  
from embedding import process_embeddings  
from indexing import process_indexing
from generation import run_generation  
from evaluation import perform_rag_evaluation  
  
def load_config(config_path):  
    with open(config_path, 'r') as file:  
        return json.load(file)  
  
def load_test_data(test_data_path):  
    with open(test_data_path, 'r') as file:  
        return json.load(file)  
  
def main(config_path):  
    # Load configuration  
    config = load_config(config_path)  
    metadata = config.get("evaluation_metadata", {})  
    chunking_strategy = config.get("chunking_strategy", {})  
    embedding_strategy = config.get("embedding_strategy", {})  
    retrieval_strategy = config.get("retrieval_strategy", {})  
    generation_strategy = config.get("generation_strategy", {})  
  
    # Load test data  
    test_data_path = os.path.join('data', 'evaluation', 'test_data.json')  
    test_data = load_test_data(test_data_path)  
    questions = test_data["question"]  
    ground_truths = test_data["ground_truth"]  
    contexts = test_data["contexts"]  
  
    # Create dynamic output folder based on current time  
    eval_folder = f'eval_{datetime.now().strftime("%Y%m%d_%H%M")}'  
    eval_output_folder = os.path.join('evaluations', eval_folder)  
  
    # Start evaluation  
    start_time = time.time()  
  
    try:  
        # # Chunking  
        documents_folder = 'data/source_documents_txt'  
        process_documents(documents_folder, chunking_strategy, eval_output_folder)  
  
        # # Embedding  
        process_embeddings(eval_output_folder, embedding_strategy)
        
        # # Indexing  
        process_indexing(eval_output_folder) 
        
        # eval_output_folder = 'evaluations/eval_20240722_0735'
        
        # Generation
        run_generation(eval_output_folder, embedding_strategy, retrieval_strategy, generation_strategy)        
  
        # Evaluation
        perform_rag_evaluation(eval_output_folder, embedding_strategy, generation_strategy)  
  
        # # Log results  
        # end_time = time.time()  
        # log_evaluation(metadata, start_time, end_time, chunking_strategy, embedding_strategy, retrieval_strategy, generation_strategy, results)  
  
    except Exception as e:  
        print(f"An error occurred during evaluation: {e}")  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")  
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation configuration file.")  
    args = parser.parse_args()  
    main(args.config)  
