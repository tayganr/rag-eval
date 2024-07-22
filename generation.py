import os  
import json  
import logging  
from dotenv import load_dotenv  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.models import VectorizedQuery, QueryType, QueryCaptionType, QueryAnswerType  
from openai import AzureOpenAI  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.messages import HumanMessage, SystemMessage

  
# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
# Load environment variables from .env file  
load_dotenv(dotenv_path=os.path.join('config', '.env'))  
  
def load_test_data(test_data_path):  
    with open(test_data_path, 'r') as file:  
        return json.load(file)  
  
def initialize_openai_client():  
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')  
    azure_openai_key = os.getenv('AZURE_OPENAI_KEY')  
    azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')  
    return AzureOpenAI(api_key=azure_openai_key, azure_endpoint=azure_openai_endpoint, api_version=azure_openai_api_version)  
  
def initialize_search_client():  
    service_name = os.getenv('AZURE_AI_SEARCH_SERVICE_NAME')  
    admin_key = os.getenv('AZURE_AI_SEARCH_ADMIN_KEY')  
    endpoint = f"https://{service_name}.search.windows.net"  
    credential = AzureKeyCredential(admin_key)  
    return SearchClient(endpoint=endpoint, index_name="example-index", credential=credential)  
  
def get_embedding(text, client, model_name):  
    response = client.embeddings.create(input=text, model=model_name)  
    return response.data[0].embedding  
  
def retrieve_context(question, embedding, search_client, retrieval_strategy):  
    search_type = retrieval_strategy["search_run_configuration"]["search_type"]  
    top_k = retrieval_strategy["search_run_configuration"]["top_k"]  
      
    if search_type == "keyword":  
        results = search_client.search(search_text=question, top=top_k)  
    elif search_type == "vector":  
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="contentVector")  
        results = search_client.search(search_text=None, vector_queries=[vector_query])  
    elif search_type == "hybrid":  
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="contentVector")  
        results = search_client.search(search_text=question, vector_queries=[vector_query], top=top_k)  
    elif search_type == "semantic":  
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="contentVector")  
        results = search_client.search(  
            search_text=question,  
            vector_queries=[vector_query],  
            query_type=QueryType.SEMANTIC,  
            semantic_configuration_name='my-semantic-config',  
            query_caption=QueryCaptionType.EXTRACTIVE,  
            query_answer=QueryAnswerType.EXTRACTIVE,  
            top=top_k  
        )  
    else:  
        raise ValueError(f"Unsupported search type: {search_type}")  
      
    contexts = [result['content'] for result in results]  
    return contexts  
  
def generate_answer(question, context, client, generation_strategy):  
    model_name = generation_strategy["model_configuration"]["model_name"]  
    deployment_name = generation_strategy["model_configuration"]["deployment_name"]  
    prompt_template_path = os.path.join('prompts', generation_strategy["model_configuration"]["prompt_template"])  
      
    with open(prompt_template_path, 'r') as file:  
        prompt_template = json.load(file)  
      
    template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template["system"]),
            ("human", prompt_template["human"])
        ]
    ) 

    prompt_value = template.format_messages(context=context, question=question)
    
    for message in prompt_value:  
        if isinstance(message, SystemMessage):  
            system_message = message.content
        elif isinstance(message, HumanMessage):  
            human_message = message.content
            
    messages = [  
        {"role": "system", "content": system_message},  
        {"role": "user", "content": human_message},  
    ]  
      
    response = client.chat.completions.create(  
        messages=messages,  
        model=model_name,  
        max_tokens=generation_strategy["model_configuration"]["max_tokens"],  
        temperature=generation_strategy["model_configuration"]["temperature"],  
        top_p=generation_strategy["model_configuration"]["top_p"]  
    )  
    
    generated_answer = response.choices[0].message.content.strip()

    return generated_answer, messages, response.usage.total_tokens  
  
def run_generation(eval_output_folder, embedding_strategy, retrieval_strategy, generation_strategy):  
    # Load test data  
    test_data_path = os.path.join('data', 'evaluation', 'test_data.json')  
    test_data = load_test_data(test_data_path)  
    questions = test_data["question"]  
    ground_truths = test_data["ground_truth"]  
    contexts = test_data["contexts"]  
      
    # Initialize clients  
    openai_client = initialize_openai_client()  
    search_client = initialize_search_client()  
      
    results = []  
      
    try:  
        for i, question in enumerate(questions):  
            logging.info(f"Processing question {i + 1} of {len(questions)}: {question}")  
              
            # Generate embedding for the question  
            question_embedding = get_embedding(question, openai_client, embedding_strategy['embedding_model_name']) 
              
            # Retrieve context based on the retrieval strategy  
            retrieved_contexts = retrieve_context(question, question_embedding, search_client, retrieval_strategy)  
            context = ' '.join(retrieved_contexts)  
              
            # Generate answer using the retrieved context  
            generated_answer, prompt, total_tokens = generate_answer(question, context, openai_client, generation_strategy)  
            print(generated_answer)
              
            result = {  
                "question": question,  
                "ground_truth": ground_truths[i],  
                "context": contexts[i],  
                "ai_generated_answer": generated_answer,  
                "prompt": prompt,  
                "total_input_tokens": total_tokens,  
                "total_output_tokens": len(generated_answer.split())  
            }  
            results.append(result)  
          
        # Save results  
        detailed_metrics_path = os.path.join(eval_output_folder, 'detailed_metrics.json')  
        with open(detailed_metrics_path, 'w', encoding='utf-8') as file:  
            json.dump(results, file, ensure_ascii=False, indent=4)  
          
    except Exception as e:  
        logging.error(f"An error occurred during evaluation: {e}")  
  
if __name__ == "__main__":  
    import argparse  
      
    parser = argparse.ArgumentParser(description="Run RAG generation.")  
    parser.add_argument("--eval_output_folder", type=str, required=True, help="Path to the evaluation output folder.")  
    parser.add_argument("--embedding_strategy", type=str, required=True, help="Embedding strategy in JSON format.")  
    parser.add_argument("--retrieval_strategy", type=str, required=True, help="Retrieval strategy in JSON format.")  
    parser.add_argument("--generation_strategy", type=str, required=True, help="Generation strategy in JSON format.")  
    args = parser.parse_args()  
      
    embedding_strategy = json.loads(args.embedding_strategy)  
    retrieval_strategy = json.loads(args.retrieval_strategy)  
    generation_strategy = json.loads(args.generation_strategy)  
      
    run_generation(args.eval_output_folder, embedding_strategy, retrieval_strategy, generation_strategy)  
