import json  
import os  
from datasets import Dataset  
from ragas import evaluate  
from ragas.metrics import AnswerRelevancy, AnswerSimilarity, ContextPrecision  
from langchain_openai.chat_models import AzureChatOpenAI  
from langchain_openai.embeddings import AzureOpenAIEmbeddings  
from dotenv import load_dotenv  
import pandas as pd 
  
load_dotenv(dotenv_path=os.path.join('config', '.env'))
  
def prepare_dataset(questions, generated_answers, ground_truths, contexts):  
    return {  
        'question': questions,  
        'answer': generated_answers,  
        'ground_truth': ground_truths,  
        'contexts': contexts  
    }  
  
def perform_rag_evaluation(eval_output_folder, embedding_strategy, generation_strategy):  
    dataset_path = os.path.join(eval_output_folder, 'detailed_metrics.json')  
  
    # Load the JSON file containing the evaluation data  
    with open(dataset_path, 'r', encoding='utf-8') as f:  
        data = json.load(f)  
  
    questions = [item['question'] for item in data]  
    generated_answers = [item['ai_generated_answer'] for item in data]  
    ground_truths = [item['ground_truth'] for item in data]  
    contexts = [item['context'] for item in data]  
  
    dataset_dict = prepare_dataset(questions, generated_answers, ground_truths, contexts)  
    dataset = Dataset.from_dict(dataset_dict)  
  
    metrics = [  
        AnswerRelevancy(),  
        AnswerSimilarity(),  
        ContextPrecision()  
    ]  
  
    azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')  
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')  
    azure_openai_key = os.getenv('AZURE_OPENAI_KEY')  
  
    azure_model = AzureChatOpenAI(  
        openai_api_version=azure_openai_api_version,  
        azure_endpoint=azure_openai_endpoint,  
        azure_deployment=generation_strategy["model_configuration"]["deployment_name"],  
        model=generation_strategy["model_configuration"]["model_name"],  
        validate_base_url=False,  
        api_key=azure_openai_key  
    )  
  
    azure_embeddings = AzureOpenAIEmbeddings(  
        openai_api_version=azure_openai_api_version,  
        azure_endpoint=azure_openai_endpoint,  
        azure_deployment=embedding_strategy["embedding_deployment_name"],  
        model=embedding_strategy["embedding_model_name"],  
        api_key=azure_openai_key  
    )  
  
    results = evaluate(  
        dataset=dataset,  
        metrics=metrics,  
        llm=azure_model,  
        embeddings=azure_embeddings  
    )  
    print(results)
    
    pd.set_option("display.max_colwidth", None)
    df = results.to_pandas()
    df
    
    # export df to csv
    df.to_csv(os.path.join(eval_output_folder, 'evaluation_results.csv'), index=False)