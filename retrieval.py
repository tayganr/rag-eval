from azure.search.documents import SearchClient  
from azure.search.documents.models import VectorizedQuery  
from embedding import get_embedding  
  
def retrieve_context(query, embeddings, config):  
    service_name = os.getenv('AZURE_AI_SEARCH_SERVICE_NAME')  
    admin_key = os.getenv('AZURE_AI_SEARCH_ADMIN_KEY')  
    index_name = "example-index"  
  
    endpoint = f"https://{service_name}.search.windows.net"  
    credential = AzureKeyCredential(admin_key)  
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)  
  
    embedding = get_embedding(query)  
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=config.get("top_k", 10), fields="contentVector")  
    results = search_client.search(search_text=None, vector_queries=[vector_query], top=config.get("top_k", 10))  
  
    context = ""  
    for result in results:  
        context += result['content'] + "\n"  
    return context  
