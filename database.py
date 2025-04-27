import chromadb
from datetime import datetime
from embedding_model import LegalEmebddings, test_get_embedding

chroma_client = chromadb.PersistentClient(path='/Users/ammarmalik/Desktop/ResumeProjects/LegalChatBot/')

def get_context(prompt):
    collection = chroma_client.get_collection(name="Base_Books", embedding_function=LegalEmebddings())
    
    context = collection.query(
        query_texts=prompt,
        n_results=3
    )
    context = ""
    return context