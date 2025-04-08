import chromadb
from datetime import datetime
from embedding_model import LegalBERTEmebddings, test_get_embedding
chroma_client = chromadb.PersistentClient(path='/Users/ammarmalik/Desktop/ResumeProjects/Legal AI Chatbot/')
embedding_function = LegalBERTEmebddings()

chroma_client.delete_collection(name='Base_Books')