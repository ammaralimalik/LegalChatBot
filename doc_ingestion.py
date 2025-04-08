import os
import PyPDF2
from chromadb import PersistentClient
from embedding_model import LegalBERTEmebddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BOOK_FOLDER = './books'
COLLECTION_NAME = 'Base_Books'
CHUNK_SIZE = 750
CHUNK_OVERLAP = 200
CHROMA_PATH = '/Users/ammarmalik/Desktop/ResumeProjects/Legal AI Chatbot/'
def extract_text(file_path):
    text = ''
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text
    
def chunk_text(text,size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len,
        is_separator_regex=False,
        
    )
    docs = text_splitter.create_documents([text])
    chunks = text_splitter.split_documents(docs)
    return chunks

client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=LegalBERTEmebddings()
)

doc_id = 0

for filename in os.listdir(BOOK_FOLDER):
    if filename.endswith('.pdf'):
        filepath = os.path.join(BOOK_FOLDER, filename)
        text = extract_text(filepath)
        chunks = chunk_text(text)
        doc_id = 0

        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk.page_content],
                metadatas=[{'source':filename, 'chunk': i}],
                ids=[f"{filename}-{doc_id}"]
            )
            doc_id += 1
            print(f'Ingested {filename}:{doc_id}')
            
