import os
import PyPDF2
import time
import re
from chromadb import PersistentClient
from embedding_model import LegalEmebddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BOOK_FOLDER = './books'
COLLECTION_NAME = 'Base_Books'
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 175
CHROMA_PATH = '/LegalChatbot/'

HEADING_REGEX = r'(?i)(Chapter|Section|Article|Part|Clause|Rule|Subsection)[\s\-]*\d+[A-Za-z\-]*\.?'


def extract_text(file_path):
    text = ''
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
 
    split_by_headings = re.split(HEADING_REGEX, text)
    pre_chunks = []
    
    for i in range(1, len(split_by_headings), 2):
        heading = split_by_headings[i].strip()
        body = split_by_headings[i+1].strip() if i + 1 < len(split_by_headings) else ""
        combined = f"{heading}\n{body}".strip()
        if combined:
            pre_chunks.append(combined)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    all_chunks = []
    for chunk in pre_chunks:
        docs = text_splitter.create_documents([chunk])
        sub_chunks = text_splitter.split_documents(docs)
        all_chunks.extend(sub_chunks)

    return all_chunks

client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=LegalEmebddings()
)

doc_id = 0

for filename in os.listdir(BOOK_FOLDER):
    if filename.endswith('.pdf'):
        filepath = os.path.join(BOOK_FOLDER, filename)
        print(f"Processing: {filename}")
        text = extract_text(filepath)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk.page_content],
                metadatas=[{
                    'source': filename,
                    'chunk': i,
                    'length': len(chunk.page_content),
                    'preview': chunk.page_content[:100]
                }],
                ids=[f"{filename}-{doc_id}"],
            )
            doc_id += 1
            print(f'Ingested {filename}:{doc_id}')

        time.sleep(45)
