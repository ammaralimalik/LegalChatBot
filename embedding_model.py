from transformers import AutoTokenizer, AutoModel
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

import torch

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

def test_get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.numpy().tolist()

class LegalBERTEmebddings(EmbeddingFunction):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.model.eval()
        
    def __call__(self, input: Documents) -> Embeddings:
        inputs = self.tokenizer(input, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedddings = outputs.last_hidden_state[:,0,:]
        return embedddings.numpy().tolist()
    
    def name(self) -> str:
        return 'legal-bert'