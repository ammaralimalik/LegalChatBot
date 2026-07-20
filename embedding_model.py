from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from transformers import AutoModel, AutoTokenizer

import torch

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
# legal-bert is a BERT-base checkpoint: 512 wordpiece positions, no more.
MAX_LENGTH = 512
# Chunks are embedded in slices so a large upsert batch can't spike memory.
ENCODE_BATCH_SIZE = 16


class LegalEmebddings(EmbeddingFunction):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: list[list[float]] = []
        for start in range(0, len(input), ENCODE_BATCH_SIZE):
            batch = input[start:start + ENCODE_BATCH_SIZE]
            embeddings.extend(self._encode(batch))
        return embeddings

    def _encode(self, batch: Documents) -> list[list[float]]:
        inputs = self.tokenizer(
            list(batch),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            hidden_state = self.model(**inputs).last_hidden_state

        # Mean-pool over real tokens only. Averaging the raw last_hidden_state
        # would fold [PAD] positions into the mean, so a short chunk padded to
        # the length of the longest chunk in its batch gets its signal diluted
        # and its embedding drifts toward whatever the padding encodes.
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # Chroma scores with squared L2 by default. On unit vectors that is a
        # monotonic function of cosine similarity, so normalizing here is what
        # makes distance reflect meaning rather than vector magnitude (which
        # otherwise tracks chunk length).
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.tolist()

    def name(self) -> str:
        return 'legal-bert'
