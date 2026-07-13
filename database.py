from pathlib import Path

import chromadb

from embedding_model import LegalEmebddings

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_COLLECTION = "Base_Books"
DEFAULT_FETCH_K = 12
DEFAULT_TOP_K = 6
MIN_RESULTS = 3
MAX_CHUNKS_PER_SOURCE = 2
NEIGHBOR_WINDOW = 1
RELATIVE_DISTANCE_FACTOR = 1.35


class LegalVectorStore:
    def __init__(
        self,
        persist_path: Path | str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        fetch_k: int = DEFAULT_FETCH_K,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.persist_path = str(persist_path or PROJECT_ROOT)
        self.collection_name = collection_name
        self.fetch_k = fetch_k
        self.top_k = top_k
        self._embedding_fn = LegalEmebddings()
        self._client = chromadb.PersistentClient(path=self.persist_path)
        self._collection = self._client.get_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
        )

    def get_context(self, prompt: str) -> str:
        chunks = self.retrieve(prompt)
        if not chunks:
            return ""

        return "\n\n".join(
            self._format_chunk(chunk["document"], chunk["metadata"])
            for chunk in chunks
        )

    def retrieve(self, prompt: str) -> list[dict]:
        query = prompt.strip()
        if not query:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=self.fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        candidates = self._parse_query_results(results)
        if not candidates:
            return []

        filtered = self._filter_by_relative_distance(candidates)
        diversified = self._diversify_by_source(filtered)
        expanded = self._expand_with_neighbors(diversified)

        return expanded[: self.top_k]

    def _parse_query_results(self, results: dict) -> list[dict]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
            if document
        ]

    def _filter_by_relative_distance(self, candidates: list[dict]) -> list[dict]:
        best_distance = candidates[0]["distance"]
        max_distance = best_distance * RELATIVE_DISTANCE_FACTOR

        filtered = [
            candidate
            for candidate in candidates
            if candidate["distance"] <= max_distance
        ]

        if len(filtered) < MIN_RESULTS:
            return candidates[:MIN_RESULTS]

        return filtered

    def _diversify_by_source(self, candidates: list[dict]) -> list[dict]:
        source_counts: dict[str, int] = {}
        diversified: list[dict] = []

        for candidate in candidates:
            source = candidate["metadata"].get("source", "unknown")
            if source_counts.get(source, 0) >= MAX_CHUNKS_PER_SOURCE:
                continue

            source_counts[source] = source_counts.get(source, 0) + 1
            diversified.append(candidate)

        return diversified

    def _expand_with_neighbors(self, candidates: list[dict]) -> list[dict]:
        seen_ids: set[str] = set()
        expanded: list[dict] = []

        for candidate in candidates:
            metadata = candidate["metadata"]
            source = metadata.get("source")
            chunk_idx = metadata.get("chunk")

            if source is None or chunk_idx is None:
                chunk_id = self._chunk_key(candidate)
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    expanded.append(candidate)
                continue

            neighbor_chunks = self._get_neighbor_chunks(source, int(chunk_idx))
            for neighbor in neighbor_chunks:
                chunk_id = self._chunk_key(neighbor)
                if chunk_id in seen_ids:
                    continue

                seen_ids.add(chunk_id)
                expanded.append(neighbor)

        expanded.sort(
            key=lambda chunk: (
                chunk["metadata"].get("source", ""),
                chunk["metadata"].get("chunk", 0),
            )
        )
        return expanded

    def _get_neighbor_chunks(self, source: str, chunk_idx: int) -> list[dict]:
        result = self._collection.get(
            where={
                "$and": [
                    {"source": {"$eq": source}},
                    {"chunk": {"$gte": chunk_idx - NEIGHBOR_WINDOW}},
                    {"chunk": {"$lte": chunk_idx + NEIGHBOR_WINDOW}},
                ]
            },
            include=["documents", "metadatas"],
        )

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        neighbors = [
            {"document": document, "metadata": metadata or {}, "distance": None}
            for document, metadata in zip(documents, metadatas)
            if document
        ]
        neighbors.sort(key=lambda chunk: chunk["metadata"].get("chunk", 0))
        return neighbors

    @staticmethod
    def _chunk_key(chunk: dict) -> str:
        metadata = chunk["metadata"]
        source = metadata.get("source", "unknown")
        chunk_idx = metadata.get("chunk", metadata.get("preview", chunk["document"][:80]))
        return f"{source}:{chunk_idx}"

    @staticmethod
    def _format_chunk(document: str, metadata: dict) -> str:
        source = metadata.get("source", "Unknown source")
        chunk_idx = metadata.get("chunk")
        header = f"[Source: {source}"
        if chunk_idx is not None:
            header += f", chunk {chunk_idx}"
        header += "]"
        return f"{header}\n{document}"


_default_store: LegalVectorStore | None = None


def get_store() -> LegalVectorStore:
    global _default_store
    if _default_store is None:
        _default_store = LegalVectorStore()
    return _default_store


def get_context(prompt: str) -> str:
    return get_store().get_context(prompt)
