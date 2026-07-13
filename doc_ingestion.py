import argparse
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import PyPDF2
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_model import LegalEmebddings

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = str(Path(__file__).resolve().parent)
HEADING_REGEX = r'(?i)(Chapter|Section|Article|Part|Clause|Rule|Subsection)[\s\-]*\d+[A-Za-z\-]*\.?'


@dataclass
class IngestionConfig:
    """Configuration for a document ingestion run."""

    book_folder: str = './books'
    collection_name: str = 'Base_Books'
    chroma_path: str = DEFAULT_CHROMA_PATH
    chunk_size: int = 1500
    chunk_overlap: int = 175
    # Number of chunks embedded/written per collection call. Keeps memory bounded
    # for the transformer embedding model.
    batch_size: int = 32
    # Optional pause (seconds) between files to relieve CPU pressure on machines
    # without a GPU. Defaults to 0 (no pause).
    per_file_delay: float = 0.0
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", " ", ""]
    )


class DocumentIngestor:
    """Ingests PDF documents into a persistent Chroma collection.

    Chunking is heading-aware first (splitting on legal headings such as
    "Section 5" or "Article 12") and then recursively size-limited. Writes are
    idempotent: re-running upserts chunks under deterministic ids instead of
    duplicating them.
    """

    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig()
        self._client = None
        self._collection = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
        )

    @property
    def collection(self) -> Collection:
        """Lazily create the client/collection so importing has no side effects."""
        if self._collection is None:
            self._client = PersistentClient(path=self.config.chroma_path)
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=LegalEmebddings(),
            )
        return self._collection

    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text from a PDF, skipping pages that fail to parse."""
        text_parts: List[str] = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                except Exception as exc:  # a single bad page shouldn't abort the file
                    logger.warning("Failed to read page %d of %s: %s", page_num, file_path, exc)
                    continue
                if page_text:
                    text_parts.append(page_text)
        return '\n'.join(text_parts)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into heading-aware, size-bounded chunks."""
        split_by_headings = re.split(HEADING_REGEX, text)
        pre_chunks: List[str] = []

        for i in range(1, len(split_by_headings), 2):
            heading = split_by_headings[i].strip()
            body = split_by_headings[i + 1].strip() if i + 1 < len(split_by_headings) else ""
            combined = f"{heading}\n{body}".strip()
            if combined:
                pre_chunks.append(combined)

        # Fall back to the whole document when no legal headings were found.
        if not pre_chunks:
            stripped = text.strip()
            if stripped:
                pre_chunks.append(stripped)

        all_chunks: List[str] = []
        for chunk in pre_chunks:
            docs = self._splitter.create_documents([chunk])
            sub_chunks = self._splitter.split_documents(docs)
            all_chunks.extend(doc.page_content for doc in sub_chunks)

        return all_chunks

    def ingest_file(self, file_path: str) -> int:
        """Ingest a single PDF file. Returns the number of chunks written."""
        filename = os.path.basename(file_path)
        logger.info("Processing %s", filename)

        text = self.extract_text(file_path)
        if not text.strip():
            logger.warning("No extractable text in %s; skipping", filename)
            return 0

        chunks = self.chunk_text(text)
        if not chunks:
            logger.warning("No chunks produced for %s; skipping", filename)
            return 0

        written = 0
        for start in range(0, len(chunks), self.config.batch_size):
            batch = chunks[start:start + self.config.batch_size]
            ids = [f"{filename}-{start + offset}" for offset in range(len(batch))]
            metadatas = [
                {
                    'source': filename,
                    'chunk': start + offset,
                    'length': len(content),
                    'preview': content[:100],
                }
                for offset, content in enumerate(batch)
            ]
            self.collection.upsert(documents=batch, metadatas=metadatas, ids=ids)
            written += len(batch)
            logger.info("Upserted %d/%d chunks for %s", written, len(chunks), filename)

        return written

    def ingest_directory(self) -> int:
        """Ingest every PDF in the configured book folder. Returns total chunks."""
        folder = self.config.book_folder
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Book folder not found: {folder}")

        pdf_files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.pdf'))
        if not pdf_files:
            logger.warning("No PDF files found in %s", folder)
            return 0

        total = 0
        for filename in pdf_files:
            file_path = os.path.join(folder, filename)
            try:
                total += self.ingest_file(file_path)
            except Exception:  # keep going even if one file is corrupt
                logger.exception("Failed to ingest %s", filename)
                continue

            if self.config.per_file_delay > 0:
                time.sleep(self.config.per_file_delay)

        logger.info("Ingestion complete: %d chunks across %d files", total, len(pdf_files))
        return total


def _parse_args(argv: List[str] | None = None) -> IngestionConfig:
    parser = argparse.ArgumentParser(description="Ingest legal PDFs into Chroma.")
    parser.add_argument('--book-folder', default=IngestionConfig.book_folder)
    parser.add_argument('--collection-name', default=IngestionConfig.collection_name)
    parser.add_argument('--chroma-path', default=DEFAULT_CHROMA_PATH)
    parser.add_argument('--chunk-size', type=int, default=IngestionConfig.chunk_size)
    parser.add_argument('--chunk-overlap', type=int, default=IngestionConfig.chunk_overlap)
    parser.add_argument('--batch-size', type=int, default=IngestionConfig.batch_size)
    parser.add_argument('--per-file-delay', type=float, default=IngestionConfig.per_file_delay)
    args = parser.parse_args(argv)

    return IngestionConfig(
        book_folder=args.book_folder,
        collection_name=args.collection_name,
        chroma_path=args.chroma_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        per_file_delay=args.per_file_delay,
    )


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    config = _parse_args(argv)
    DocumentIngestor(config).ingest_directory()


if __name__ == '__main__':
    main()
