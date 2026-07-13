import argparse
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import PyPDF2
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection

from embedding_model import LegalEmebddings

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = str(Path(__file__).resolve().parent)
# Legal headings (e.g. "Section 5", "Article 12") and numbered sections (e.g. "302. Punishment").
HEADING_REGEX = re.compile(
    r'(?i)(?:'
    r'(?:Chapter|Section|Article|Part|Clause|Rule|Subsection|Order)[\s\-:]*\d+[A-Za-z\-]*\.?'
    r'|'
    r'\d{1,4}[A-Za-z]?\.(?=\s+[A-Z])'
    r')'
)


@dataclass
class IngestionConfig:
    """Configuration for a document ingestion run."""

    book_folder: str = './books'
    collection_name: str = 'Base_Books'
    chroma_path: str = DEFAULT_CHROMA_PATH
    # Number of chunks embedded/written per collection call. Keeps memory bounded
    # for the transformer embedding model.
    batch_size: int = 32
    # Optional pause (seconds) between files to relieve CPU pressure on machines
    # without a GPU. Defaults to 0 (no pause).
    per_file_delay: float = 0.0


class DocumentIngestor:
    """Ingests PDF documents into a persistent Chroma collection.

    Chunking is heading-aware: each chunk is a legal heading and the text
    beneath it (e.g. "Section 5" plus its body, or "302. Punishment..." plus
    its body). Writes are idempotent: re-running upserts chunks under
    deterministic ids instead of duplicating them.
    """

    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig()
        self._client = None
        self._collection = None

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

    @staticmethod
    def _extract_heading(chunk: str) -> str:
        """Return the heading label at the start of a chunk, if present."""
        match = HEADING_REGEX.search(chunk)
        return match.group(0).strip() if match else ""

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of each heading and the text beneath it."""
        matches = list(HEADING_REGEX.finditer(text))
        if not matches:
            stripped = text.strip()
            return [stripped] if stripped else []

        chunks: List[str] = []

        preamble = text[:matches[0].start()].strip()
        if preamble:
            chunks.append(preamble)

        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

        return chunks

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
                    'heading': self._extract_heading(content),
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
    parser.add_argument('--batch-size', type=int, default=IngestionConfig.batch_size)
    parser.add_argument('--per-file-delay', type=float, default=IngestionConfig.per_file_delay)
    args = parser.parse_args(argv)

    return IngestionConfig(
        book_folder=args.book_folder,
        collection_name=args.collection_name,
        chroma_path=args.chroma_path,
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
