"""
src/core/ingest.py

Multi-Format Financial RAG Ingestion Pipeline.

Implements a polymorphic file-routing factory that ingests heterogeneous
financial document formats (SEC 10-K/10-Q filings, legal contracts, market
news, macroeconomic summaries) from `data/raw/`, applies token-aware
recursive chunking aligned to the `cl100k_base` tokenizer, and enriches
every resulting chunk with deterministic, hash-based tracking metadata.

Design goals:
    - Zero silent failures: every I/O and parsing boundary is wrapped in
      explicit try/except blocks with structured logging.
    - Deterministic chunk identity: chunk_id is a pure function of
      (page_content, index), so re-running ingestion on unchanged source
      documents reproduces identical chunk_ids.
    - Format-agnostic output: regardless of source format, every chunk is
      emitted as a standard `langchain_core.documents.Document`.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("financial_rag.ingest")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_DIR: Path = Path("data/raw")
CHUNK_SIZE_TOKENS: int = 600
CHUNK_OVERLAP_TOKENS: int = 120
TOKEN_ENCODING_NAME: str = "cl100k_base"
STRUCTURAL_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]

DOC_TYPE_SEC_FILING: str = "sec_filing"
DOC_TYPE_LEGAL_CONTRACT: str = "legal_contract"
DOC_TYPE_MARKET_NEWS: str = "market_news"

SEC_FILENAME_MARKERS: Tuple[str, ...] = ("10k", "10q", "sec")

SUPPORTED_PDF_EXTENSIONS: frozenset = frozenset({".pdf"})
SUPPORTED_TEXT_EXTENSIONS: frozenset = frozenset({".txt", ".md"})


@dataclass
class RawPage:
    """Container for a single unit of raw extracted text prior to chunking.

    Attributes:
        text: The raw extracted text content for this unit.
        page_number: The 1-indexed page number (defaults to 1 for flat
            text documents that have no native page concept).
        doc_type: The taxonomy classification for the source document.
        source_file: The clean file name string of the origin document.
    """

    text: str
    page_number: int
    doc_type: str
    source_file: str


class DocumentTypeClassifier:
    """Classifies raw files into a financial document taxonomy based on
    filename heuristics and file extension.
    """

    @staticmethod
    def classify_pdf(file_path: Path) -> str:
        """Classify a PDF file as either a SEC filing or a legal contract.

        Args:
            file_path: Path to the candidate PDF file.

        Returns:
            The doc_type taxonomy string: "sec_filing" if the filename
            contains a SEC marker ("10k", "10q", "sec"), otherwise
            "legal_contract".
        """
        normalized_name = file_path.stem.lower()
        if any(marker in normalized_name for marker in SEC_FILENAME_MARKERS):
            return DOC_TYPE_SEC_FILING
        return DOC_TYPE_LEGAL_CONTRACT

    @staticmethod
    def classify_text(file_path: Path) -> str:
        """Classify a flat text/markdown file as market news.

        Args:
            file_path: Path to the candidate text file.

        Returns:
            The doc_type taxonomy string: "market_news".
        """
        return DOC_TYPE_MARKET_NEWS


class PDFParser:
    """Handles structural extraction of PDF documents using pdfplumber,
    preserving table/column layout alignment for financial statements and
    contract clause numbering.
    """

    @staticmethod
    def extract_pages(file_path: Path) -> List[RawPage]:
        """Extract layout-preserved text from every page of a PDF.

        Uses `page.extract_text(layout=True)` to retain whitespace-based
        column alignment, which is critical for correctly reading SEC
        financial statement tables and numbered legal contract clauses.

        Args:
            file_path: Path to the PDF file to parse.

        Returns:
            A list of RawPage objects, one per non-empty page. Returns an
            empty list if the file cannot be opened or parsed at all.
        """
        pages: List[RawPage] = []
        doc_type = DocumentTypeClassifier.classify_pdf(file_path)

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    try:
                        raw_text = page.extract_text(layout=True)
                    except Exception as page_error:  # noqa: BLE001
                        logger.warning(
                            "Failed to extract layout text on page %d of %s: %s",
                            page_index,
                            file_path.name,
                            page_error,
                        )
                        raw_text = None

                    if raw_text and raw_text.strip():
                        pages.append(
                            RawPage(
                                text=raw_text,
                                page_number=page_index,
                                doc_type=doc_type,
                                source_file=file_path.name,
                            )
                        )
                    else:
                        logger.debug(
                            "Skipping empty page %d in %s", page_index, file_path.name
                        )
        except Exception as file_error:  # noqa: BLE001
            logger.error(
                "Failed to open/parse PDF %s: %s", file_path.name, file_error
            )

        return pages


class TextFileParser:
    """Handles ingestion of flat .txt / .md market news and macroeconomic
    summary documents using native Python UTF-8 encoded stream reads.
    """

    @staticmethod
    def extract_pages(file_path: Path) -> List[RawPage]:
        """Read a plain text or markdown file as a single flat page.

        Args:
            file_path: Path to the .txt or .md file to read.

        Returns:
            A single-element list containing one RawPage, or an empty list
            if the file could not be decoded, read, or was empty.
        """
        doc_type = DocumentTypeClassifier.classify_text(file_path)

        try:
            with open(file_path, mode="r", encoding="utf-8") as stream:
                raw_text = stream.read()
        except UnicodeDecodeError as decode_error:
            logger.error(
                "UTF-8 decode failure reading %s: %s", file_path.name, decode_error
            )
            return []
        except OSError as os_error:
            logger.error("OS error reading %s: %s", file_path.name, os_error)
            return []

        if not raw_text or not raw_text.strip():
            logger.warning("File %s is empty; skipping.", file_path.name)
            return []

        return [
            RawPage(
                text=raw_text,
                page_number=1,
                doc_type=doc_type,
                source_file=file_path.name,
            )
        ]


class MetadataFactory:
    """Builds deterministic, permanence-guaranteed metadata dictionaries
    for every chunk emitted by the pipeline.
    """

    @staticmethod
    def build_chunk_id(source: str, page_content: str, index: int) -> str:
        """Construct a deterministic SHA-256 hash chunk identifier.

        The hash is computed over the source file name, the chunk's page
        content, and its positional index *local to that source file*.
        Including the source name prevents hash collisions between two
        different files whose Nth chunk happens to contain identical text
        (e.g. boilerplate legal headers, repeated table headers). Using a
        per-file local index rather than a global run-wide counter also
        means chunk_id is independent of file processing order: adding,
        removing, or reordering unrelated files elsewhere in data/raw/
        never changes an existing file's chunk_ids.

        Args:
            source: Clean file name string of the origin document. Scopes
                the hash so identical content in two different files never
                collides.
            page_content: The exact text content of the chunk.
            index: The positional index of the chunk within its own
                source file's chunk sequence (not the global run index).

        Returns:
            A hex-digest SHA-256 string uniquely identifying the chunk.
        """
        hash_input = f"{source}{page_content}{index}".encode("utf-8")
        return hashlib.sha256(hash_input).hexdigest()

    @staticmethod
    def build_metadata(
        source: str,
        chunk_content: str,
        chunk_index: int,
        page_number: int,
        doc_type: str,
    ) -> Dict[str, object]:
        """Assemble the full metadata dictionary for a single chunk.

        Args:
            source: Clean file name string of the origin document.
            chunk_content: The exact text content of the chunk.
            chunk_index: Positional index of the chunk within its own
                source file's chunk sequence (local index, not global).
            page_number: The page number the chunk originated from.
            doc_type: The taxonomy classification of the source document.

        Returns:
            A metadata dictionary containing "source", "chunk_id",
            "page_number", and "doc_type" keys.
        """
        return {
            "source": source,
            "chunk_id": MetadataFactory.build_chunk_id(
                source, chunk_content, chunk_index
            ),
            "page_number": page_number,
            "doc_type": doc_type,
        }


class FinancialIngestionPipeline:
    """Polymorphic multi-format ingestion factory for financial RAG.

    Scans a raw data directory, routes each file to the correct parser
    based on extension and filename heuristics, applies token-aware
    recursive chunking via LangChain's tiktoken-backed splitter, and
    enriches every chunk with permanent tracking metadata.
    """

    def __init__(
        self,
        raw_data_dir: Path = RAW_DATA_DIR,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
        encoding_name: str = TOKEN_ENCODING_NAME,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            raw_data_dir: Directory to scan for raw source documents.
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Token overlap between consecutive chunks.
            encoding_name: The tiktoken encoding scheme to align chunk
                boundaries against.
        """
        self.raw_data_dir = raw_data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name

        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.encoding_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=STRUCTURAL_SEPARATORS,
        )

    def _discover_files(self) -> List[Path]:
        """Discover all supported files within the raw data directory.

        Returns:
            A sorted list of file paths matching supported extensions. If
            the directory does not exist, an error is logged and an empty
            list is returned.
        """
        if not self.raw_data_dir.exists():
            logger.error(
                "Raw data directory does not exist: %s", self.raw_data_dir
            )
            return []

        supported_extensions = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_TEXT_EXTENSIONS
        discovered = sorted(
            path
            for path in self.raw_data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        )
        logger.info(
            "Discovered %d candidate file(s) in %s", len(discovered), self.raw_data_dir
        )
        return discovered

    def _route_file(self, file_path: Path) -> List[RawPage]:
        """Route a single file to the correct parser based on extension.

        Args:
            file_path: Path to the file to route and parse.

        Returns:
            A list of RawPage objects extracted from the file. Empty on
            failure or unsupported extension.
        """
        extension = file_path.suffix.lower()

        try:
            if extension in SUPPORTED_PDF_EXTENSIONS:
                return PDFParser.extract_pages(file_path)
            if extension in SUPPORTED_TEXT_EXTENSIONS:
                return TextFileParser.extract_pages(file_path)
        except Exception as routing_error:  # noqa: BLE001
            logger.error(
                "Unhandled routing error for %s: %s", file_path.name, routing_error
            )
            return []

        logger.warning(
            "Unsupported extension '%s' for file %s; skipping.",
            extension,
            file_path.name,
        )
        return []

    def _chunk_page(
        self, raw_page: RawPage, file_local_index: int
    ) -> Tuple[List[Document], int]:
        """Split a single RawPage into token-aware chunks with metadata.

        Args:
            raw_page: The RawPage to split into chunks.
            file_local_index: The current chunk index counter *scoped to
                this page's source file only* (resets to 0 for every new
                file). Combined with the source filename in
                MetadataFactory.build_chunk_id, this keeps chunk_id fully
                deterministic and independent of which other files exist
                in the run or what order they were processed in.

        Returns:
            A tuple of (list of Document chunks, updated file-local index).
        """
        documents: List[Document] = []

        try:
            split_texts = self.splitter.split_text(raw_page.text)
        except Exception as split_error:  # noqa: BLE001
            logger.error(
                "Chunking failure on %s (page %d): %s",
                raw_page.source_file,
                raw_page.page_number,
                split_error,
            )
            return documents, file_local_index

        for chunk_text in split_texts:
            if not chunk_text or not chunk_text.strip():
                continue

            metadata = MetadataFactory.build_metadata(
                source=raw_page.source_file,
                chunk_content=chunk_text,
                chunk_index=file_local_index,
                page_number=raw_page.page_number,
                doc_type=raw_page.doc_type,
            )
            documents.append(Document(page_content=chunk_text, metadata=metadata))
            file_local_index += 1

        return documents, file_local_index

    def run(self) -> List[Document]:
        """Execute the full ingestion pipeline end-to-end.

        Returns:
            A flat list of LangChain Document objects representing every
            chunk extracted and enriched across all discovered source
            files. Returns an empty list if no supported files are found
            or none could be successfully parsed.
        """
        all_documents: List[Document] = []

        files = self._discover_files()
        if not files:
            logger.warning(
                "No supported files found in %s. Nothing to ingest.",
                self.raw_data_dir,
            )
            return all_documents

        for file_path in files:
            logger.info("Processing file: %s", file_path.name)
            raw_pages = self._route_file(file_path)

            if not raw_pages:
                logger.warning("No extractable content found in %s.", file_path.name)
                continue

            # Reset the chunk index for every new source file (rather than
            # accumulating a single counter across the whole run). This
            # keeps each file's chunk_ids stable and reproducible on their
            # own, regardless of which other files are present in
            # data/raw/ or what order they happen to be discovered in.
            file_local_chunk_index = 0
            for raw_page in raw_pages:
                page_documents, file_local_chunk_index = self._chunk_page(
                    raw_page, file_local_chunk_index
                )
                all_documents.extend(page_documents)

            logger.info(
                "Completed %s -> %d page unit(s), %d chunk(s) produced.",
                file_path.name,
                len(raw_pages),
                file_local_chunk_index,
            )

        logger.info(
            "Ingestion complete. Total chunks generated: %d", len(all_documents)
        )
        return all_documents


class IngestionVerifier:
    """Performs post-ingestion verification, metrics computation, and
    structured audit logging for pipeline quality assurance.
    """

    def __init__(self, encoding_name: str = TOKEN_ENCODING_NAME) -> None:
        """Initialize the verifier with a tiktoken encoder for token
        payload measurement.

        Args:
            encoding_name: The tiktoken encoding scheme name to use for
                token counting.
        """
        self.encoding_name = encoding_name
        self.encoder: Optional[tiktoken.Encoding]
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception as encoder_error:  # noqa: BLE001
            logger.error(
                "Failed to load tiktoken encoding '%s': %s",
                encoding_name,
                encoder_error,
            )
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: The text to tokenize and count.

        Returns:
            The integer token count, or 0 if the encoder failed to load.
        """
        if self.encoder is None:
            return 0
        return len(self.encoder.encode(text, disallowed_special=()))

    def run_verification(self, documents: List[Document], elapsed_ms: float) -> None:
        """Execute the full verification and audit logging pass.

        Args:
            documents: The list of Document chunks produced by the
                ingestion pipeline.
            elapsed_ms: Total elapsed processing time in milliseconds.
        """
        logger.info("=" * 78)
        logger.info("FINANCIAL RAG INGESTION - VERIFICATION REPORT")
        logger.info("=" * 78)

        logger.info("Elapsed processing latency: %.3f ms", elapsed_ms)

        total_chars = sum(len(doc.page_content) for doc in documents)
        total_tokens = sum(self.count_tokens(doc.page_content) for doc in documents)

        logger.info("Total character mass:         %d characters", total_chars)
        logger.info("Total resolved token payload: %d tokens", total_tokens)

        if total_tokens > 0:
            compression_ratio = total_chars / total_tokens
            logger.info(
                "Character-to-token compression ratio: %.3f", compression_ratio
            )

        doc_type_counts: Dict[str, int] = {}
        for doc in documents:
            doc_type = str(doc.metadata.get("doc_type", "unknown"))
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        logger.info("-" * 78)
        logger.info("Chunk count by doc_type taxonomy:")
        for doc_type, count in sorted(doc_type_counts.items()):
            logger.info("  %-20s : %d chunk(s)", doc_type, count)
        logger.info("-" * 78)

        self._audit_sample_chunks(documents)

        logger.info("=" * 78)
        logger.info("VERIFICATION REPORT COMPLETE")
        logger.info("=" * 78)

    @staticmethod
    def _audit_sample_chunks(documents: List[Document]) -> None:
        """Print a visual audit log of representative chunks for manual
        layout integrity inspection.

        Prints the first available sec_filing chunk (to verify pdfplumber
        `layout=True` table alignment survived chunking) immediately
        followed by the first available market_news chunk (to visually
        prove format-agnostic chunk boundary handling).

        Args:
            documents: The list of Document chunks to sample from.
        """
        sec_chunk: Optional[Document] = next(
            (
                doc
                for doc in documents
                if doc.metadata.get("doc_type") == DOC_TYPE_SEC_FILING
            ),
            None,
        )
        news_chunk: Optional[Document] = next(
            (
                doc
                for doc in documents
                if doc.metadata.get("doc_type") == DOC_TYPE_MARKET_NEWS
            ),
            None,
        )

        logger.info("VISUAL LAYOUT AUDIT LOG")
        logger.info("-" * 78)

        if sec_chunk is not None:
            logger.info(
                "[SEC FILING TABLE LAYOUT SAMPLE] source=%s | page=%s | chunk_id=%s",
                sec_chunk.metadata.get("source"),
                sec_chunk.metadata.get("page_number"),
                sec_chunk.metadata.get("chunk_id"),
            )
            print("\n--- SEC FILING CHUNK (layout=True verification) ---")
            print(sec_chunk.page_content)
            print("--- END SEC FILING CHUNK ---\n")
        else:
            logger.warning(
                "No sec_filing chunks found; skipping PDF layout audit sample."
            )

        if news_chunk is not None:
            logger.info(
                "[MARKET NEWS TEXT SAMPLE] source=%s | page=%s | chunk_id=%s",
                news_chunk.metadata.get("source"),
                news_chunk.metadata.get("page_number"),
                news_chunk.metadata.get("chunk_id"),
            )
            print("\n--- MARKET NEWS CHUNK (format-agnostic boundary verification) ---")
            print(news_chunk.page_content)
            print("--- END MARKET NEWS CHUNK ---\n")
        else:
            logger.warning(
                "No market_news chunks found; skipping text audit sample."
            )


def main() -> None:
    """Entrypoint for standalone execution of the ingestion pipeline with
    full metric verification and audit logging.
    """
    logger.info("Booting Financial RAG Ingestion Pipeline...")

    pipeline = FinancialIngestionPipeline(
        raw_data_dir=RAW_DATA_DIR,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        encoding_name=TOKEN_ENCODING_NAME,
    )

    start_time = time.perf_counter()
    documents = pipeline.run()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if not documents:
        logger.warning(
            "No documents were produced. Verify that %s contains supported "
            "files (.pdf, .txt, .md).",
            RAW_DATA_DIR,
        )
        sys.exit(0)

    verifier = IngestionVerifier(encoding_name=TOKEN_ENCODING_NAME)
    verifier.run_verification(documents, elapsed_ms)


if __name__ == "__main__":
    main()