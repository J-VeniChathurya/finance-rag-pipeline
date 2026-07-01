"""
src/core/index.py

Day 2 — Financial RAG Pipeline
==============================
Vector Provisioning & Search Indexing Engine.

Aligned to match the precise 'page_number' contract exposed by src/core/ingest.py.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Third-party / LangChain imports
# --------------------------------------------------------------------------
try:
    from langchain_core.documents import Document
except ImportError as import_error:
    raise ImportError(
        "langchain-core is required. Install it with: pip install langchain-core"
    ) from import_error

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    except ImportError as import_error:
        raise ImportError(
            "Could not import HuggingFaceEmbeddings. Install with: "
            "pip install langchain-huggingface sentence-transformers"
        ) from import_error

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma  # type: ignore
    except ImportError as import_error:
        raise ImportError(
            "Could not import Chroma. Install with: pip install langchain-chroma chromadb"
        ) from import_error

# --------------------------------------------------------------------------
# Upstream ingestion pipeline (Day 1 Contract Alignment)
# --------------------------------------------------------------------------
try:
    from src.core.ingest import FinancialIngestionPipeline
except ImportError as import_error:
    raise ImportError(
        "Could not import 'FinancialIngestionPipeline' from 'src.core.ingest'. "
        "Ensure src/core/ingest.py exists."
    ) from import_error


# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
LOG_DIR: Path = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger: logging.Logger = logging.getLogger("financial_rag.index")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _console_handler = logging.StreamHandler(stream=sys.stdout)
    _console_handler.setLevel(logging.INFO)

    _file_handler = logging.FileHandler(
        filename=LOG_DIR / "index_engine.log", encoding="utf-8"
    )
    _file_handler.setLevel(logging.DEBUG)

    _formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _console_handler.setFormatter(_formatter)
    _file_handler.setFormatter(_formatter)

    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)


# --------------------------------------------------------------------------
# Module-level constants (Aligned to ingest.py contract)
# --------------------------------------------------------------------------
DEFAULT_PERSIST_DIRECTORY: str = "data/chroma_db"
DEFAULT_RAW_DATA_DIRECTORY: str = "data/raw"
DEFAULT_COLLECTION_NAME: str = "financial_rag_collection"
DEFAULT_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K: int = 4

# ALIGNED WATCHPOINT: Changed 'page' to match ingest.py's fields
REQUIRED_METADATA_KEYS: Tuple[str, ...] = ("chunk_id", "source", "doc_type", "page_number")


# --------------------------------------------------------------------------
# Data contracts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoredChunk:
    """Immutable representation of a single retrieved chunk with its vector score."""

    text: str
    source: str
    page_number: Any  # Aligned key name
    doc_type: str
    chunk_id: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the scored chunk into a plain dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "page_number": self.page_number,
            "doc_type": self.doc_type,
            "chunk_id": self.chunk_id,
            "score": self.score,
        }


@dataclass
class IndexingReport:
    """Summary of a single build_index execution for operational logging."""

    total_documents_ingested: int = 0
    total_chunks_indexed: int = 0
    elapsed_seconds: float = 0.0
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY
    collection_name: str = DEFAULT_COLLECTION_NAME
    errors: List[str] = field(default_factory=list)


class IndexingError(Exception):
    """Raised when the vector store cannot be provisioned or populated."""


class SearchError(Exception):
    """Raised when a search operation against the vector store fails."""


# --------------------------------------------------------------------------
# Core indexing controller
# --------------------------------------------------------------------------
class FinancialIndexer:
    """Decoupled indexing controller handling storage and polymorphic hybrid searches."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        raw_data_dir: str = DEFAULT_RAW_DATA_DIRECTORY,
    ) -> None:
        self.persist_directory: str = persist_directory
        self.collection_name: str = collection_name
        self.embedding_model_name: str = embedding_model_name
        self.raw_data_dir: str = raw_data_dir

        self._ensure_persist_directory_exists()

        logger.info(
            "Initializing HuggingFaceEmbeddings with model '%s' on CPU device.",
            self.embedding_model_name,
        )
        try:
            self.embedding_function: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as embedding_init_error:
            logger.exception("Failed to initialize the embedding model.")
            raise IndexingError(
                f"Could not initialize HuggingFaceEmbeddings: {embedding_init_error}"
            ) from embedding_init_error

        logger.info(
            "Provisioning persistent ChromaDB collection '%s' at '%s'.",
            self.collection_name,
            self.persist_directory,
        )
        try:
            self.vector_store: Chroma = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )
        except Exception as store_init_error:
            logger.exception("Failed to initialize the Chroma vector store.")
            raise IndexingError(
                f"Could not initialize Chroma vector store: {store_init_error}"
            ) from store_init_error

        logger.info("FinancialIndexer initialized successfully.")

    def _ensure_persist_directory_exists(self) -> None:
        try:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        except OSError as os_error:
            logger.exception("Failed to create persist directory '%s'.", self.persist_directory)
            raise IndexingError(f"Unable to create persist directory: {os_error}") from os_error

    @staticmethod
    def _validate_document_metadata(document: Document, index_position: int) -> bool:
        if document.metadata is None:
            logger.warning("Document at index %d has no metadata; skipping.", index_position)
            return False

        missing_keys: List[str] = [
            key for key in REQUIRED_METADATA_KEYS if key not in document.metadata
        ]
        if missing_keys:
            logger.warning(
                "Document at index %d is missing required metadata keys %s; skipping.",
                index_position, missing_keys
            )
            return False

        chunk_id_value = document.metadata.get("chunk_id")
        if not isinstance(chunk_id_value, str) or not chunk_id_value.strip():
            logger.warning("Document at index %d has an invalid or empty 'chunk_id'; skipping.", index_position)
            return False

        return True

    def _extract_chunks_from_pipeline(self) -> List[Document]:
        logger.info("Running FinancialIngestionPipeline against '%s'.", self.raw_data_dir)
        try:
            # Matches your ingest.py instantiation perfectly
            ingestion_pipeline = FinancialIngestionPipeline(raw_data_dir=Path(self.raw_data_dir))
            extracted_documents = ingestion_pipeline.run()
        except Exception as ingestion_error:
            logger.exception("FinancialIngestionPipeline.run() raised an exception.")
            raise IndexingError(f"Ingestion pipeline execution failed: {ingestion_error}") from ingestion_error

        if not isinstance(extracted_documents, list):
            raise IndexingError("FinancialIngestionPipeline.run() must return a List[Document].")

        return extracted_documents

    def build_index(self, batch_size: int = 64) -> IndexingReport:
        start_time: float = time.perf_counter()
        report = IndexingReport(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        raw_documents: List[Document] = self._extract_chunks_from_pipeline()
        report.total_documents_ingested = len(raw_documents)

        if not raw_documents:
            raise IndexingError("No documents were returned by the ingestion pipeline.")

        valid_documents: List[Document] = []
        valid_ids: List[str] = []

        for position, document in enumerate(raw_documents):
            try:
                if not self._validate_document_metadata(document, position):
                    report.errors.append(f"Document at index {position} failed metadata validation.")
                    continue
                valid_documents.append(document)
                valid_ids.append(str(document.metadata["chunk_id"]))
            except Exception as validation_error:
                logger.exception("Unexpected error validating document at index %d.", position)
                report.errors.append(f"Document at index {position} error: {validation_error}")
                continue

        if not valid_documents:
            raise IndexingError(f"All chunks failed validation. Keys needed: {REQUIRED_METADATA_KEYS}")

        logger.info("Validated %d/%d chunks for indexing.", len(valid_documents), len(raw_documents))

        total_upserted: int = 0
        for batch_start in range(0, len(valid_documents), batch_size):
            batch_documents = valid_documents[batch_start : batch_start + batch_size]
            batch_ids = valid_ids[batch_start : batch_start + batch_size]

            try:
                self.vector_store.add_documents(documents=batch_documents, ids=batch_ids)
                total_upserted += len(batch_documents)
            except Exception as upsert_error:
                logger.exception("Failed to upsert batch starting at index %d.", batch_start)
                report.errors.append(f"Batch upsert index {batch_start} failed: {upsert_error}")
                continue

        if total_upserted == 0:
            raise IndexingError("Zero chunks were successfully upserted into ChromaDB.")

        report.total_chunks_indexed = total_upserted
        report.elapsed_seconds = round(time.perf_counter() - start_time, 3)
        return report

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredChunk]:
        if not isinstance(query, str) or not query.strip():
            raise SearchError("Query must be a non-empty string.")

        if top_k <= 0:
            raise SearchError(f"top_k must be a positive integer, received: {top_k}")

        search_mode: str = "metadata-filtered" if metadata_filter else "generic"
        logger.info("Executing %s search | query='%s' | top_k=%d", search_mode, query, top_k)

        try:
            raw_results: List[Tuple[Document, float]] = (
                self.vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=top_k,
                    filter=metadata_filter,
                )
            )
        except Exception as search_error:
            logger.exception("Similarity search failed for query '%s'.", query)
            raise SearchError(f"Similarity search failed: {search_error}") from search_error

        scored_chunks: List[ScoredChunk] = []
        for document, score in raw_results:
            metadata: Dict[str, Any] = document.metadata or {}
            scored_chunks.append(
                ScoredChunk(
                    text=document.page_content,
                    source=str(metadata.get("source", "unknown_source")),
                    page_number=metadata.get("page_number", "unknown_page"), # Aligned key mapping
                    doc_type=str(metadata.get("doc_type", "unknown_doc_type")),
                    chunk_id=str(metadata.get("chunk_id", "unknown_chunk_id")),
                    score=float(score),
                )
            )
        return scored_chunks

    def count(self) -> int:
        try:
            return self.vector_store._collection.count()  # noqa: SLF001
        except Exception:
            logger.exception("Failed to retrieve vector store count.")
            return -1


# --------------------------------------------------------------------------
# Verification / Smoke-test entry point
# --------------------------------------------------------------------------
def _print_scored_chunks(scored_chunks: List[ScoredChunk], limit: int = 2) -> None:
    if not scored_chunks:
        print("    No results returned.")
        return

    for rank, chunk in enumerate(scored_chunks[:limit], start=1):
        preview_text: str = chunk.text.strip().replace("\n", " ")
        if len(preview_text) > 220:
            preview_text = preview_text[:220].rstrip() + "..."

        print(f"    [{rank}] score={chunk.score:.4f}")
        print(f"        source : {chunk.source}")
        print(f"        page   : {chunk.page_number}") # Updated print call
        print(f"        text   : {preview_text}")


def main() -> int:
    logger.info("=" * 78)
    logger.info("FINANCIAL RAG PIPELINE — DAY 2 — INDEX ENGINE SMOKE TEST")
    logger.info("=" * 78)

    try:
        indexer = FinancialIndexer(
            persist_directory=DEFAULT_PERSIST_DIRECTORY,
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
            raw_data_dir=DEFAULT_RAW_DATA_DIRECTORY,
        )
    except IndexingError as init_error:
        logger.critical("Fatal initialization failure: %s", init_error)
        return 1

    provisioning_start: float = time.perf_counter()
    try:
        report: IndexingReport = indexer.build_index()
    except IndexingError as build_error:
        logger.critical("Fatal error during index build: %s", build_error)
        return 1
    except Exception as unexpected_error:
        logger.critical("Unexpected fatal error during index build: %s", unexpected_error)
        return 1
    provisioning_elapsed_ms: float = (time.perf_counter() - provisioning_start) * 1000.0

    print("\n" + "=" * 78)
    print("VECTOR DATABASE PROVISIONING REPORT")
    print("=" * 78)
    print(f"  Raw files ingested     : {report.total_documents_ingested}")
    print(f"  Chunks indexed         : {report.total_chunks_indexed}")
    print(f"  Elapsed time           : {report.elapsed_seconds:.3f} s ({provisioning_elapsed_ms:.3f} ms)")
    print(f"  Persist directory      : {report.persist_directory}")
    print(f"  Collection name        : {report.collection_name}")
    print(f"  Non-fatal errors       : {len(report.errors)}")

    # Test Query 1: Generic financial query
    generic_query: str = "What are the company's financial results or agreements?"
    print("\n" + "=" * 78)
    print(f"TEST QUERY 1 (generic, unfiltered): '{generic_query}'")
    print("=" * 78)
    try:
        generic_results: List[ScoredChunk] = indexer.search(
            query=generic_query, top_k=DEFAULT_TOP_K, metadata_filter=None
        )
        _print_scored_chunks(generic_results, limit=2)
    except SearchError as search_error:
        logger.error("Test Query 1 failed: %s", search_error)

    # Test Query 2: Metadata-filtered query (doc_type == "sec_filing")
    filtered_query: str = "What is Apple's revenue or trading symbol?"
    metadata_filter: Dict[str, Any] = {"doc_type": "sec_filing"}
    print("\n" + "=" * 78)
    print(f"TEST QUERY 2 (metadata-filtered, doc_type='sec_filing'): '{filtered_query}'")
    print("=" * 78)
    try:
        filtered_results: List[ScoredChunk] = indexer.search(
            query=filtered_query, top_k=DEFAULT_TOP_K, metadata_filter=metadata_filter
        )
        _print_scored_chunks(filtered_results, limit=2)
    except SearchError as search_error:
        logger.error("Test Query 2 failed: %s", search_error)

    print("\n" + "=" * 78)
    print("SMOKE TEST COMPLETE")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())