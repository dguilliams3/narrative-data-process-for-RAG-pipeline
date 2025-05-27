"""
Object Oriented GPT - A Python package for managing GPT conversations with context management and document retrieval.

This package provides tools for:
- Managing GPT conversations with context and summarization
- Handling text summarization and chunking
- Elasticsearch and FAISS integration for semantic document retrieval
- FastAPI service for GPT interactions
- Document QA and keyword extraction
"""

__version__ = "0.1.0"

# Core GPT functionality
from .GPTClient import GPTClient, GPTServiceError
from .chunk_manager import ChunkManager
from .gpt_service import app as gpt_service_app
from .gpt_config import (
    GPT_MODEL,
    GPT_MAX_TOKENS,
    GPT_TEMPERATURE,
    GPT_MAX_CONTEXT_TOKENS,
    ROLE_ANSWER,
    MAIN_PROMPT_TEMPLATE,
    SUMMARIZER_PROMPT_TEMPLATE,
    ELASTICSEARCH_HOST,
    FAISS_TOP_K,
    ES_NUMBER_DOCS
)

# Document retrieval and processing
from .elasticsearch_and_faiss_query_line import (
    query_faiss,
    retrieve_documents_from_elasticsearch,
    extract_keywords_with_keybert,
    test_qa_pipeline
)

# Logging utilities
from .logging_utils import configure_logging, ensure_log_dir

__all__ = [
    # Core GPT
    'GPTClient',
    'GPTServiceError',
    'ChunkManager',
    'gpt_service_app',
    
    # Configuration
    'GPT_MODEL',
    'GPT_MAX_TOKENS',
    'GPT_TEMPERATURE',
    'GPT_MAX_CONTEXT_TOKENS',
    'ROLE_ANSWER',
    'MAIN_PROMPT_TEMPLATE',
    'SUMMARIZER_PROMPT_TEMPLATE',
    'ELASTICSEARCH_HOST',
    'FAISS_TOP_K',
    'ES_NUMBER_DOCS',
    
    # Document retrieval
    'query_faiss',
    'retrieve_documents_from_elasticsearch',
    'extract_keywords_with_keybert',
    'test_qa_pipeline',
    
    # Utilities
    'configure_logging',
    'ensure_log_dir'
]
