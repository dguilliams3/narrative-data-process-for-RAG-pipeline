# GPT Service Architecture Documentation

## Overview

This system implements a microservice-based architecture for handling GPT interactions with context management and RAG (Retrieval-Augmented Generation) capabilities. The system is containerized using Docker and consists of three main services:

1. GPT Service (FastAPI)
2. RAG API (Flask)
3. Elasticsearch

## Service Components

### 1. GPT Service (`gpt_service.py`)

A FastAPI service that provides a clean interface to OpenAI's GPT models.

**Key Features:**
- Handles direct OpenAI API interactions
- Provides response metadata (tokens, duration, model info)
- Implements health checks and metrics
- Uses Pydantic for request/response validation

**Response Format:**
```json
{
    "content": "Generated text response",
    "model": "Model used for generation",
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    },
    "finish_reason": "stop",
    "duration": 1.23
}
```

### 2. GPT Client (`GPTClient.py`)

A Python client that manages conversations and context.

**Key Features:**
- Context management via ChunkManager
- RAG integration
- Token counting and model selection
- Response metadata tracking

**Main Methods:**
- `send_prompt()`: Main conversation interface
- `summarize_text()`: Text summarization
- `update_context_with_rag()`: RAG context injection
- `select_model_by_token_length()`: Smart model selection

### 3. RAG API (`gpt_general_questions.py`)

A Flask service that provides the user interface and handles RAG operations.

**Key Features:**
- Web interface for user interaction
- RAG toggle functionality
- Context management
- Response sanitization

**Endpoints:**
- `/ask`: Main question endpoint
- `/health`: Health check
- `/metrics`: Usage metrics
- `/logs`: Log viewing (authenticated)

## Data Flow

1. **User Input Flow:**
   ```
   User -> RAG API -> GPT Client -> GPT Service -> OpenAI API
   ```

2. **RAG Flow:**
   ```
   Query -> FAISS -> Elasticsearch -> Document Retrieval -> GPT Summarization -> Context Injection
   ```

## Configuration

The system uses environment variables for configuration:

**GPT Service:**
- `GPT_MODEL`: Model to use
- `GPT_MAX_TOKENS`: Max tokens per response
- `GPT_TEMPERATURE`: Response randomness
- `OPENAI_API_KEY`: OpenAI API key

**RAG Settings:**
- `ELASTICSEARCH_HOST`: Elasticsearch connection
- `ES_PASSWORD`: Elasticsearch password
- `RAG_SUMMARY_CONTEXT_CHUNKS`: Context window size

## Docker Configuration

Services are orchestrated using Docker Compose:

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.1
    ports: ["9200:9200"]
    
  rag-api:
    build: .
    ports: ["5000:5000"]
    
  gpt-service:
    build:
      context: ./app
      dockerfile: Dockerfile
    ports: ["8001:8001"]
```

## Response Metadata

The system tracks and provides detailed metadata for each response:

1. **Token Usage:**
   - Prompt tokens
   - Completion tokens
   - Total tokens

2. **Performance Metrics:**
   - Request duration
   - Model used
   - Finish reason

3. **Context Information:**
   - Current context size
   - Token count
   - RAG status

## Error Handling

The system implements comprehensive error handling:

1. **GPT Service Errors:**
   - API failures
   - Invalid responses
   - Timeout handling

2. **RAG Errors:**
   - Document retrieval failures
   - Summarization errors
   - Context management issues

## Logging

Structured logging is implemented across all services:

- Request/response logging
- Error tracking
- Performance metrics
- Context changes

## Security

1. **Authentication:**
   - Basic auth for log access
   - API key management
   - Environment variable protection

2. **Data Protection:**
   - Response sanitization
   - Input validation
   - Secure API key handling

## Monitoring

The system provides several monitoring endpoints:

1. **Health Checks:**
   - Service status
   - Uptime
   - Request counts

2. **Metrics:**
   - Token usage
   - Response times
   - Context sizes

## Future Improvements

1. **Planned Features:**
   - Enhanced RAG capabilities
   - Better context management
   - Advanced model selection

2. **Potential Optimizations:**
   - Caching layer
   - Load balancing
   - Rate limiting

## Development Guidelines

1. **Code Style:**
   - Follow PEP 8
   - Use type hints
   - Document all functions

2. **Testing:**
   - Unit tests for core functionality
   - Integration tests for services
   - Performance benchmarks

## Deployment

1. **Requirements:**
   - Docker
   - Docker Compose
   - Environment variables

2. **Steps:**
   ```bash
   # Build and start services
   docker-compose up --build
   
   # Check logs
   docker-compose logs -f
   
   # Monitor health
   curl http://localhost:5000/health
   ``` 