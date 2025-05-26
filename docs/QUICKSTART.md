# Quick Start Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- OpenAI API key
- Elasticsearch credentials

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your credentials:
   ```
   OPENAI_API_KEY=your_api_key
   ES_PASSWORD=your_elasticsearch_password
   ELASTICSEARCH_HOST=http://elasticsearch:9200
   ```

3. **Build and start services:**
   ```bash
   docker-compose up --build
   ```

## Testing the Setup

1. **Check service health:**
   ```bash
   curl http://localhost:5000/health
   ```

2. **Send a test query:**
   ```bash
   curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"user_input": "What is the main theme of the story?"}'
   ```

## Development

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r app/requirements_gpt_service.txt
   pip install -r app/requirements_elasticsearch.txt
   ```

2. **Run services locally:**
   ```bash
   # Terminal 1 - GPT Service
   cd app
   uvicorn gpt_service:app --host 0.0.0.0 --port 8001

   # Terminal 2 - RAG API
   cd app
   python gpt_general_questions.py
   ```

### Common Tasks

1. **View logs:**
   ```bash
   curl -u admin:password http://localhost:5000/logs
   ```

2. **Check metrics:**
   ```bash
   curl http://localhost:5000/metrics
   ```

3. **Toggle RAG:**
   ```bash
   curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"user_input": "rag"}'
   ```

## Response Format

Example response from the `/ask` endpoint:
```json
{
    "response": "The generated text response",
    "duration": 1.23,
    "model": "gpt-4",
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    },
    "finish_reason": "stop"
}
```

## Troubleshooting

1. **Service won't start:**
   - Check Docker logs: `docker-compose logs`
   - Verify environment variables
   - Check port availability

2. **RAG not working:**
   - Verify Elasticsearch connection
   - Check document indexing
   - Review RAG toggle status

3. **High latency:**
   - Check token usage
   - Review context size
   - Monitor system resources

## Next Steps

1. Review the full [Architecture Documentation](ARCHITECTURE.md)
2. Set up monitoring
3. Configure logging
4. Implement custom prompts

## Support

For issues and feature requests, please:
1. Check existing documentation
2. Review open issues
3. Create a new issue with:
   - Description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details 