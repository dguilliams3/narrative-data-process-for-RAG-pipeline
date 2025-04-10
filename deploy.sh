#!/bin/bash
# Default to Dockerfile if no argument is given
DOCKERFILE="${1:-Dockerfile}"

echo "🧹 Bringing down existing containers and removing volumes..."
docker-compose down -v

echo "🐳 Building rag-api image using $DOCKERFILE"
docker build -f "$DOCKERFILE" -t dguilliams3/rag-api:latest .

echo "🚀 Starting containers..."
docker-compose up -d

echo "⏳ Sleeping for 60 seconds to let Elasticsearch boot fully..."
sleep 60
echo "⏳ Sleeping another 30 seconds..."
sleep 30

# Detect if venv exists inside container
if docker-compose exec rag-api test -d /app/venv; then
  PYTHON_CMD="./venv/bin/python"
else
  PYTHON_CMD="python"
fi

echo "📡 Populating Elasticsearch with data..."
docker-compose exec rag-api bash -c "$PYTHON_CMD ./populate_elasticsearch.py"

echo "✅ Done! Visit your Flask server at: http://localhost:5000"
echo "🔍 Visit your Elasticsearch server at: http://localhost:9200"
