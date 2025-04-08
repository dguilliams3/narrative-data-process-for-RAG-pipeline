#!/bin/bash

echo "🧹 Bringing down existing containers and removing volumes..."
docker-compose down -v

echo "🐳 Rebuilding containers..."
docker-compose build

echo "🚀 Starting containers..."
docker-compose up -d

echo "⏳ Sleeping for 60 seconds to let Elasticsearch boot fully..."
sleep 60

echo "⏳ Sleeping another 30 seconds..."
sleep 30

echo "📡 Populating Elasticsearch with data..."
docker-compose exec rag-api bash -c './venv/bin/python ./populate_elasticsearch.py'

echo "✅ Done! Visit your Flask server at: http://localhost:5000"
echo "🔍 Visit your Elasticsearch server at: http://localhost:9200"