FROM python:3.12-slim

WORKDIR /app

# Install build tools needed for tokenizers + curl for healthcheck
RUN apt-get update && apt-get install -y \
    curl build-essential libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first, to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY app/ .
COPY populate_elasticsearch.py . 
COPY summaries.json .            

# Expose port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the app directly with system Python (no venv)
CMD ["python", "gpt_general_questions.py"] 