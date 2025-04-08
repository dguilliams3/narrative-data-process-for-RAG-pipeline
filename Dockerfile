# ==========================
# Stage 1: Builder
# ==========================
FROM python:3.12-slim AS builder
WORKDIR /app

# Rust & build tools for tokenizers
RUN apt-get update && apt-get install -y \
    curl build-essential libgomp1 git \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable \
    && ln -s $HOME/.cargo/bin/rustc /usr/local/bin/rustc \
    && ln -s $HOME/.cargo/bin/cargo /usr/local/bin/cargo \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies in isolated virtualenv
COPY requirements.txt /app/
RUN python -m venv venv && \
    venv/bin/pip install --upgrade pip && \
    venv/bin/pip install --no-cache-dir -r requirements.txt

# ==========================
# Stage 2: Runtime
# ==========================
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy environment and code
COPY --from=builder /app/venv /app/venv
COPY . /app

ENV ELASTICSEARCH_HOST=http://elasticsearch:9200
ENV ELASTICSEARCH_PORT=9200
ENV FLASK_APP=gpt_general_questions.py
ENV FLASK_ENV=production

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

ENTRYPOINT ["/app/venv/bin/python"]
CMD ["/app/app/gpt_general_questions.py"]
