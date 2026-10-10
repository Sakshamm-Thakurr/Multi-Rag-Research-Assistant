# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_DATA_DIR=/data

# OpenMP runtime used by faiss (slim images do not always include it)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# never run as root; /data holds the per-document indexes
RUN useradd --create-home --uid 10001 app \
 && mkdir /data \
 && chown app:app /data

WORKDIR /app

# dependencies first: this layer is cached until a requirements file changes
COPY requirements-core.txt requirements-api.txt ./
RUN pip install -r requirements-core.txt -r requirements-api.txt

# only the code the API needs (no tests, no evaluation data, no .env, no sample PDFs)
COPY core/ core/
COPY agents/ agents/
COPY api/ api/

USER app
VOLUME /data
EXPOSE 8000

# the service is "healthy" when /health answers 200
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]