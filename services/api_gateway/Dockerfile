FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY factorypulse/ factorypulse/
COPY services/ services/
COPY ml/ ml/
COPY rag/ rag/
COPY agents/ agents/
COPY data/ data/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "services.api_gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
