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

EXPOSE 8002

CMD ["uvicorn", "services.copilot.app:app", "--host", "0.0.0.0", "--port", "8002"]
