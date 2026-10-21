FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY services/ services/

CMD ["python", "-m", "services.simulator", "--machines", "10", "--rate", "1000", "--target-url", "http://ingestion:8003/api/v1/telemetry/batch"]
