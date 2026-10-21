FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY factorypulse/ factorypulse/
COPY services/ services/
COPY ml/ ml/
COPY models/ models/

EXPOSE 8001

CMD ["uvicorn", "services.inference.app:app", "--host", "0.0.0.0", "--port", "8001"]
