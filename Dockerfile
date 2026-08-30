FROM python:3.12-slim

WORKDIR /app

# Hugging Face writes its cache here. TRANSFORMERS_CACHE is deprecated in
# favour of HF_HOME, so only the latter is set.
ENV HF_HOME=/app/.cache
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api ./api
COPY ./db ./db
COPY ./ingest ./ingest
COPY ./retrieval ./retrieval
COPY ./generation ./generation

# Bake both encoders into the image. A cold Space would otherwise spend its
# first request downloading them, on top of building the term index.
RUN python -c "from ingest.embed import model; model()" \
    && python -c "from retrieval.rerank import model; model()"

RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

EXPOSE 7860

# The platform injects $PORT. Exec form would not expand it, so this goes
# through sh; "exec" keeps uvicorn as PID 1 so it still receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
