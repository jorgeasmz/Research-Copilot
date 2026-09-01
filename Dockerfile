# The encoders are exported here and copied forward as graphs. torch is needed
# to write them and never to run them, so it stays out of the image that serves.
FROM python:3.12-slim AS encoders

WORKDIR /build
ENV HF_HOME=/build/.cache

COPY requirements.txt requirements-offline.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-offline.txt

COPY ./ingest ./ingest
COPY ./retrieval ./retrieval
COPY ./tools ./tools
RUN python -m tools.export_encoders \
    && rm -f artifacts/onnx/*/model-fp32.onnx


FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENCODER_DIR=/app/artifacts/onnx

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --from=encoders /build/artifacts/onnx ./artifacts/onnx
COPY ./api ./api
COPY ./db ./db
COPY ./ingest ./ingest
COPY ./retrieval ./retrieval
COPY ./generation ./generation
COPY ./alembic ./alembic
COPY alembic.ini .

EXPOSE 8000

# The platform injects $PORT. Exec form would not expand it, so this goes
# through sh; "exec" keeps uvicorn as PID 1 so it still receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
