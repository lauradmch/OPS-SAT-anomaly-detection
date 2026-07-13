FROM python:3.12-slim

WORKDIR /app

# 1. CPU-only torch FIRST, from the CPU wheel index (no CUDA build)
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# 2. remaining serving deps (pinned to match the pickled artifacts)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 3. install the package that holds the predict path (no deps: they're above)
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# 4. app code + the baked model artifact
COPY api ./api
COPY model_artifact ./model_artifact

# 5. serve from the local artifact, local-path load instead of the registry alias
ENV MODEL_URI=/app/model_artifact
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]