FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEMOTRON_MODEL=nvidia/nemotron-3-super-120b-a12b \
    NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1 \
    OUTPUT_DIR=outputs

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY requirements-deploy.txt ./requirements-deploy.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -r requirements-deploy.txt

COPY . .
RUN mkdir -p outputs logs rag_sources medical_inputs assets

EXPOSE 8080
CMD ["uvicorn", "app_nemoclaw:app", "--host", "0.0.0.0", "--port", "8080"]
