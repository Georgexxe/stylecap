FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/Georgexxe/stylecap" \
      org.opencontainers.image.title="StyleCap" \
      org.opencontainers.image.description="Grounded four-style video captioning with Gemma"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY . .

ENTRYPOINT ["python", "run.py"]
CMD ["evaluate"]
