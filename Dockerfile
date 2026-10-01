# Receipt verifier service image.
#
# Tesseract is installed as a system package so the OCR stage always has language data;
# the `tesserocr` wheel ships the engine itself, and the `ocr` extra keeps it optional for
# plain-Python installs.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OCR_TESSDATA=/usr/share/tesseract-ocr/5/tessdata \
    OCR_LANGUAGES=spa+eng

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --no-cache-dir ".[ocr]"

# Non-root, no shell, no home directory: the service only reads request bodies.
RUN useradd --system --uid 10001 --no-create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
