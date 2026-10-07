FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image a few GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Bake the embedding + reranker models into the image so the first start
# doesn't depend on a HuggingFace download. Done before copying the source
# so code changes don't invalidate this layer.
ENV HF_HOME=/models
RUN pip install --no-cache-dir "sentence-transformers>=3.0.0" \
    && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Models are baked in above; don't contact the HF Hub at runtime.
ENV HF_HUB_OFFLINE=1

COPY pyproject.toml README.md ./
COPY gnosis ./gnosis
RUN pip install --no-cache-dir .

COPY examples ./examples
COPY docker/entrypoint.sh ./docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["sh", "/app/docker/entrypoint.sh"]
CMD ["gnosis", "serve", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
