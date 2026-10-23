FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN pip install --no-cache-dir uv

WORKDIR /app

# dependencies first, so this layer is cached until pyproject/uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# only what the app needs at run time (see .dockerignore: it is a whitelist, so .env can never get in)
COPY app ./app
COPY web ./web
COPY data/docs ./data/docs
COPY data/raw/known_exploited_vulnerabilities.csv ./data/raw/known_exploited_vulnerabilities.csv
COPY data/index/naive_800_150.npz data/index/naive_800_150.json ./data/index/

# the key is NOT here: set OPENAI_API_KEY as a secret on the host. The retrieval settings are the code
# defaults (hybrid, naive:800:150, top_k 10, floor 0.55), which is exactly the index copied above.
ENV PATH="/app/.venv/bin:$PATH" OPENAI_BASE_URL=https://llm.nat-d.uk/v1 TRUST_PROXY=1 PORT=8000

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
