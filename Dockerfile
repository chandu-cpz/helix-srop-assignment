FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120

# Proxy args: forwarded from the build host when present; NOT baked into the
# image ENV so the submission works without any proxy configuration.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG ALL_PROXY
ARG all_proxy

# Install dependencies from the lockfile first (cacheable layer).
# --no-install-project skips installing the local package so this layer
# is only invalidated when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the application source and install the project itself.
COPY alembic.ini /app/
COPY app /app/app
COPY docs /app/docs
COPY alembic /app/alembic

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]