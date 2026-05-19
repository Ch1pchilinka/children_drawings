FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN mkdir -p .dvc data

COPY pyproject.toml uv.lock README.md ./
COPY children_drawings ./children_drawings
COPY conf ./conf
COPY scripts ./scripts
COPY data/*.dvc ./data/
COPY data/.gitignore ./data/.gitignore
COPY .dvc/config ./.dvc/config
COPY .dvc/.gitignore ./.dvc/.gitignore
COPY .dvcignore ./.dvcignore

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["children-drawings-infer"]
