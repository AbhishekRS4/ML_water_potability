# Build from repo root:
#   docker build -f backend_app/Dockerfile -t ml-water-potability-backend .

FROM python:3.10-slim

WORKDIR /app

# install linux package dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# copy pyproject.toml and uv.lock from repo root for reproducible dependency installation
COPY pyproject.toml uv.lock ./

# install backend dependencies using uv (into the system Python, no venv)
RUN uv sync --group dev-backend --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

COPY backend_app/model_for_production/ /data/model_for_production
COPY backend_app/*.py .

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
