FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Ensure slightly faster startup times.
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-install-project --no-dev

COPY . .

RUN cp example.config.yaml config.yaml

RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "--port", "8000"]
