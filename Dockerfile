FROM python:3.13
WORKDIR .

COPY . /

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv sync --locked

CMD ["uv", "run", "fastapi", "run", "--port", "8000"]

