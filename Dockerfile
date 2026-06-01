FROM python:3.13
WORKDIR /

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /src

RUN uv sync --locked

CMD ["uv", "run", "fastapi", "run"]

