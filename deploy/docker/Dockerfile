# syntax=docker/dockerfile:1.7
# One image, many services: pick with SERVICE=<channels|gateway|orchestrator|tools|internal-api|worker|migrate|ingest|evals>
FROM python:3.12-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

FROM python:3.12-slim AS runtime
ARG VERSION=dev
LABEL org.opencontainers.image.title="agentic-platform" org.opencontainers.image.version="${VERSION}"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app HOME=/tmp
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels agentic-platform \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY evals ./evals
COPY data ./data
COPY deploy/docker/entrypoint.sh /entrypoint.sh
RUN chmod 0555 /entrypoint.sh
USER 10001:10001
EXPOSE 8000 8001 8002 8003 8004 9100
ENTRYPOINT ["/entrypoint.sh"]
