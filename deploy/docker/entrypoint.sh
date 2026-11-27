#!/bin/sh
set -euf
UV="python -m uvicorn --factory --host 0.0.0.0 --no-server-header --proxy-headers --forwarded-allow-ips=* --timeout-graceful-shutdown ${GRACEFUL_SECONDS:-25}"
case "${SERVICE:-}" in
  channels)     exec $UV --port 8000 --workers "${WEB_CONCURRENCY:-2}" agentic.channels.app:app_factory ;;
  gateway)      exec $UV --port 8001 --workers "${WEB_CONCURRENCY:-2}" agentic.gateway.app:app_factory ;;
  orchestrator) exec $UV --port 8002 --workers "${WEB_CONCURRENCY:-2}" agentic.orchestration.app:app_factory ;;
  tools)        exec $UV --port 8003 --workers "${WEB_CONCURRENCY:-2}" agentic.tools.server:app_factory ;;
  internal-api) exec $UV --port 8004 --workers "${WEB_CONCURRENCY:-1}" agentic.tools.internal_api:app_factory ;;
  worker)       exec python -m agentic.backbone.worker ;;
  migrate)      exec agentic-migrate ;;
  ingest)       exec agentic-ingest "${INGEST_PATH:-/app/data/sample_docs}" --tenant "${INGEST_TENANT:-default}" ${INGEST_ARGS:-} ;;
  evals)        exec agentic-evals "$@" ;;
  *)            exec "$@" ;;
esac
