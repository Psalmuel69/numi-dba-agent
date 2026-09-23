# Single image, multiple entrypoints — the `SERVICE` build/run arg picks
# which FastAPI app to serve. Each of gateway/execution/agent/channels runs
# this same image with a different SERVICE value and port, per the
# docker-compose.yml in this repo.

FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg

RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/microsoft-prod.list

RUN apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
    msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY migrations ./migrations
COPY alembic.ini ./

RUN pip install --no-cache-dir -e ".[db-drivers]"

ARG SERVICE=gateway
ENV SERVICE=${SERVICE}

# Non-root runtime user.
RUN useradd --create-home --uid 10001 numi
USER numi

EXPOSE 8000 8001 8002 8003

CMD ["sh", "-c", "\
    case \"$SERVICE\" in \
      gateway)   exec uvicorn numi.gateway.api.app:app --host 0.0.0.0 --port ${GATEWAY_PORT:-8001} ;; \
      execution) exec uvicorn numi.execution.api.app:app --host 0.0.0.0 --port ${EXECUTION_PORT:-8002} ;; \
      agent)     exec uvicorn numi.agent.api.app:app --host 0.0.0.0 --port ${AGENT_PORT:-8000} ;; \
      channels)  exec uvicorn numi.channels.api.app:app --host 0.0.0.0 --port ${CHANNELS_PORT:-8003} ;; \
      *) echo \"Unknown SERVICE '$SERVICE'\"; exit 1 ;; \
    esac \
"]
