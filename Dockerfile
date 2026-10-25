# ---------- 1. 前端构建 ----------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------- 2. 运行时 ----------
FROM python:3.11-slim

LABEL org.opencontainers.image.title="On-Call Agent"
LABEL org.opencontainers.image.description="可评测的 AIOps 诊断 Agent"

ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_ENDPOINT=https://hf-mirror.com \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 依赖层：先装依赖再加代码，利用缓存
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv pip install --system -e ".[dev]" \
    && rm -rf /root/.cache

COPY . .
COPY --from=web /web/dist ./web/dist

EXPOSE 9900 8003 8004 8005

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9900/health', timeout=3)"

ENTRYPOINT ["/docker-entrypoint.sh"]
