FROM python:3.11-slim

LABEL org.opencontainers.image.title="On-Call Agent"
LABEL org.opencontainers.image.description="企业级 AIOps 智能运维助手"

# 国内镜像加速
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_ENDPOINT=https://hf-mirror.com
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY pyproject.toml uv.lock ./

# 安装 Python 依赖
RUN pip install --no-cache-dir uv \
    && uv pip install --system -e . \
    && rm -rf /root/.cache

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 9900 8003 8004 8005

# 启动脚本
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
