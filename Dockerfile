# Bagent RAG —— 运行时镜像
# 说明：模型权重不打进镜像（1GB+ 且离线环境），运行期用 compose 挂载 ./models。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖：lxml/psycopg 等均有 manylinux 轮子，一般无需额外 apt；保留 curl 供健康检查
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# 先装 CPU 版 torch（避免拉取 CUDA 大包），再装其余依赖
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
