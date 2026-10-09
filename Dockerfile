FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai

# ============================================================
# 系统运行依赖
# ============================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ca-certificates \
        tzdata \
        procps \
        && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Python依赖
# ============================================================
COPY requirements-sherpa.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
        -r /tmp/requirements.txt

# ============================================================
# 创建挂载目录
# ============================================================
WORKDIR /app

RUN mkdir -p /models /logs

# ============================================================
# 不 COPY 业务代码
# 不 COPY 模型
# 不定义 CMD
# ============================================================
