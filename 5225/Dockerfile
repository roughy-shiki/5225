# CloudPose Docker镜像
# 基于Python 3.9 slim镜像构建
FROM python:3.9-slim

# 设置元数据
LABEL maintainer="CloudPose Team" \
      description="CloudPose姿态估计Web服务 - Model 3 YOLO"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgcc-s1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用文件
COPY model3-yolo1/ ./model3-yolo1/
COPY main.py .
COPY client/ ./client/

# 创建非root用户
RUN groupadd -r app && \
    useradd -r -g app --home-dir /app --shell /bin/bash app && \
    chown -R app:app /app
USER app

# 环境变量
ENV PYTHONPATH="${PYTHONPATH}:/app" \
    PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 60001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:60001/health >/dev/null || exit 1

# 启动命令
CMD ["python", "main.py"]