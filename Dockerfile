 # CloudPose Docker镜像
# 基于Python 3.9 slim镜像构建
FROM python:3.9-slim

# 设置维护者信息
LABEL maintainer="CloudPose Team"
LABEL description="CloudPose姿态估计Web服务 - Model 3 YOLO"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
# 这些包是OpenCV和其他库运行所必需的
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制Python依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制Model 3 YOLO相关文件
COPY model3-yolo1/ ./model3-yolo1/

# 复制主要应用代码
COPY main.py .

# 复制其他必要文件（如果存在）
COPY client/ ./client/ 2>/dev/null || true

# 创建非root用户（安全最佳实践）
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# 设置环境变量
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# 暴露服务端口
EXPOSE 60001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:60001/health || exit 1

# 启动命令
CMD ["python", "main.py"]
