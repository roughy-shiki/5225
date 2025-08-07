# CloudPose项目阿里云实施指南

## 项目概述
本指南将帮助您使用阿里云完成CloudPose姿态检测Web服务项目，包括容器化部署、Kubernetes集群管理和负载测试。

## 第一部分：Web服务开发 [20分]

### 1.1 创建FastAPI Web服务

首先，基于您提供的pose-detection.py，我们需要创建FastAPI服务：

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import tensorflow as tf
import uuid
import time
import threading
import logging
from typing import List, Dict
import os
import tempfile

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CloudPose API", version="1.0.0")

# 请求模型
class ImageRequest(BaseModel):
    id: str
    image: str  # base64编码的图像

class KeyPoint(BaseModel):
    x: float
    y: float
    p: float

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    probability: float

class PoseResponse(BaseModel):
    id: str
    count: int
    boxes: List[BoundingBox]
    keypoints: List[List[List[float]]]
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float

class ImageResponse(BaseModel):
    id: str
    image: str  # base64编码的注释图像

# 全局模型加载
MODEL_PATH = 'movenet-full-256.tflite'
interpreter = None

def load_model():
    global interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def decode_base64_image(base64_string: str) -> np.ndarray:
    """解码base64图像"""
    try:
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 image")

def encode_image_to_base64(image: np.ndarray) -> str:
    """编码图像为base64"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise HTTPException(status_code=500, detail="Error encoding image")

def detect_pose(img: np.ndarray) -> tuple:
    """姿态检测核心函数"""
    global interpreter
    
    start_preprocess = time.time()
    
    # 获取输入和输出详情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 调整图像大小和归一化
    img_height, img_width, _ = img.shape
    input_shape = input_details[0]['shape'][1:3]
    resized_image = cv2.resize(img, input_shape)
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0
    
    preprocess_time = time.time() - start_preprocess
    
    # 推理
    start_inference = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    inference_time = time.time() - start_inference
    
    # 后处理
    start_postprocess = time.time()
    keypoints_output = interpreter.get_tensor(output_details[0]['index'])
    keypoints = keypoints_output[0]  # Shape: (17, 3)
    
    # 创建边界框（简化版本，实际应该基于关键点计算）
    boxes = []
    if len(keypoints) > 0:
        # 计算边界框
        valid_points = keypoints[keypoints[:, 2] > 0.3]  # 置信度阈值
        if len(valid_points) > 0:
            min_x = np.min(valid_points[:, 1]) * img_width
            max_x = np.max(valid_points[:, 1]) * img_width
            min_y = np.min(valid_points[:, 0]) * img_height
            max_y = np.max(valid_points[:, 0]) * img_height
            
            boxes.append({
                "x": float(min_x),
                "y": float(min_y),
                "width": float(max_x - min_x),
                "height": float(max_y - min_y),
                "probability": float(np.mean(valid_points[:, 2]))
            })
    
    postprocess_time = time.time() - start_postprocess
    
    return keypoints, boxes, preprocess_time * 1000, inference_time * 1000, postprocess_time * 1000

def annotate_image(img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """在图像上注释关键点"""
    annotated_image = img.copy()
    img_height, img_width, _ = img.shape
    
    # 关键点连接
    connections = [
        (5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (6, 8), (7, 9), (8, 10),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    
    # 绘制关键点
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > 0.3:  # 置信度阈值
            x_coord = int(x * img_width)
            y_coord = int(y * img_height)
            cv2.circle(annotated_image, (x_coord, y_coord), 5, (0, 255, 0), -1)
    
    # 绘制连接线
    for start, end in connections:
        if start < len(keypoints) and end < len(keypoints):
            y1, x1, c1 = keypoints[start]
            y2, x2, c2 = keypoints[end]
            if c1 > 0.3 and c2 > 0.3:
                x1_coord = int(x1 * img_width)
                y1_coord = int(y1 * img_height)
                x2_coord = int(x2 * img_width)
                y2_coord = int(y2 * img_height)
                cv2.line(annotated_image, (x1_coord, y1_coord), (x2_coord, y2_coord), (0, 0, 255), 2)
    
    return annotated_image

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    load_model()

@app.get("/")
async def root():
    return {"message": "CloudPose API is running"}

@app.post("/api/pose_detection", response_model=PoseResponse)
async def pose_detection_json(request: ImageRequest):
    """姿态检测JSON API"""
    try:
        # 解码图像
        img = decode_base64_image(request.image)
        
        # 姿态检测
        keypoints, boxes, preprocess_time, inference_time, postprocess_time = detect_pose(img)
        
        # 格式化关键点
        formatted_keypoints = []
        if len(keypoints) > 0:
            kp_list = []
            for kp in keypoints:
                kp_list.append([float(kp[1]), float(kp[0]), float(kp[2])])  # x, y, confidence
            formatted_keypoints.append(kp_list)
        
        response = PoseResponse(
            id=request.id,
            count=1 if len(keypoints) > 0 else 0,
            boxes=boxes,
            keypoints=formatted_keypoints,
            speed_preprocess=preprocess_time,
            speed_inference=inference_time,
            speed_postprocess=postprocess_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in pose detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pose_detection_annotation", response_model=ImageResponse)
async def pose_detection_image(request: ImageRequest):
    """姿态检测图像API"""
    try:
        # 解码图像
        img = decode_base64_image(request.image)
        
        # 姿态检测
        keypoints, _, _, _, _ = detect_pose(img)
        
        # 注释图像
        annotated_img = annotate_image(img, keypoints)
        
        # 编码为base64
        annotated_base64 = encode_image_to_base64(annotated_img)
        
        response = ImageResponse(
            id=request.id,
            image=annotated_base64
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in pose detection annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=60000)
```

### 1.2 更新requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
tensorflow==2.13.0
opencv-python==4.8.1.78
numpy==1.24.3
pydantic==2.5.0
pillow==10.1.0
```

## 第二部分：Dockerfile [10分]

### 2.1 创建Dockerfile

```dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模型
COPY app.py .
COPY movenet-full-256.tflite .

# 暴露端口
EXPOSE 60000

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:60000/ || exit 1

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "60000", "--workers", "4"]
```

### 2.2 构建Docker镜像

```bash
# 构建镜像
docker build -t cloudpose:latest .

# 测试本地运行
docker run -p 60000:60000 cloudpose:latest
```

## 第三部分：阿里云Kubernetes集群 [10分]

### 3.1 创建阿里云ECS实例

1. **登录阿里云控制台**
   - 进入ECS实例页面
   - 选择香港地域

2. **创建2台ECS实例**
   
   **Master节点配置：**
   - 实例规格：ecs.c6.large (2vCPU, 4GB内存)
   - 操作系统：Ubuntu 20.04 LTS
   - 系统盘：40GB SSD
   - 网络：VPC网络，分配公网IP
   - 安全组：开放22, 6443, 2379-2380, 10250-10252端口

   **Worker节点配置：**
   - 实例规格：ecs.c6.large (2vCPU, 4GB内存)
   - 操作系统：Ubuntu 20.04 LTS
   - 系统盘：40GB SSD
   - 网络：VPC网络
   - 安全组：开放22, 10250, 30000-32767端口

### 3.2 安装Docker和Kubernetes

**在所有节点上执行：**

```bash
# 更新系统
sudo apt-get update
sudo apt-get upgrade -y

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 启动并启用Docker
sudo systemctl start docker
sudo systemctl enable docker

# 安装Kubernetes
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl

curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-archive-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt-get update
sudo apt-get install -y kubelet=1.28.2-00 kubeadm=1.28.2-00 kubectl=1.28.2-00
sudo apt-mark hold kubelet kubeadm kubectl

# 禁用swap
sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

# 配置内核参数
cat <<EOF | sudo tee /etc/modules-load.d/containerd.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

cat <<EOF | sudo tee /etc/sysctl.d/99-kubernetes-cri.conf
net.bridge.bridge-nf-call-iptables  = 1
net.ipv4.ip_forward                 = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF

sudo sysctl --system
```

### 3.3 初始化Kubernetes集群

**在Master节点上：**

```bash
# 初始化集群
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=<MASTER_PRIVATE_IP>

# 配置kubectl
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# 安装Flannel网络插件
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

# 获取join命令
kubeadm token create --print-join-command
```

**在Worker节点上：**

```bash
# 使用上面获取的join命令
sudo kubeadm join <MASTER_IP>:6443 --token <TOKEN> --discovery-token-ca-cert-hash <HASH>
```

**验证集群：**

```bash
kubectl get nodes
kubectl get pods -A
```

## 第四部分：Kubernetes服务 [10分]

### 4.1 创建部署配置

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cloudpose-deployment
  labels:
    app: cloudpose
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cloudpose
  template:
    metadata:
      labels:
        app: cloudpose
    spec:
      containers:
      - name: cloudpose
        image: cloudpose:latest
        imagePullPolicy: Never  # 使用本地镜像
        ports:
        - containerPort: 60000
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.5"
          limits:
            memory: "512Mi"
            cpu: "0.5"
        livenessProbe:
          httpGet:
            path: /
            port: 60000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 60000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 4.2 创建服务配置

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cloudpose-service
  labels:
    app: cloudpose
spec:
  type: NodePort
  ports:
  - port: 60000
    targetPort: 60000
    nodePort: 30000
    protocol: TCP
  selector:
    app: cloudpose
```

### 4.3 部署到Kubernetes

```bash
# 在Master节点上构建并加载镜像
docker build -t cloudpose:latest .

# 将镜像保存并传输到Worker节点
docker save cloudpose:latest | gzip > cloudpose.tar.gz
scp cloudpose.tar.gz user@worker-node:/tmp/

# 在Worker节点上加载镜像
ssh user@worker-node "docker load < /tmp/cloudpose.tar.gz"

# 部署应用
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 验证部署
kubectl get deployments
kubectl get pods
kubectl get services

# 查看Pod详情
kubectl describe pod <pod-name>
```

### 4.4 配置阿里云安全组

在阿里云控制台配置安全组规则：

```
入方向规则：
- 协议类型：TCP，端口范围：30000/30000，授权对象：0.0.0.0/0
- 协议类型：TCP，端口范围：6443/6443，授权对象：VPC内网段
- 协议类型：TCP，端口范围：22/22，授权对象：0.0.0.0/0
```

## 第五部分：Locust负载生成 [10分]

### 5.1 创建Locust脚本

```python
# locust_client.py
from locust import HttpUser, task, between
import base64
import uuid
import json
import os
import glob
import random

class CloudPoseUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """在测试开始时加载图像"""
        self.images = self.load_images()
        if not self.images:
            raise Exception("No images found to test with")
    
    def load_images(self):
        """加载测试图像"""
        images = []
        # 假设images目录下有测试图像
        image_files = glob.glob("images/*.jpg") + glob.glob("images/*.png")
        
        for image_file in image_files:
            try:
                with open(image_file, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    images.append(image_data)
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")
                
        return images
    
    @task(3)
    def pose_detection_json(self):
        """测试JSON API"""
        if not self.images:
            return
            
        image_data = random.choice(self.images)
        img_id = str(uuid.uuid4())
        
        payload = {
            "id": img_id,
            "image": image_data
        }
        
        headers = {'Content-Type': 'application/json'}
        
        with self.client.post(
            "/api/pose_detection",
            json=payload,
            headers=headers,
            catch_response=True,
            name="pose_detection_json"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("id") == img_id:
                        response.success()
                    else:
                        response.failure("ID mismatch in response")
                except Exception as e:
                    response.failure(f"Invalid JSON response: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def pose_detection_image(self):
        """测试图像API"""
        if not self.images:
            return
            
        image_data = random.choice(self.images)
        img_id = str(uuid.uuid4())
        
        payload = {
            "id": img_id,
            "image": image_data
        }
        
        headers = {'Content-Type': 'application/json'}
        
        with self.client.post(
            "/api/pose_detection_annotation",
            json=payload,
            headers=headers,
            catch_response=True,
            name="pose_detection_image"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("id") == img_id and result.get("image"):
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except Exception as e:
                    response.failure(f"Invalid JSON response: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")

# 运行Locust的命令
# locust -f locust_client.py --host=http://<MASTER_PUBLIC_IP>:30000
```

### 5.2 创建自动化实验脚本

```python
# experiment_runner.py
import subprocess
import time
import json
import csv
import os
from typing import List, Dict

class ExperimentRunner:
    def __init__(self, host_url: str, image_dir: str):
        self.host_url = host_url
        self.image_dir = image_dir
        self.results = []
    
    def scale_deployment(self, replicas: int):
        """扩展部署的Pod数量"""
        cmd = f"kubectl scale deployment cloudpose-deployment --replicas={replicas}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to scale deployment: {result.stderr}")
        
        # 等待Pod就绪
        print(f"Scaling to {replicas} replicas...")
        time.sleep(30)
        
        # 验证Pod状态
        cmd = "kubectl get pods -l app=cloudpose"
        subprocess.run(cmd, shell=True)
    
    def run_locust_test(self, users: int, spawn_rate: int = 1, duration: int = 300) -> Dict:
        """运行Locust测试"""
        print(f"Running test with {users} users, spawn rate {spawn_rate}")
        
        cmd = [
            "locust",
            "-f", "locust_client.py",
            "--host", self.host_url,
            "--users", str(users),
            "--spawn-rate", str(spawn_rate),
            "--run-time", f"{duration}s",
            "--headless",
            "--csv", f"results_{users}users"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Locust test failed: {result.stderr}")
            return None
        
        # 解析结果
        return self.parse_locust_results(f"results_{users}users_stats.csv")
    
    def parse_locust_results(self, csv_file: str) -> Dict:
        """解析Locust CSV结果"""
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                stats = list(reader)
                
                # 计算总体统计
                total_requests = sum(int(row['Request Count']) for row in stats if row['Name'] != 'Aggregated')
                total_failures = sum(int(row['Failure Count']) for row in stats if row['Name'] != 'Aggregated')
                avg_response_time = sum(float(row['Average Response Time']) for row in stats if row['Name'] != 'Aggregated') / len([row for row in stats if row['Name'] != 'Aggregated'])
                
                success_rate = ((total_requests - total_failures) / total_requests * 100) if total_requests > 0 else 0
                
                return {
                    'total_requests': total_requests,
                    'total_failures': total_failures,
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time
                }
        except Exception as e:
            print(f"Error parsing results: {e}")
            return None
    
    def find_max_users(self, pod_count: int) -> Dict:
        """找到最大并发用户数"""
        print(f"\n=== Testing with {pod_count} pods ===")
        self.scale_deployment(pod_count)
        
        # 二分查找最大用户数
        low, high = 1, 200
        max_successful_users = 0
        best_result = None
        
        while low <= high:
            mid = (low + high) // 2
            print(f"Testing {mid} users...")
            
            result = self.run_locust_test(mid, spawn_rate=1, duration=120)
            
            if result and result['success_rate'] >= 100:
                max_successful_users = mid
                best_result = result
                low = mid + 1
                print(f"✓ {mid} users successful (success rate: {result['success_rate']:.1f}%)")
            else:
                high = mid - 1
                if result:
                    print(f"✗ {mid} users failed (success rate: {result['success_rate']:.1f}%)")
                else:
                    print(f"✗ {mid} users failed (test error)")
        
        # 确认测试
        if max_successful_users > 0:
            print(f"Confirming {max_successful_users} users...")
            final_result = self.run_locust_test(max_successful_users, spawn_rate=1, duration=300)
            if final_result and final_result['success_rate'] >= 100:
                best_result = final_result
        
        return {
            'pod_count': pod_count,
            'max_users': max_successful_users,
            'avg_response_time': best_result['avg_response_time'] if best_result else 0,
            'success_rate': best_result['success_rate'] if best_result else 0
        }
    
    def run_experiments(self, pod_counts: List[int]):
        """运行完整实验"""
        print("Starting CloudPose Load Testing Experiments")
        print("=" * 50)
        
        for pod_count in pod_counts:
            result = self.find_max_users(pod_count)
            self.results.append(result)
            print(f"Pod Count: {pod_count}, Max Users: {result['max_users']}, Avg Response Time: {result['avg_response_time']:.2f}ms")
        
        # 保存结果
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """保存实验结果"""
        with open('experiment_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        with open('experiment_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['pod_count', 'max_users', 'avg_response_time', 'success_rate'])
            writer.writeheader()
            writer.writerows(self.results)
    
    def generate_report(self):
        """生成实验报告"""
        print("\n" + "=" * 50)
        print("EXPERIMENT RESULTS")
        print("=" * 50)
        print(f"{'Pods':<6} {'Max Users':<12} {'Avg Response Time (ms)':<22} {'Success Rate (%)'}")
        print("-" * 50)
        
        for result in self.results:
            print(f"{result['pod_count']:<6} {result['max_users']:<12} {result['avg_response_time']:<22.2f} {result['success_rate']:<15.1f}")

if __name__ == "__main__":
    # 配置实验参数
    MASTER_IP = "<YOUR_MASTER_PUBLIC_IP>"  # 替换为实际IP
    HOST_URL = f"http://{MASTER_IP}:30000"
    IMAGE_DIR = "images"
    
    # 创建实验运行器
    runner = ExperimentRunner(HOST_URL, IMAGE_DIR)
    
    # 运行实验：1, 2, 4个Pod
    pod_counts = [1, 2, 4]
    runner.run_experiments(pod_counts)
```

## 第六部分：实验和报告 [40分]

### 6.1 准备测试环境

```bash
# 安装Locust
pip install locust

# 创建测试图像目录
mkdir images
# 将测试图像放入images目录

# 运行实验
python experiment_runner.py
```

### 6.2 实验数据收集

根据Assignment要求，需要收集以下数据：

1. **本地测试**（在Master节点上运行Locust）
2. **远程测试**（在阿里云另一个区域的ECS上运行Locust）

创建结果表格：

| 位置 | Pod数量 | 最大用户数 | 平均响应时间(ms) |
|------|---------|------------|------------------|
| Master节点 | 1 | ? | ? |
| Master节点 | 2 | ? | ? |
| Master节点 | 4 | ? | ? |
| 远程节点 | 1 | ? | ? |
| 远程节点 | 2 | ? | ? |
| 远程节点 | 4 | ? | ? |

### 6.3 报告撰写指导

**报告结构（1500字以内）：**

1. **实验结果分析**（1000字）
   - 性能指标分析
   - 扩展性评估
   - 网络延迟影响
   - 资源利用率分析

2. **分布式系统挑战**（500字）
   选择三个挑战，例如：
   - **负载均衡**：Kubernetes Service如何分发请求
   - **容错性**：Pod失败时的自动恢复机制
   - **一致性**：多个Pod间的状态同步

### 6.4 性能优化建议

```yaml
# 优化后的deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cloudpose-deployment
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  template:
    spec:
      containers:
      - name: cloudpose
        image: cloudpose:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.5"
          limits:
            memory: "512Mi"
            cpu: "0.5"
        env:
        - name: WORKERS
          value: "4"
        readinessProbe:
          httpGet:
            path: /health
            port: 60000
          initialDelaySeconds: 10
          periodSeconds: 5
```

## 第七部分：视频录制指南

### 7.1 视频内容清单（8分钟）

1. **Web服务演示**（2分钟）
   - 展示app.py代码结构
   - 解释API端点和JSON处理

2. **Dockerfile说明**（1分钟）
   - 展示Dockerfile内容
   - 解释优化策略

3. **Kubernetes集群**（4分钟）
   - 展示阿里云ECS实例
   - 执行`kubectl get nodes -o wide`
   - 展示YAML配置文件
   - 演示镜像构建和部署过程
   - 展示安全组配置
   - 验证4个Pod运行状态
   - 展示负载均衡日志

4. **Locust脚本**（1分钟）
   - 展示locust_client.py
   - 运行测试演示

### 7.2 提交文件清单

```
submission.zip
├── app.py
├── Dockerfile
├── deployment.yaml
├── service.yaml
├── locust_client.py
├── experiment_runner.py
├── requirements.txt
└── readme.md
```

**readme.md示例：**
```markdown
# CloudPose Project

## Video URL
https://www.youtube.com/watch?v=YOUR_VIDEO_ID

## Service Endpoints
- JSON API: http://47.XXX.XXX.XXX:30000/api/pose_detection
- Image API: http://47.XXX.XXX.XXX:30000/api/pose_detection_annotation

## Notes
- Model used: MoveNet (based on student ID)
- Kubernetes version: 1.28.2
- Network plugin: Flannel
```

## 故障排除

### 常见问题解决

1. **Pod无法启动**
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

2. **服务无法访问**
```bash
kubectl get svc
kubectl describe svc cloudpose-service
```

3. **网络问题**
```bash
# 检查安全组配置
# 确保30000端口开放
```

4. **资源不足**
```bash
kubectl top nodes
kubectl top pods
```

这个详细指南涵盖了使用阿里云完成CloudPose项目的所有步骤。请根据实际情况调整IP地址、域名等配置信息。
