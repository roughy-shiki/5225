"""
CloudPose Locust负载测试脚本
用于模拟并发用户访问CloudPose API进行性能测试
"""

import base64
import json
import os
import random
import uuid
import sys
from locust import HttpUser, task, between
import requests

class PoseEstimationUser(HttpUser):
    """模拟用户类，用于姿态估计API负载测试"""
    
    # 用户请求间隔时间（秒）
    wait_time = between(1, 3)
    
    def on_start(self):
        """用户开始时的初始化操作"""
        print(f"用户 {self.environment.parsed_options.locustfile} 开始测试")
        
        # 加载测试图片
        self.images = self.load_test_images()
        if not self.images:
            raise Exception("❌ 没有加载到测试图片！请确保images/目录包含图片文件。")
        
        print(f"✅ 成功加载 {len(self.images)} 张测试图片")
        
        # 添加提供的测试图片
        self.add_provided_test_images()
    
    def load_test_images(self):
        """加载测试图片并转换为base64"""
        images = []
        
        # 可能的图片目录
        image_dirs = ["images", "../images", "./images"]
        image_dir = None
        
        for dir_path in image_dirs:
            if os.path.exists(dir_path):
                image_dir = dir_path
                break
        
        if not image_dir:
            print("⚠️ 未找到images目录")
            return images
        
        # 支持的图片格式
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(supported_formats):
                try:
                    filepath = os.path.join(image_dir, filename)
                    with open(filepath, "rb") as f:
                        image_data = f.read()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        images.append({
                            'filename': filename,
                            'data': base64_image
                        })
                        print(f"  📸 加载图片: {filename}")
                except Exception as e:
                    print(f"⚠️ 加载图片失败 {filename}: {e}")
        
        return images
    
    def add_provided_test_images(self):
        """添加提供的测试图片"""
        provided_images = [
            "model3-yolo1/test.jpg",
            "model3-yolo1/bus.jpg", 
            "model3-yolo1/test_with_keypoints.jpg",
            "../model3-yolo1/test.jpg",
            "../model3-yolo1/bus.jpg", 
            "../model3-yolo1/test_with_keypoints.jpg"
        ]
        
        for image_path in provided_images:
            if os.path.exists(image_path):
                try:
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        self.images.append({
                            'filename': os.path.basename(image_path),
                            'data': base64_image
                        })
                        print(f"  ✅ 添加提供的图片: {image_path}")
                except Exception as e:
                    print(f"⚠️ 加载提供图片失败 {image_path}: {e}")
    
    def generate_uuid_for_image(self, filename):
        """为图像生成UUID - 与客户端代码一致"""
        return str(uuid.uuid5(uuid.NAMESPACE_OID, filename))
    
    @task(5)  # 权重为5，这是主要的测试任务
    def pose_estimation_json(self):
        """测试姿态估计JSON API"""
        if not self.images:
            return
            
        # 随机选择一张图片
        image = random.choice(self.images)
        
        # 创建请求payload - 符合Assignment规范
        payload = {
            "id": self.generate_uuid_for_image(image['filename']),
            "image": image['data']
        }
        
        # 发送POST请求到主要端点
        with self.client.post(
            "/api/pose_estimation",
            json=payload,
            catch_response=True,
            name="pose_estimation_json"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    # 验证响应格式
                    required_fields = ['id', 'count', 'boxes', 'keypoints', 
                                     'speed_preprocess', 'speed_inference', 'speed_postprocess']
                    if all(field in result for field in required_fields):
                        response.success()
                        # 记录性能指标
                        self.environment.events.request.fire(
                            request_type="METRIC",
                            name="people_detected",
                            response_time=result.get('count', 0),
                            response_length=0
                        )
                    else:
                        response.failure(f"响应缺少必要字段")
                except json.JSONDecodeError:
                    response.failure("无效的JSON响应")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(2)  # 权重为2，较少测试图像API
    def pose_estimation_image(self):
        """测试姿态估计图像注释API"""
        if not self.images:
            return
            
        # 随机选择一张图片
        image = random.choice(self.images)
        
        # 创建请求payload
        payload = {
            "id": self.generate_uuid_for_image(image['filename']),
            "image": image['data']
        }
        
        # 发送POST请求到图像注释端点
        with self.client.post(
            "/api/pose_estimation_annotation",
            json=payload,
            catch_response=True,
            name="pose_estimation_image"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    # 验证响应格式
                    if 'id' in result and 'image' in result:
                        # 验证返回的是有效的base64图像
                        try:
                            base64.b64decode(result['image'])
                            response.success()
                        except:
                            response.failure("返回的图像数据无效")
                    else:
                        response.failure("响应缺少必要字段")
                except json.JSONDecodeError:
                    response.failure("无效的JSON响应")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)  # 权重为1，偶尔测试健康检查
    def health_check(self):
        """健康检查"""
        with self.client.get("/health", name="health_check") as response:
            if response.status_code != 200:
                response.failure(f"健康检查失败: {response.status_code}")
    
    @task(1)  # 测试兼容性端点
    def pose_detection_compatibility(self):
        """测试与原始客户端兼容的端点"""
        if not self.images:
            return
            
        # 随机选择一张图片
        image = random.choice(self.images)
        
        payload = {
            "id": self.generate_uuid_for_image(image['filename']),
            "image": image['data']
        }
        
        # 测试兼容性端点 /api/pose_detection
        with self.client.post(
            "/api/pose_detection",
            json=payload,
            catch_response=True,
            name="pose_detection_compatibility"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    required_fields = ['id', 'count', 'boxes', 'keypoints']
                    if all(field in result for field in required_fields):
                        response.success()
                    else:
                        response.failure("响应格式错误")
                except json.JSONDecodeError:
                    response.failure("无效的JSON响应")
            else:
                response.failure(f"HTTP {response.status_code}")

# Locust配置类
class WebsiteUser(PoseEstimationUser):
    """网站用户类 - 继承自PoseEstimationUser"""
    pass

# 如果直接运行此文件，提供使用说明
if __name__ == "__main__":
    print("""
CloudPose Locust负载测试脚本

使用方法：
1. 基本使用：
   locust -f locustfile.py --host=http://localhost:60001

2. 无界面模式：
   locust -f locustfile.py --host=http://localhost:60001 --headless --users=10 --spawn-rate=2 --run-time=60s

3. 指定输出：
   locust -f locustfile.py --host=http://localhost:60001 --headless --users=50 --spawn-rate=5 --run-time=300s --csv=results

参数说明：
- --host: 目标服务器地址
- --users: 模拟用户数量
- --spawn-rate: 用户生成速率（用户/秒）
- --run-time: 测试运行时间
- --csv: 结果输出到CSV文件
""")