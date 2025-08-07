#!/usr/bin/env python3
"""
CloudPose测试客户端
用于测试CloudPose Web服务的功能和性能
"""

import requests
import base64
import json
import uuid
import time
import os
import sys
from typing import Optional, Dict, Any

class CloudPoseClient:
    """CloudPose API客户端 - 用于测试服务功能"""
    
    def __init__(self, base_url: str = "http://localhost:60001"):
        """
        初始化客户端
        
        Args:
            base_url: CloudPose服务的基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CloudPose-TestClient/1.0'
        })
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        将图像文件编码为base64字符串
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            base64编码的图像字符串，失败时返回None
        """
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode('utf-8')
                return base64_encoded
        except Exception as e:
            print(f"❌ 图像编码失败 {image_path}: {e}")
            return None
    
    def generate_uuid_for_image(self, image_path: str) -> str:
        """
        为图像生成UUID - 与原始客户端保持一致
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            UUID字符串
        """
        return str(uuid.uuid5(uuid.NAMESPACE_OID, image_path))
    
    def test_health_check(self) -> bool:
        """
        测试服务健康状态
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            print("🔍 检查服务健康状态...")
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 服务健康")
                print(f"   状态: {result.get('status', 'unknown')}")
                print(f"   模型状态: {result.get('model_status', 'unknown')}")
                print(f"   模型类型: {result.get('model', 'unknown')}")
                return True
            else:
                print(f"❌ 健康检查失败: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 健康检查网络错误: {e}")
            return False
    
    def test_pose_estimation_json(self, image_path: str) -> Optional[Dict[Any, Any]]:
        """
        测试JSON姿态估计API
        
        Args:
            image_path: 测试图像路径
            
        Returns:
            API响应结果或None
        """
        print(f"\n🧪 测试JSON API - 图像: {os.path.basename(image_path)}")
        
        # 编码图像
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return None
        
        # 准备请求
        payload = {
            "id": self.generate_uuid_for_image(image_path),
            "image": base64_image
        }
        
        try:
            # 发送请求
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/pose_estimation",
                json=payload,
                timeout=60
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 请求成功 - 响应时间: {elapsed_time:.2f}s")
                print(f"   请求ID: {result.get('id', 'N/A')}")
                print(f"   检测到人数: {result.get('count', 0)}")
                print(f"   处理时间详情:")
                print(f"     预处理: {result.get('speed_preprocess', 0):.1f}ms")
                print(f"     推理: {result.get('speed_inference', 0):.1f}ms")
                print(f"     后处理: {result.get('speed_postprocess', 0):.1f}ms")
                
                # 显示检测框信息
                boxes = result.get('boxes', [])
                if boxes:
                    print(f"   检测框信息:")
                    for i, box in enumerate(boxes[:3]):  # 只显示前3个
                        print(f"     #{i+1}: 位置({box.get('x', 0)}, {box.get('y', 0)}) "
                              f"大小{box.get('width', 0)}x{box.get('height', 0)} "
                              f"置信度{box.get('probability', 0):.2f}")
                
                return result
            else:
                print(f"❌ 请求失败: HTTP {response.status_code}")
                print(f"   错误信息: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ 请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求异常: {e}")
            return None
    
    def test_pose_estimation_image(self, image_path: str, save_path: Optional[str] = None) -> Optional[Dict[Any, Any]]:
        """
        测试图像注释API
        
        Args:
            image_path: 输入图像路径
            save_path: 保存注释图像的路径
            
        Returns:
            API响应结果或None
        """
        print(f"\n🎨 测试图像注释API - 图像: {os.path.basename(image_path)}")
        
        # 编码图像
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return None
        
        # 准备请求
        payload = {
            "id": self.generate_uuid_for_image(image_path),
            "image": base64_image
        }
        
        try:
            # 发送请求
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/pose_estimation_annotation",
                json=payload,
                timeout=60
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 图像注释成功 - 响应时间: {elapsed_time:.2f}s")
                print(f"   请求ID: {result.get('id', 'N/A')}")
                
                # 保存注释后的图像
                if save_path and 'image' in result:
                    try:
                        annotated_image_data = base64.b64decode(result['image'])
                        with open(save_path, 'wb') as f:
                            f.write(annotated_image_data)
                        print(f"   ✅ 注释图像已保存: {save_path}")
                    except Exception as e:
                        print(f"   ❌ 保存图像失败: {e}")
                
                return result
            else:
                print(f"❌ 图像注释失败: HTTP {response.status_code}")
                print(f"   错误信息: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ 请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求异常: {e}")
            return None
    
    def test_compatibility_endpoint(self, image_path: str) -> Optional[Dict[Any, Any]]:
        """
        测试兼容性端点 /api/pose_detection
        
        Args:
            image_path: 测试图像路径
            
        Returns:
            API响应结果或None
        """
        print(f"\n🔄 测试兼容性端点 - 图像: {os.path.basename(image_path)}")
        
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return None
        
        payload = {
            "id": self.generate_uuid_for_image(image_path),
            "image": base64_image
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/pose_detection",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 兼容性端点工作正常")
                print(f"   检测到人数: {result.get('count', 0)}")
                return result
            else:
                print(f"❌ 兼容性端点失败: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ 兼容性测试异常: {e}")
            return None

def find_test_images() -> list:
    """寻找可用的测试图像"""
    test_images = []
    
    # 可能的图像位置
    image_paths = [
        "model3-yolol/test.jpg",
        "model3-yolol/bus.jpg",
        "model3-yolol/test_with_keypoints.jpg",
        "images/test.jpg",
        "images/bus.jpg"
    ]
    
    for path in image_paths:
        if os.path.exists(path):
            test_images.append(path)
    
    return test_images

def main():
    """主测试函数"""
    print("🚀 CloudPose客户端测试工具")
    print("=" * 50)
    
    # 解析命令行参数
    service_url = "http://localhost:60001"
    if len(sys.argv) > 1:
        service_url = sys.argv[1]
    
    print(f"📡 目标服务: {service_url}")
    
    # 创建客户端
    client = CloudPoseClient(service_url)
    
    # 1. 健康检查
    print("\n📋 步骤1: 服务健康检查")
    if not client.test_health_check():
        print("❌ 服务不可用，请检查服务是否正在运行")
        print("💡 提示: 运行 'python main.py' 启动服务")
        return
    
    # 2. 寻找测试图像
    print("\n📋 步骤2: 查找测试图像")
    test_images = find_test_images()
    if not test_images:
        print("❌ 未找到测试图像")
        print("💡 请确保以下文件存在:")
        print("   - model3-yolol/test.jpg")
        print("   - model3-yolol/bus.jpg")
        return
    
    print(f"✅ 找到 {len(test_images)} 张测试图像")
    for img in test_images:
        print(f"   📸 {img}")
    
    # 3. 测试每张图像
    print(f"\n📋 步骤3: API功能测试")
    success_count = 0
    
    for i, image_path in enumerate(test_images):
        print(f"\n--- 测试图像 {i+1}/{len(test_images)} ---")
        
        # JSON API测试
        json_result = client.test_pose_estimation_json(image_path)
        if json_result:
            success_count += 1
        
        # 图像注释API测试
        save_name = f"annotated_{os.path.basename(image_path)}"
        image_result = client.test_pose_estimation_image(image_path, save_name)
        if image_result:
            success_count += 1
        
        # 兼容性端点测试
        compat_result = client.test_compatibility_endpoint(image_path)
        if compat_result:
            success_count += 1
        
        print("-" * 40)
    
    # 4. 测试汇总
    total_tests = len(test_images) * 3  # 每张图像测试3个端点
    print(f"\n📊 测试结果汇总:")
    print(f"   总测试数: {total_tests}")
    print(f"   成功: {success_count}")
    print(f"   失败: {total_tests - success_count}")
    print(f"   成功率: {(success_count / total_tests * 100):.1f}%")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！服务工作正常。")
    else:
        print("⚠️ 存在测试失败，请检查服务配置。")
    
    print(f"\n💡 下一步:")
    print("   1. Docker构建: docker build -t cloudpose:v1.0 .")
    print("   2. 负载测试: locust -f locust/locustfile.py --host={service_url}")

if __name__ == "__main__":
    main()