# main.py
import base64
import cv2
import numpy as np
import uuid
import time
import io
import sys
import os
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入YOLO模块
try:
    from ultralytics import YOLO
    logger.info("✅ 成功导入ultralytics YOLO")
except ImportError as e:
    logger.error(f"❌ 导入ultralytics失败: {e}")
    logger.error("请运行: pip install ultralytics")

# FastAPI应用初始化
app = FastAPI(
    title="CloudPose API - Model 3 YOLO",
    description="人体姿态估计Web服务，使用YOLO11l-pose模型",
    version="1.0.0"
)

# 全局变量存储模型实例
pose_detector = None

# Pydantic模型定义
class ImageRequest(BaseModel):
    """图像请求模型"""
    id: str
    image: str  # base64编码的图像

class KeypointResponse(BaseModel):
    """关键点检测响应模型"""
    id: str
    count: int
    boxes: list
    keypoints: list
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float

class AnnotatedImageResponse(BaseModel):
    """注释图像响应模型"""
    id: str
    image: str  # base64编码的注释图像

def initialize_model():
    """初始化YOLO姿态检测模型"""
    global pose_detector
    try:
        model_path = "./model3-yolo1/yolo11l-pose.pt"
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.warning(f"⚠️  本地模型文件不存在: {model_path}")
            logger.info("使用默认YOLO11l-pose模型，将自动下载...")
            model_path = "yolo11l-pose.pt"  # ultralytics会自动下载
        
        # 初始化YOLO模型
        pose_detector = YOLO(model_path)
        logger.info("✅ YOLO11l-pose模型初始化成功")
            
    except Exception as e:
        logger.error(f"❌ 模型初始化失败: {e}")
        pose_detector = None

def base64_to_image(base64_str):
    """将base64字符串转换为OpenCV图像"""
    try:
        image_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图像数据")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无效的图像数据: {str(e)}")

def image_to_base64(image):
    """将OpenCV图像转换为base64字符串"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像编码错误: {str(e)}")

def process_pose_estimation(image):
    """
    使用YOLO11l-pose进行姿态估计
    
    Args:
        image: OpenCV图像 (BGR格式)
        
    Returns:
        tuple: (boxes, keypoints, preprocess_time, inference_time, postprocess_time)
    """
    global pose_detector
    
    if pose_detector is None:
        raise HTTPException(status_code=500, detail="模型未初始化")
    
    # 预处理
    start_time = time.time()
    h, w = image.shape[:2]
    processed_image = image.copy()
    preprocess_time = time.time() - start_time
    
    # 推理
    start_time = time.time()
    try:
        # YOLO推理，verbose=False避免输出过多信息
        results = pose_detector(processed_image, verbose=False)
    except Exception as e:
        logger.error(f"推理错误: {e}")
        results = []
    
    inference_time = time.time() - start_time
    
    # 后处理
    start_time = time.time()
    
    boxes = []
    keypoints = []
    
    if results:
        for result in results:
            # 处理检测框
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if hasattr(box, 'xyxy') and hasattr(box, 'conf'):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        boxes.append({
                            "x": int(x1),
                            "y": int(y1),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1),
                            "probability": float(confidence)
                        })
            
            # 处理关键点
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                kpts = result.keypoints.data.cpu().numpy()  # shape: [num_persons, num_keypoints, 3]
                
                for person_kpts in kpts:
                    person_keypoints = []
                    for kpt in person_kpts:
                        x, y, conf = kpt
                        person_keypoints.append([float(x), float(y), float(conf)])
                    keypoints.append(person_keypoints)
    
    postprocess_time = time.time() - start_time
    
    return boxes, keypoints, preprocess_time, inference_time, postprocess_time

def annotate_image_yolo(image, boxes, keypoints):
    """在图像上绘制检测框和关键点"""
    annotated = image.copy()
    
    # YOLO姿态估计的17个关键点连接关系（COCO格式）
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # 关键点颜色 (BGR格式)
    colors = [(0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
              (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),
              (255, 128, 0), (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255),
              (51, 153, 255), (51, 153, 255)]
    
    # 绘制检测框
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"{box['probability']:.2f}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 绘制关键点和骨架
    for person_keypoints in keypoints:
        if len(person_keypoints) >= 17:  # 确保有17个关键点
            # 绘制关键点
            for i, (x, y, confidence) in enumerate(person_keypoints[:17]):
                if confidence > 0.5:  # 只绘制置信度高的点
                    color = colors[i] if i < len(colors) else (0, 0, 255)
                    cv2.circle(annotated, (int(x), int(y)), 4, color, -1)
                    cv2.circle(annotated, (int(x), int(y)), 4, (255, 255, 255), 1)
            
            # 绘制骨架连接线
            for connection in skeleton:
                kpt_a, kpt_b = connection[0] - 1, connection[1] - 1  # 转换为0索引
                if (kpt_a < len(person_keypoints) and kpt_b < len(person_keypoints)):
                    if (person_keypoints[kpt_a][2] > 0.5 and person_keypoints[kpt_b][2] > 0.5):
                        x1, y1 = int(person_keypoints[kpt_a][0]), int(person_keypoints[kpt_a][1])
                        x2, y2 = int(person_keypoints[kpt_b][0]), int(person_keypoints[kpt_b][1])
                        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    return annotated

# API端点定义
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    logger.info("正在启动CloudPose服务...")
    initialize_model()

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "CloudPose API - Model 3 YOLO",
        "version": "1.0.0",
        "model": "YOLO11l-pose",
        "endpoints": ["/api/pose_estimation", "/api/pose_estimation_annotation", "/health"]
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    model_status = "loaded" if pose_detector else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "model": "YOLO11l-pose"
    }

@app.post("/api/pose_estimation", response_model=KeypointResponse)
async def pose_estimation_json(request: ImageRequest):
    """
    姿态估计JSON API - Assignment要求的主要端点
    
    接收包含base64编码图像的JSON请求，返回检测到的关键点信息
    """
    try:
        logger.info(f"处理姿态估计请求，ID: {request.id}")
        
        # 解码图像
        image = base64_to_image(request.image)
        
        # 处理姿态估计
        boxes, keypoints, prep_time, inf_time, post_time = process_pose_estimation(image)
        
        response = KeypointResponse(
            id=request.id,
            count=len(boxes),
            boxes=boxes,
            keypoints=keypoints,
            speed_preprocess=prep_time * 1000,  # 转换为毫秒
            speed_inference=inf_time * 1000,
            speed_postprocess=post_time * 1000
        )
        
        logger.info(f"成功处理请求 {request.id}，检测到 {len(boxes)} 个人")
        return response
        
    except Exception as e:
        logger.error(f"处理请求失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理错误: {str(e)}")

@app.post("/api/pose_estimation_annotation", response_model=AnnotatedImageResponse)
async def pose_estimation_image(request: ImageRequest):
    """
    姿态估计图像注释API - Assignment要求的图像端点
    
    接收包含base64编码图像的JSON请求，返回带有关键点标注的图像
    """
    try:
        logger.info(f"处理图像注释请求，ID: {request.id}")
        
        # 解码图像
        image = base64_to_image(request.image)
        
        # 处理姿态估计
        boxes, keypoints, _, _, _ = process_pose_estimation(image)
        
        # 绘制注释
        annotated_image = annotate_image_yolo(image, boxes, keypoints)
        
        # 编码返回
        annotated_base64 = image_to_base64(annotated_image)
        
        response = AnnotatedImageResponse(
            id=request.id,
            image=annotated_base64
        )
        
        logger.info(f"成功处理图像注释请求 {request.id}")
        return response
        
    except Exception as e:
        logger.error(f"处理图像注释失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理错误: {str(e)}")

@app.post("/api/pose_detection", response_model=KeypointResponse)
async def pose_detection_alias(request: ImageRequest):
    """
    与原始客户端兼容的端点别名
    重定向到主要的姿态估计端点
    """
    return await pose_estimation_json(request)

if __name__ == "__main__":
    # 运行服务器
    logger.info("启动CloudPose Web服务...")
    logger.info("访问 http://localhost:60001/health 检查服务状态")
    logger.info("API文档: http://localhost:60001/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=60001)