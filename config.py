import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    
    # MongoDB Configuration
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/heart_attack_surveillance'
    
    # Mail Configuration
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'p47199655@gmail.com'
    MAIL_PASSWORD = 'sjrmgzsjtfaluirb'
    MAIL_DEFAULT_SENDER = 'p47199655@gmail.com'
    
    # Upload Configuration
    UPLOAD_FOLDER = 'static/uploads'
    SCREENSHOT_FOLDER = 'static/screenshots'
    OUTPUT_FOLDER = 'static/outputs'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'mp4', 'avi', 'mov', 'mkv', 'webm'}
    
    # Detection Configuration
    DETECTION_CONFIG = {
        "pose_model_path": "yolov8n-pose.pt",
        "pose_conf": 0.7,
        "hand_chest_distance": 150,
        "elbow_angle_min": 30,
        "elbow_angle_max": 140,
        "shoulder_hunch_threshold": 0.15,
        "shoulder_hip_collapse": 60,
        "head_ground_ratio": 0.75,
        "fall_velocity_threshold": 150,
        "horizontal_body_ratio": 1.5,
        "heart_attack_confirm_seconds": 3,
        "alert_display_seconds": 5,
        "fall_confirm_frames": 5,
        "alarm_enabled": False,
        "screenshot_enabled": True,
    }