from flask_login import UserMixin
from bson.objectid import ObjectId
from datetime import datetime
import bcrypt

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.full_name = user_data.get('full_name', '')
        self.phone = user_data.get('phone', '')
        self.emergency_contact = user_data.get('emergency_contact', '')
        self.emergency_email = user_data.get('emergency_email', '')
        self.created_at = user_data.get('created_at', datetime.utcnow())
    
    @staticmethod
    def create_user(mongo, username, email, password, full_name='', phone='', emergency_contact='', emergency_email=''):
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_data = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'full_name': full_name,
            'phone': phone,
            'emergency_contact': emergency_contact,
            'emergency_email': emergency_email,
            'created_at': datetime.utcnow()
        }
        result = mongo.db.users.insert_one(user_data)
        user_data['_id'] = result.inserted_id
        return User(user_data)
    
    @staticmethod
    def get_by_id(mongo, user_id):
        user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if user_data:
            return User(user_data)
        return None
    
    @staticmethod
    def get_by_username(mongo, username):
        user_data = mongo.db.users.find_one({'username': username})
        if user_data:
            return User(user_data), user_data['password']
        return None, None
    
    @staticmethod
    def get_by_email(mongo, email):
        user_data = mongo.db.users.find_one({'email': email})
        if user_data:
            return User(user_data)
        return None
    
    @staticmethod
    def verify_password(stored_password, provided_password):
        return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)


class Detection:
    @staticmethod
    def create_detection(mongo, user_id, detection_type, file_type, file_path, output_path, 
                         screenshot_path, result, confidence, signs, emergency=False):
        detection_data = {
            'user_id': ObjectId(user_id),
            'detection_type': detection_type,  # 'heart_attack', 'fall', 'emergency', 'normal'
            'file_type': file_type,  # 'image', 'video', 'webcam'
            'file_path': file_path,
            'output_path': output_path,
            'screenshot_path': screenshot_path,
            'result': result,
            'confidence': confidence,
            'signs': signs,
            'emergency': emergency,
            'created_at': datetime.utcnow()
        }
        result = mongo.db.detections.insert_one(detection_data)
        detection_data['_id'] = result.inserted_id
        return detection_data
    
    @staticmethod
    def get_user_detections(mongo, user_id, limit=50):
        detections = mongo.db.detections.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).limit(limit)
        return list(detections)
    
    @staticmethod
    def get_detection_by_id(mongo, detection_id):
        return mongo.db.detections.find_one({'_id': ObjectId(detection_id)})
    
    @staticmethod
    def get_detection_stats(mongo, user_id):
        pipeline = [
            {'$match': {'user_id': ObjectId(user_id)}},
            {'$group': {
                '_id': '$detection_type',
                'count': {'$sum': 1}
            }}
        ]
        stats = list(mongo.db.detections.aggregate(pipeline))
        return {stat['_id']: stat['count'] for stat in stats}