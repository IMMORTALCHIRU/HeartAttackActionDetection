from bson import ObjectId
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from flask_pymongo import PyMongo
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import os
import cv2
import time
import base64
from datetime import datetime
import json

from config import Config
from models import User, Detection

# Import email service conditionally
try:
    from email_service import mail, send_alert_email, send_welcome_email
    EMAIL_AVAILABLE = True
except ImportError:
    print("WARNING: Flask-Mail not available. Email functionality will be disabled.")
    EMAIL_AVAILABLE = False
    mail = None
    send_alert_email = None
    send_welcome_email = None

from detection_system import HeartAttackDetectionSystem, process_image, process_video

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
mongo = PyMongo(app)
if EMAIL_AVAILABLE:
    mail.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Create upload directories
for folder in [Config.UPLOAD_FOLDER, Config.SCREENSHOT_FOLDER, Config.OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global detection system for live feed
live_detection_system = None
live_detection_active = False


@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(mongo, user_id)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('detection'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()
        phone = request.form.get('phone', '').strip()
        emergency_contact = request.form.get('emergency_contact', '').strip()
        emergency_email = request.form.get('emergency_email', '').strip()
        
        # Validation
        errors = []
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if not email:
            errors.append('Email is required.')
        if not password or len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        if not emergency_email:
            errors.append('Emergency email is required for alerts.')
        if emergency_email == email:
            errors.append('Emergency email must be different from your regular email.')
        
        # Check if user exists
        if User.get_by_email(mongo, email):
            errors.append('Email already registered.')
        existing_user, _ = User.get_by_username(mongo, username)
        if existing_user:
            errors.append('Username already taken.')
        
        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')
        
        # Create user
        try:
            user = User.create_user(
                mongo, username, email, password,
                full_name, phone, emergency_contact, emergency_email
            )
            
            # Send welcome email
            if EMAIL_AVAILABLE:
                print(f"DEBUG: About to send welcome email to {email}")
                send_welcome_email(email, full_name or username)
                print(f"DEBUG: Welcome email function called")
            else:
                print("DEBUG: Email service not available - skipping welcome email")
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Registration failed: {str(e)}', 'danger')
            return render_template('register.html')
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('detection'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        user, stored_password = User.get_by_username(mongo, username)
        
        if user and User.verify_password(stored_password, password):
            login_user(user, remember=remember)
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('detection'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        emergency_email = request.form.get('emergency_email', '').strip()
        emergency_contact = request.form.get('emergency_contact', '').strip()
        phone = request.form.get('phone', '').strip()
        
        errors = []
        if not emergency_email:
            errors.append('Emergency email is required.')
        if emergency_email == current_user.email:
            errors.append('Emergency email must be different from your regular email.')
        
        if errors:
            for error in errors:
                flash(error, 'danger')
        else:
            # Update user data
            mongo.db.users.update_one(
                {'_id': ObjectId(current_user.id)},
                {'$set': {
                    'emergency_email': emergency_email,
                    'emergency_contact': emergency_contact,
                    'phone': phone
                }}
            )
            flash('Profile updated successfully!', 'success')
            # Reload user data
            current_user.emergency_email = emergency_email
            current_user.emergency_contact = emergency_contact
            current_user.phone = phone
    
    return render_template('profile.html', user=current_user)


@app.route('/history')
@login_required
def history():
    detections = Detection.get_user_detections(mongo, current_user.id)
    stats = Detection.get_detection_stats(mongo, current_user.id)
    return render_template('history.html', detections=detections, stats=stats)


@app.route('/detection')
@login_required
def detection():
    return render_template('detection.html')


# ==================== API ROUTES ====================

@app.route('/api/detect/image', methods=['POST'])
@login_required
def detect_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process image
        output_path, results = process_image(
            filepath,
            Config.DETECTION_CONFIG,
            Config.SCREENSHOT_FOLDER
        )
        
        # Determine detection type
        combined = results.get('combined', {})
        if combined.get('emergency'):
            detection_type = 'emergency'
        elif combined.get('heart_attack'):
            detection_type = 'heart_attack'
        elif combined.get('fall'):
            detection_type = 'fall'
        else:
            detection_type = 'normal'
        
        # Get screenshot path
        screenshot_path = results.get('screenshot_path', '')
        
        # Save to database
        detection_record = Detection.create_detection(
            mongo,
            current_user.id,
            detection_type,
            'image',
            filepath,
            output_path,
            screenshot_path,
            combined.get('status', 'normal'),
            combined.get('confidence', 0),
            combined.get('signs', []),
            combined.get('emergency', False)
        )
        
        # Send email alert if detection occurred
        if detection_type in ['heart_attack', 'fall', 'emergency']:
            if EMAIL_AVAILABLE and current_user.emergency_email:
                print(f"DEBUG: About to send alert email to {current_user.emergency_email} for {detection_type}")
                send_alert_email(
                    current_user.emergency_email,
                    detection_type,
                    combined.get('confidence', 0),
                    combined.get('signs', []),
                    screenshot_path,
                    current_user.full_name or current_user.username
                )
                print(f"DEBUG: Alert email function called")
            elif not EMAIL_AVAILABLE:
                print("DEBUG: Email service not available - skipping email alert")
            else:
                print(f"DEBUG: No emergency email found for user {current_user.id}")
        
        # Prepare response
        response = {
            'success': True,
            'detection_type': detection_type,
            'confidence': combined.get('confidence', 0),
            'signs': combined.get('signs', []),
            'output_path': os.path.relpath(output_path, 'static').replace('\\', '/') if output_path else None,
            'screenshot_path': os.path.relpath(screenshot_path, 'static').replace('\\', '/') if screenshot_path else None,
            'detection_id': str(detection_record['_id'])
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/video', methods=['POST'])
@login_required
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process video
        output_path, results = process_video(
            filepath,
            Config.DETECTION_CONFIG,
            Config.SCREENSHOT_FOLDER,
            Config.OUTPUT_FOLDER
        )
        
        # Determine detection type
        combined = results.get('combined', {})
        if combined.get('emergency'):
            detection_type = 'emergency'
        elif combined.get('heart_attack'):
            detection_type = 'heart_attack'
        elif combined.get('fall'):
            detection_type = 'fall'
        else:
            detection_type = 'normal'
        
        screenshot_path = results.get('screenshot_path', '')
        
        # Save to database
        detection_record = Detection.create_detection(
            mongo,
            current_user.id,
            detection_type,
            'video',
            filepath,
            output_path,
            screenshot_path,
            combined.get('status', 'normal'),
            combined.get('confidence', 0),
            combined.get('signs', []),
            combined.get('emergency', False)
        )
        
        # Send email alert
        if detection_type in ['heart_attack', 'fall', 'emergency']:
            if EMAIL_AVAILABLE and current_user.emergency_email:
                print(f"DEBUG: About to send alert email to {current_user.emergency_email} for {detection_type}")
                send_alert_email(
                    current_user.emergency_email,
                    detection_type,
                    combined.get('confidence', 0),
                    combined.get('signs', []),
                    screenshot_path,
                    current_user.full_name or current_user.username
                )
                print(f"DEBUG: Alert email function called")
            elif not EMAIL_AVAILABLE:
                print("DEBUG: Email service not available - skipping email alert")
            else:
                print(f"DEBUG: No emergency email found for user {current_user.id}")
        
        response = {
            'success': True,
            'detection_type': detection_type,
            'confidence': combined.get('confidence', 0),
            'signs': combined.get('signs', []),
            'output_path': os.path.relpath(output_path, 'static').replace('\\', '/') if output_path else None,
            'screenshot_path': os.path.relpath(screenshot_path, 'static').replace('\\', '/') if screenshot_path else None,
            'detection_id': str(detection_record['_id'])
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Live detection variables
live_camera = None
live_system = None

def generate_frames():
    """Generator function for live video feed."""
    global live_camera, live_system, live_detection_active
    
    live_camera = cv2.VideoCapture(0)
    live_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    live_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    live_system = HeartAttackDetectionSystem(Config.DETECTION_CONFIG)
    
    while live_detection_active:
        success, frame = live_camera.read()
        if not success:
            break
        
        annotated, results = live_system.process_frame(frame, Config.SCREENSHOT_FOLDER)
        
        # Check for alerts
        combined = results.get('combined', {})
        if combined.get('heart_attack') or combined.get('fall') or combined.get('emergency'):
            # Save annotated frame to outputs folder for email attachment
            try:
                os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                output_filename = f"live_output_{timestamp}.jpg"
                output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)
                cv2.imwrite(output_path, annotated)
                print(f"DEBUG: Saved live annotated frame to outputs: {output_path}")
            except Exception as e:
                print(f"DEBUG: Error saving live frame to outputs: {e}")
            
            # Store detection info in session for retrieval
            session['live_detection'] = {
                'type': combined.get('status'),
                'confidence': combined.get('confidence'),
                'signs': combined.get('signs', []),
                'timestamp': datetime.now().isoformat()
            }
        
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    if live_camera:
        live_camera.release()


@app.route('/api/detect/live/start', methods=['POST'])
@login_required
def start_live_detection():
    global live_detection_active
    live_detection_active = True
    return jsonify({'success': True, 'message': 'Live detection started'})


@app.route('/api/detect/live/stop', methods=['POST'])
@login_required
def stop_live_detection():
    global live_detection_active, live_camera, live_system
    live_detection_active = False
    
    if live_camera:
        live_camera.release()
        live_camera = None
    
    if live_system:
        # Save detection record
        if hasattr(live_system, 'state'):
            state = live_system.state
            if state.get('heart_attack_confirmed') or state.get('fall_confirmed'):
                detection_type = 'emergency' if state.get('emergency') else \
                               'heart_attack' if state.get('heart_attack_confirmed') else 'fall'
                
                Detection.create_detection(
                    mongo,
                    current_user.id,
                    detection_type,
                    'webcam',
                    '',
                    '',
                    '',
                    detection_type,
                    state.get('heart_attack_confidence', 0),
                    state.get('all_signs', []),
                    state.get('emergency', False)
                )
                
                # Send email alert for live detection
                if EMAIL_AVAILABLE and current_user.emergency_email:
                    print(f"DEBUG: About to send alert email to {current_user.emergency_email} for live {detection_type}")
                    
                    # Save the screenshot frame to outputs folder if available
                    screenshot_path = None
                    print(f"DEBUG: Checking for screenshot_frame in state: {state.get('screenshot_frame') is not None}")
                    
                    if state.get('screenshot_frame') is not None:
                        try:
                            os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            filename = f"live_detection_{detection_type}_{timestamp}.jpg"
                            screenshot_path = os.path.join(Config.OUTPUT_FOLDER, filename)
                            cv2.imwrite(screenshot_path, state['screenshot_frame'])
                            print(f"DEBUG: Saved live detection screenshot: {screenshot_path}")
                            print(f"DEBUG: File exists after save: {os.path.exists(screenshot_path)}")
                        except Exception as e:
                            print(f"DEBUG: Error saving live screenshot: {e}")
                    else:
                        print("DEBUG: No screenshot_frame found, checking outputs folder")
                        # Fallback: Find the most recent image from outputs folder
                        try:
                            # Use absolute path based on the app location
                            app_dir = os.path.dirname(os.path.abspath(__file__))
                            output_dir = os.path.join(app_dir, Config.OUTPUT_FOLDER)
                            print(f"DEBUG: App directory: {app_dir}")
                            print(f"DEBUG: Full output_dir path: {output_dir}")
                            print(f"DEBUG: output_dir exists: {os.path.exists(output_dir)}")
                            
                            if os.path.exists(output_dir):
                                files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                print(f"DEBUG: Found files in outputs: {files}")
                                
                                if files:
                                    # Sort by modification time, get the most recent
                                    files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
                                    screenshot_path = os.path.join(output_dir, files[0])
                                    print(f"DEBUG: Using most recent output image: {screenshot_path}")
                                    print(f"DEBUG: File exists: {os.path.exists(screenshot_path)}")
                                    print(f"DEBUG: File size: {os.path.getsize(screenshot_path) if os.path.exists(screenshot_path) else 'N/A'}")
                                else:
                                    print("DEBUG: No image files found in outputs folder")
                            else:
                                print(f"DEBUG: Output directory does not exist: {output_dir}")
                        except Exception as e:
                            print(f"DEBUG: Error finding output image: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print(f"DEBUG: Final screenshot_path: {screenshot_path}")
                    
                    send_alert_email(
                        current_user.emergency_email,
                        detection_type,
                        state.get('heart_attack_confidence', 0),
                        state.get('all_signs', []),
                        screenshot_path,
                        current_user.full_name or current_user.username
                    )
                    print(f"DEBUG: Live alert email function called")
                elif not EMAIL_AVAILABLE:
                    print("DEBUG: Email service not available - skipping live email alert")
                else:
                    print(f"DEBUG: No emergency email found for user {current_user.id}")
        
        live_system = None
    
    return jsonify({'success': True, 'message': 'Live detection stopped'})


@app.route('/api/detect/live/feed')
@login_required
def live_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/detect/live/status')
@login_required
def live_status():
    global live_system
    
    if live_system and hasattr(live_system, 'state'):
        state = live_system.state
        return jsonify({
            'active': live_detection_active,
            'heart_attack_detected': state.get('heart_attack_confirmed', False),
            'heart_attack_detecting': state.get('heart_attack_detecting', False),
            'fall_detected': state.get('fall_confirmed', False),
            'emergency': state.get('emergency', False),
            'confidence': state.get('heart_attack_confidence', 0),
            'signs': state.get('all_signs', [])
        })
    
    return jsonify({'active': live_detection_active})


@app.route('/api/history/<detection_id>')
@login_required
def get_detection_detail(detection_id):
    detection = Detection.get_detection_by_id(mongo, detection_id)
    if detection and str(detection['user_id']) == current_user.id:
        detection['_id'] = str(detection['_id'])
        detection['user_id'] = str(detection['user_id'])
        detection['created_at'] = detection['created_at'].isoformat()
        return jsonify(detection)
    return jsonify({'error': 'Detection not found'}), 404


@app.route('/api/history/delete/<detection_id>', methods=['DELETE'])
@login_required
def delete_detection(detection_id):
    try:
        from bson.objectid import ObjectId
        result = mongo.db.detections.delete_one({
            '_id': ObjectId(detection_id),
            'user_id': ObjectId(current_user.id)
        })
        if result.deleted_count:
            return jsonify({'success': True})
        return jsonify({'error': 'Detection not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('base.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('base.html', error='Server error'), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)