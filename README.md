# Heart Attack & Fall Detection Surveillance System

A comprehensive web-based AI surveillance system that uses advanced computer vision and pose detection to identify heart attack and fall incidents in real-time, with automated emergency alerts and multi-user support.

## 🎯 Project Overview

The **Heart Attack & Fall Detection Surveillance System** is an intelligent monitoring application designed to enhance public safety and emergency response. It leverages state-of-the-art YOLOv8 pose detection technology to analyze human body postures and movements, identifying critical health emergencies such as heart attacks and falls. When dangerous situations are detected, the system automatically notifies emergency contacts via email, enabling rapid intervention.

### Key Capabilities
- **Real-time Detection**: Analyze live webcam feeds, video files, and images
- **Dual Threat Detection**: Identify both heart attacks and falls with high accuracy
- **Smart Alerting**: Automated email notifications with screenshot evidence
- **User Management**: Secure multi-user authentication system
- **Detection History**: Comprehensive logging and tracking of all detections
- **Emergency Contacts**: Configure emergency recipients for instant notifications

---

## 🌟 Features

### 1. **Heart Attack Detection**
The system identifies cardiac emergency patterns by analyzing body keypoints and posture:
- **Hand-on-Chest Detection**: Monitors proximity of hands to the chest area (distance < 150 pixels)
- **Elbow Angle Analysis**: Detects bent arms at angles between 30° and 140°
- **Hunched Posture Recognition**: Identifies forward-bending body position indicating distress
- **Combined Signs Assessment**: Requires multiple indicators for accurate detection
- **Confidence Scoring**: Provides confidence percentages for detection reliability

**Detection Signature**: Hands on chest + bent elbows OR hunched posture + body position indicators

### 2. **Fall Detection**
The system detects sudden falls and dangerous body collapses:
- **Body Orientation Analysis**: Detects rapid changes from vertical to horizontal position
- **Velocity Estimation**: Calculates movement speed to identify sudden drops
- **Position Validation**: Confirms sustained ground-level body position
- **Head Position Tracking**: Monitors head-to-ground relationship
- **Confidence Scoring**: Distinguishes actual falls from intentional ground movements

**Detection Signature**: Rapid orientation change + low body position + sustained ground contact

### 3. **User Authentication & Management**
- **Secure Registration**: Email verification and password hashing with bcrypt
- **Login System**: Remember-me functionality for convenient access
- **Profile Management**: Update emergency contacts and personal information
- **Session Management**: Secure Flask-Login integration

### 4. **Alert System**
- **Email Notifications**: HTML-formatted alert emails with embedded screenshots
- **Emergency Contacts**: Configure separate emergency email for alerts
- **Alert Details**: Includes detection type, confidence level, identified signs, and timestamp
- **Attachments**: Screenshots automatically attached to alert emails

### 5. **Detection History**
- **Comprehensive Logging**: All detections stored in MongoDB with metadata
- **User-specific Records**: View personal detection history with timestamps
- **Statistics Dashboard**: Aggregated detection statistics by type
- **Result Tracking**: Stores detection results, confidence levels, and identified signs

### 6. **Multi-Format Support**
Supported file formats for processing:
- **Images**: PNG, JPG, JPEG, GIF, WebP, BMP
- **Videos**: MP4, AVI, MOV, MKV, WebM
- **Live Input**: Webcam feed streaming

---

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| **Flask** | Web application framework |
| **Flask-PyMongo** | MongoDB integration |
| **Flask-Login** | User session management |
| **Flask-Mail** | Email service integration |
| **Flask-WTF** | Form validation & CSRF protection |

### Machine Learning & Computer Vision
| Technology | Purpose |
|-----------|---------|
| **YOLOv8n-Pose** | Human pose estimation (17-point keypoint detection) |
| **Ultralytics** | YOLOv8 model management |
| **OpenCV** | Image/video processing and visualization |
| **NumPy** | Numerical computations for angle/distance calculations |

### Database & Security
| Technology | Purpose |
|-----------|---------|
| **MongoDB** | NoSQL database for user profiles and detection records |
| **PyMongo** | MongoDB Python driver |
| **bcrypt** | Password hashing and verification |
| **python-dotenv** | Environment variable management |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **HTML5** | Page structure and semantic markup |
| **CSS3** | Responsive styling |
| **JavaScript** | Dynamic UI interactions and AJAX |

---

## 📋 Project Structure

```
heart-attack-action/
├── app.py                      # Main Flask application & route handlers
├── config.py                   # Configuration settings & constants
├── models.py                   # Database models (User, Detection)
├── detection_system.py         # Core AI detection logic
├── email_service.py            # Email alert functionality
├── requirements.txt            # Python dependencies
├── yolov8n-pose.pt            # Pre-trained YOLOv8 nano pose model
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css          # Application styling
│   ├── js/
│   │   └── main.js            # Frontend interactions
│   ├── uploads/               # User-uploaded files
│   ├── screenshots/           # Detection screenshots
│   └── outputs/               # Processed videos
├── templates/                  # HTML templates
│   ├── base.html              # Base layout template
│   ├── index.html             # Landing page
│   ├── login.html             # Login form
│   ├── register.html          # Registration form
│   ├── profile.html           # User profile page
│   ├── detection.html         # Detection interface
│   ├── history.html           # Detection history
│   ├── components/
│   │   ├── navbar.html        # Navigation bar
│   │   └── footer.html        # Footer component
└── test/                      # Test files & sample media
    ├── 1.webp                 # Test image
    ├── 2.webp                 # Test image
    └── h.mp4                  # Test video

Database Collections:
├── users                       # User profiles with credentials
├── detections                  # Detection records with results
```

---

## 🔑 Core Components

### 1. Detection System (`detection_system.py`)

#### HeartAttackPoseDetector Class
Analyzes body keypoints from YOLOv8 pose detection to identify cardiac emergency patterns.

**Key Methods**:
- `detect(keypoints)`: Returns detection status with confidence score and identified signs

**Detection Logic**:
- Extracts 17 keypoints from pose model (nose, shoulders, elbows, wrists, hips, ankles, etc.)
- Calculates distances between hand and chest positions
- Computes elbow joint angles
- Analyzes posture and body alignment
- Generates composite confidence score

#### FallDetector Class
Identifies falls and dangerous body collapses using multiple indicators.

**Key Methods**:
- `detect(keypoints, frame_height, frame_width)`: Returns fall detection with type and confidence

**Detection Indicators**:
- Body orientation change from vertical to horizontal
- Velocity-based drop detection
- Head-to-ground proximity
- Sustained low body position

#### HeartAttackDetectionSystem Class
Main orchestrator combining pose detection, fall detection, and result management.

**Key Methods**:
- `detect_from_image(image_path)`: Analyze static images
- `detect_from_video(video_path)`: Process video files frame-by-frame
- `detect_from_webcam()`: Real-time webcam monitoring
- `process_frame(frame)`: Single frame analysis

### 2. Database Models (`models.py`)

#### User Model
Represents user profiles with authentication data.

**Fields**:
- `username`: Unique username (3+ characters)
- `email`: User email address
- `password`: Bcrypt-hashed password
- `full_name`: Display name
- `phone`: Contact phone number
- `emergency_contact`: Name/description of emergency recipient
- `emergency_email`: Email for alert notifications
- `created_at`: Registration timestamp

**Methods**:
- `create_user()`: Register new user with bcrypt hashing
- `get_by_id()`: Retrieve user by MongoDB ObjectID
- `get_by_username()`: Fetch user and hashed password by username
- `get_by_email()`: Find user by email address
- `verify_password()`: Validate plaintext password against hash

#### Detection Model
Represents detection records and historical data.

**Fields**:
- `user_id`: LinkedUser ID (ObjectID)
- `detection_type`: Type of detection ('heart_attack', 'fall', 'emergency', 'normal')
- `file_type`: Input source ('image', 'video', 'webcam')
- `file_path`: Path to original uploaded file
- `output_path`: Path to processed output
- `screenshot_path`: Path to captured screenshot
- `result`: Detection result (true/false)
- `confidence`: Confidence score (0.0-1.0)
- `signs`: List of identified signs/indicators
- `emergency`: Boolean flag for emergency-level detections
- `created_at`: Detection timestamp

**Methods**:
- `create_detection()`: Log new detection record
- `get_user_detections()`: Retrieve user's detection history
- `get_detection_by_id()`: Fetch specific detection record
- `get_detection_stats()`: Aggregate statistics by detection type

### 3. Email Service (`email_service.py`)

**Key Functions**:
- `send_alert_email()`: Send HTML-formatted alert with screenshot
  - Customized subject lines for detection types
  - Rich HTML formatting with alert styling
  - Inline screenshot attachments
  - Action instructions for emergency response

- `send_welcome_email()`: Welcome message for new registrations

**Email Features**:
- Professional HTML templates with gradient headers
- Emergency action instructions
- Detection details (type, confidence, signs, timestamp)
- Screenshot evidence included
- Color-coded alert styling for urgency

### 4. Configuration (`config.py`)

**Settings**:
```python
# Security
SECRET_KEY          # Flask session key (change in production)

# Database
MONGO_URI          # MongoDB connection string

# Email
MAIL_SERVER        # SMTP server (Gmail in this case)
MAIL_PORT          # SMTP port (587 for TLS)
MAIL_USERNAME      # Gmail account
MAIL_PASSWORD      # App-specific password
MAIL_DEFAULT_SENDER # From address

# File Handling
UPLOAD_FOLDER      # User uploads location
SCREENSHOT_FOLDER  # Screenshot storage
OUTPUT_FOLDER      # Processed video output
MAX_CONTENT_LENGTH # Max file upload (100MB)
ALLOWED_EXTENSIONS # Supported formats

# Detection Thresholds
hand_chest_distance      # 150px - proximity for hand-on-chest
elbow_angle_min/max      # 30°-140° - bent arm angle range
shoulder_hunch_threshold # 0.15 - hunched posture indicator
shoulder_hip_collapse    # 60px - vertical collapse distance
head_ground_ratio        # 0.75 - head-to-body position
fall_velocity_threshold  # 150 - drop speed indicator
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- MongoDB 4.0+
- Webcam (optional, for live detection)
- NVIDIA GPU (optional, for faster inference)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/heart-attack-surveillance.git
cd heart-attack-action
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```env
# Flask Configuration
SECRET_KEY=your-secret-key-here-change-in-production
FLASK_ENV=development
FLASK_DEBUG=True

# MongoDB
MONGO_URI=mongodb://localhost:27017/heart_attack_surveillance
# or for MongoDB Atlas:
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/heart_attack_surveillance

# Email Configuration (Gmail)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-specific-password
```

**Note on Gmail**: Use an [App-Specific Password](https://support.google.com/accounts/answer/185833) instead of your regular password.

### Step 5: Verify MongoDB Connection
Ensure MongoDB is running:
```bash
# Local MongoDB
mongod

# Or use MongoDB Atlas (cloud)
# Update MONGO_URI with your connection string
```

### Step 6: Run Application
```bash
python app.py
```

Application will be available at `http://localhost:5000`

---

## 📖 User Guide

### Registration
1. Click **"Register"** on the landing page
2. Enter username, email, and password (6+ characters)
3. Provide full name, phone number, and emergency contact
4. **Critical**: Enter emergency email (must differ from registration email)
5. Submit to create account
6. Welcome email will be sent to your address

### Login
1. Enter username and password
2. Optional: Check "Remember me" to stay logged in
3. Access detection dashboard

### Profile Management
1. Navigate to **Profile** after login
2. Update emergency contact information
3. Update emergency email address (for critical alerts)
4. Update phone number
5. Changes take effect immediately

### Detection (Image)
1. Go to **Detection** page
2. Click **Upload Image**
3. Select image file (PNG, JPG, WebP, etc.)
4. System analyzes and displays results:
   - Detection status (Heart Attack / Fall / Normal)
   - Confidence percentage
   - Identified signs
   - Annotated image with pose keypoints
5. Results saved to detection history

### Detection (Video)
1. Go to **Detection** page
2. Click **Upload Video**
3. Select video file (MP4, AVI, MOV, WebM)
4. System processes frame-by-frame
5. Generates output video with annotations
6. Summarizes detections across video
7. Results and timestamps saved to history

### Detection (Webcam)
1. Go to **Detection** page
2. Click **Start Webcam**
3. Allow camera permissions
4. Real-time detection in browser
5. Frame-by-frame analysis displayed
6. Automatic email alert if condition detected

### Detection History
1. Navigate to **History** after login
2. View all past detections with thumbnails
3. Click on record to see detailed information:
   - Detection type and confidence
   - Identified signs
   - Timestamp
   - Screenshot evidence
4. Filter by detection type (optional)
5. Sort by most recent

### Alert Features
When a suspicious condition is detected:
- **Email Alert**: Emergency contact receives detailed notification
- **Screenshot**: Evidence image attached to email
- **Details Included**: Detection type, confidence, signs, timestamp
- **Action Instructions**: Guidance for emergency response
- **UI Alert**: Visual alert displayed on screen

---

## 🎯 Detection Parameters & Thresholds

### Heart Attack Detection Thresholds
| Parameter | Value | Description |
|-----------|-------|-------------|
| `hand_chest_distance` | 150 px | Maximum distance between hand and chest for detection |
| `elbow_angle_min` | 30° | Minimum elbow bend angle |
| `elbow_angle_max` | 140° | Maximum elbow bend angle |
| `shoulder_hunch_threshold` | 0.15 | Posture change indicator |
| `head_ground_ratio` | 0.75 | Head position relative to body |
| `heart_attack_confirm_seconds` | 3 s | Duration for confirmation |
| `pose_conf` | 0.7 | YOLOv8 confidence threshold |

### Fall Detection Thresholds
| Parameter | Value | Description |
|-----------|-------|-------------|
| `fall_velocity_threshold` | 150 | Minimum drop speed for detection |
| `shoulder_hip_collapse` | 60 px | Vertical distance collapse indicator |
| `horizontal_body_ratio` | 1.5 | Width-to-height ratio for horizontal position |
| `fall_confirm_frames` | 5 | Frames to confirm fall status |

### System Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `screenshot_enabled` | True | Capture evidence images |
| `alarm_enabled` | False | Ring alarm on detection (configurable) |
| `alert_display_seconds` | 5 | Alert message duration |
| `max_upload_size` | 100 MB | Maximum file size for uploads |

---

## 🔒 Security Features

### Authentication & Authorization
- **Bcrypt Hashing**: Passwords hashed with bcrypt using salt rounds
- **Session Management**: Flask-Login with secure session tokens
- **Login Required**: Route protection with `@login_required` decorator
- **Password Validation**: Minimum 6 characters, confirmation matching
- **Email Verification**: Existing email detection during registration

### Data Security
- **MongoDB ObjectIDs**: Secure user ID references
- **CSRF Protection**: Flask-WTF cross-site request forgery prevention
- **File Upload Validation**: Whitelist of allowed extensions
- **Secure Filenames**: Werkzeug `secure_filename()` for uploads
- **Environment Variables**: Sensitive credentials in `.env` file

### Privacy Considerations
- **User Isolation**: Users only see their own detection data
- **Emergency Contact Separation**: Different email for alerts
- **Screenshot Storage**: Local filesystem with access control
- **Data Retention**: Detection records indefinitely (consider GDPR compliance)

---

## ⚙️ Configuration Customization

### Adjusting Detection Sensitivity
Edit `config.py` to tune detection thresholds:

```python
# More sensitive (lower thresholds)
"hand_chest_distance": 120  # Detects hands farther from chest
"elbow_angle_max": 160      # Accepts more bent positions

# Less sensitive (higher thresholds)
"hand_chest_distance": 180  # Requires hands closer to chest
"fall_velocity_threshold": 200  # Requires faster drops
```

### Email Configuration
Update `config.py` for different email providers:

```python
# Gmail (current)
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587

# Microsoft Outlook
MAIL_SERVER = 'smtp.office365.com'
MAIL_PORT = 587

# Custom SMTP
MAIL_SERVER = 'mail.yourdomain.com'
MAIL_PORT = 465
```

### MongoDB Atlas (Cloud)
```python
MONGO_URI = 'mongodb+srv://username:password@cluster.mongodb.net/heart_attack_surveillance?retryWrites=true&w=majority'
```

---

## 🐛 Troubleshooting

### Common Issues

#### Port 5000 Already in Use
```bash
# Windows: Find and kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :5000
kill -9 <PID>

# Or use different port
python app.py --port 5001
```

#### MongoDB Connection Error
```
Error: MongoServerSelectionError
```
Solution:
- Ensure MongoDB is running: `mongod`
- Check `MONGO_URI` in `.env`
- Verify connection string syntax
- Check firewall permissions

#### Email Not Sending
```
Error: SMTPAuthenticationError
```
Solutions:
- Enable "Less secure apps" in Gmail settings (not recommended)
- Use App-Specific Password instead of account password
- Verify MAIL_USERNAME and MAIL_PASSWORD in `.env`
- Check MAIL_SERVER and MAIL_PORT are correct
- Ensure `EMAIL_AVAILABLE` flag is True

#### Webcam Not Accessible
- Grant browser camera permissions
- Check webcam device is properly connected
- Verify no other application is using the camera
- Test browser console for permission errors

#### YOLOv8 Model Not Loading
```
Error: No such file or directory: 'yolov8n-pose.pt'
```
Solution:
- Ensure `yolov8n-pose.pt` is in project root directory
- Or specify correct path in `config.py`:
  ```python
  "pose_model_path": "/full/path/to/yolov8n-pose.pt"
  ```
- Download model: `from ultralytics import YOLO; YOLO('yolov8n-pose.yaml').train()`

---

## 📊 Database Schema

### Users Collection
```json
{
  "_id": ObjectId,
  "username": "string (unique)",
  "email": "string (unique)",
  "password": "bcrypt_hash",
  "full_name": "string",
  "phone": "string",
  "emergency_contact": "string",
  "emergency_email": "string",
  "created_at": "datetime"
}
```

### Detections Collection
```json
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "detection_type": "heart_attack|fall|emergency|normal",
  "file_type": "image|video|webcam",
  "file_path": "string",
  "output_path": "string",
  "screenshot_path": "string",
  "result": true|false,
  "confidence": 0.0-1.0,
  "signs": ["string"],
  "emergency": true|false,
  "created_at": "datetime"
}
```

---

## 🚦 API Endpoints

### Authentication Routes
- `GET /` - Landing page
- `GET/POST /register` - User registration
- `GET/POST /login` - User login
- `GET /logout` - User logout (requires login)

### User Routes
- `GET/POST /profile` - User profile management (requires login)

### Detection Routes
- `GET /detection` - Detection interface (requires login)
- `POST /detect_image` - Analyze uploaded image
- `POST /detect_video` - Analyze uploaded video
- `GET /webcam_feed` - Webcam live stream

### History Routes
- `GET /history` - Detection history (requires login)
- `GET /api/detection/<detection_id>` - Detailed detection info

---

## 🤖 YOLOv8 Pose Detection Model

### Model: yolov8n-pose
- **Architecture**: Nano-sized YOLOv8
- **Keypoints Detected**: 17 body points
  1. Nose
  2. Left Eye
  3. Right Eye
  4. Left Ear
  5. Right Ear
  6. Left Shoulder
  7. Right Shoulder
  8. Left Elbow
  9. Right Elbow
  10. Left Wrist
  11. Right Wrist
  12. Left Hip
  13. Right Hip
  14. Left Knee
  15. Right Knee
  16. Left Ankle
  17. Right Ankle

- **Confidence**: 0.7 minimum for detection
- **Input**: Color images (any resolution)
- **Output**: Keypoints with x, y, confidence scores

### Model Performance
- **Inference Time**: ~10-50ms per frame (GPU)
- **Memory**: ~100MB RAM
- **Model Size**: ~11MB
- **Supported Formats**: JPEG, PNG, BMP, WebP

---

## 📈 Performance Optimization

### Image Processing
- **Resolution Scaling**: Resize frames to 480x640 for faster inference
- **Batch Processing**: Process videos in 30-frame batches
- **GPU Acceleration**: Enable CUDA if available

### Database
- **Indexing**: Add indexes on `user_id` and `created_at` fields
  ```javascript
  db.detections.createIndex({ user_id: 1, created_at: -1 })
  ```
- **Query Limits**: History limited to 50 recent records
- **Archive Strategy**: Consider archiving old detections (>1 year)

### Web Server
- **Production Server**: Use Gunicorn/Waitress instead of Flask dev server
  ```bash
  pip install gunicorn
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
  ```
- **Cloud Hosting**: Deploy on Heroku, AWS, Google Cloud
- **CDN**: Serve static files from CDN for faster delivery

---

## 🔄 Deployment Guide

### Local Development
```bash
python app.py
```

### Heroku Deployment
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Create runtime.txt
echo "python-3.11.0" > runtime.txt

# Push to Heroku
heroku create your-app-name
git push heroku main
```

### Docker Deployment
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t heart-attack-surveillance .
docker run -p 5000:5000 heart-attack-surveillance
```

---

## 📚 Dependencies & Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | Latest | Web framework |
| Flask-PyMongo | Latest | MongoDB integration |
| Flask-Login | Latest | User authentication |
| Flask-WTF | Latest | Form protection |
| Flask-Mail | Latest | Email sending |
| pymongo | Latest | MongoDB driver |
| bcrypt | Latest | Password hashing |
| python-dotenv | Latest | Environment variables |
| opencv-python | Latest | Image/video processing |
| numpy | Latest | Numerical computation |
| ultralytics | Latest | YOLOv8 models |
| Werkzeug | Latest | WSGI utilities |
| email-validator | Latest | Email validation |
| dnspython | Latest | DNS resolution |

---

## 🧪 Testing

### Manual Testing
1. **Registration**: Create test account with valid email
2. **Login**: Verify session persistence
3. **Image Detection**: Upload test images and verify results
4. **Video Detection**: Process test videos frame-by-frame
5. **Webcam**: Check real-time detection accuracy
6. **History**: Verify detection records are saved
7. **Email**: Confirm alert emails arrive correctly

### Test Files
Test media included in `/test` directory:
- `1.webp`, `2.webp` - Sample images
- `h.mp4` - Sample video file
---

## ⚠️ Disclaimer

**This system is for safety monitoring and emergency assistance purposes. It should NOT be considered a medical diagnostic tool.** 

- Results are based on pose analysis and may have false positives/negatives
- Always verify alerts and seek professional medical help when needed
- For life-threatening emergencies, call emergency services (911 in US) immediately
- The developers assume no liability for missed detections or misuse

---
