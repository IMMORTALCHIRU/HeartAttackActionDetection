from flask_mail import Mail, Message
from flask import current_app
import os
import base64
from datetime import datetime

mail = Mail()

mail = Mail()

def send_alert_email(recipient_email, detection_type, confidence, signs, screenshot_path=None, user_name='User'):
    """Send alert email with screenshot attachment."""
    try:
        print(f"DEBUG: Preparing to send alert email to {recipient_email} for {detection_type}")
        
        subject_map = {
            'heart_attack': '🚨 URGENT: Heart Attack Detected!',
            'fall': '⚠️ ALERT: Fall Detected!',
            'emergency': '🆘 EMERGENCY: Heart Attack + Fall Detected!'
        }
        
        subject = subject_map.get(detection_type, '⚠️ Health Alert Notification')
        print(f"DEBUG: Email subject: {subject}")
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border: 1px solid #ddd; }}
                .alert-box {{ background: #fff; border-left: 4px solid #e74c3c; padding: 20px; margin: 20px 0; }}
                .info-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
                .footer {{ background: #2c3e50; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
                .btn {{ display: inline-block; background: #e74c3c; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 Heart Attack Surveillance System</h1>
                    <p>Emergency Alert Notification</p>
                </div>
                <div class="content">
                    <div class="alert-box">
                        <h2 style="color: #e74c3c; margin-top: 0;">⚠️ {detection_type.replace('_', ' ').title()} Detected!</h2>
                        <p>An alert has been triggered for <strong>{user_name}</strong>.</p>
                    </div>
                    
                    <h3>Detection Details:</h3>
                    <div class="info-row">
                        <span><strong>Detection Type:</strong></span>
                        <span>{detection_type.replace('_', ' ').title()}</span>
                    </div>
                    <div class="info-row">
                        <span><strong>Confidence Level:</strong></span>
                        <span>{confidence:.1%}</span>
                    </div>
                    <div class="info-row">
                        <span><strong>Signs Detected:</strong></span>
                        <span>{', '.join(signs) if signs else 'N/A'}</span>
                    </div>
                    <div class="info-row">
                        <span><strong>Time:</strong></span>
                        <span>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    
                    <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin-top: 20px; border-radius: 5px;">
                        <strong>⚡ Immediate Action Required:</strong>
                        <ul>
                            <li>Check on the person immediately</li>
                            <li>Call emergency services if needed (911)</li>
                            <li>Administer first aid if trained</li>
                        </ul>
                    </div>
                    
                    <p style="text-align: center;">
                        <a href="#" class="btn">View Full Report</a>
                    </p>
                </div>
                <div class="footer">
                    <p>This is an automated alert from the Heart Attack Surveillance System.</p>
                    <p>© 2024 Heart Attack & Smart Surveillance System</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg = Message(
            subject=subject,
            recipients=[recipient_email],
            html=html_body
        )
        
        # Attach screenshot if available
        if screenshot_path and os.path.exists(screenshot_path):
            print(f"DEBUG: Attaching screenshot: {screenshot_path}")
            with open(screenshot_path, 'rb') as f:
                msg.attach(
                    filename=os.path.basename(screenshot_path),
                    content_type='image/jpeg',
                    data=f.read()
                )
        else:
            print(f"DEBUG: No screenshot to attach or file not found: {screenshot_path}")
        
        print(f"DEBUG: Sending email via SMTP...")
        mail.send(msg)
        print(f"DEBUG: Alert email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"DEBUG: Error sending alert email: {e}")
        return False


def send_welcome_email(recipient_email, user_name):
    """Send welcome email after registration."""
    try:
        print(f"DEBUG: Preparing to send welcome email to {recipient_email} for user {user_name}")
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border: 1px solid #ddd; }}
                .feature {{ display: flex; align-items: center; margin: 15px 0; padding: 15px; background: white; border-radius: 8px; }}
                .feature-icon {{ font-size: 24px; margin-right: 15px; }}
                .footer {{ background: #2c3e50; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
                .btn {{ display: inline-block; background: #3498db; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 Welcome to Heart Attack Surveillance System</h1>
                    <p>Your Health, Our Priority</p>
                </div>
                <div class="content">
                    <h2>Hello {user_name}! 👋</h2>
                    <p>Thank you for registering with our Heart Attack & Smart Surveillance System. Your account has been successfully created.</p>
                    
                    <h3>What you can do:</h3>
                    <div class="feature">
                        <span class="feature-icon">📷</span>
                        <div>
                            <strong>Image Detection</strong>
                            <p>Upload images to detect heart attack poses</p>
                        </div>
                    </div>
                    <div class="feature">
                        <span class="feature-icon">🎥</span>
                        <div>
                            <strong>Video Analysis</strong>
                            <p>Analyze videos for continuous monitoring</p>
                        </div>
                    </div>
                    <div class="feature">
                        <span class="feature-icon">📹</span>
                        <div>
                            <strong>Live Detection</strong>
                            <p>Real-time monitoring through webcam</p>
                        </div>
                    </div>
                    <div class="feature">
                        <span class="feature-icon">📊</span>
                        <div>
                            <strong>History & Reports</strong>
                            <p>Track all your detection history</p>
                        </div>
                    </div>
                    
                    <p style="text-align: center;">
                        <a href="#" class="btn">Get Started</a>
                    </p>
                </div>
                <div class="footer">
                    <p>Stay safe and healthy!</p>
                    <p>© 2024 Heart Attack & Smart Surveillance System</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg = Message(
            subject='Welcome to Heart Attack Surveillance System! 🏥',
            recipients=[recipient_email],
            html=html_body
        )
        
        print(f"DEBUG: Sending welcome email via SMTP...")
        mail.send(msg)
        print(f"DEBUG: Welcome email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"DEBUG: Error sending welcome email: {e}")
        return False