import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
import threading
import base64

class DetectionConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def angle_between_points(a, b, c):
    """Calculate angle ABC in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    dot_product = np.dot(ab, cb)
    magnitude = np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6
    cosine_angle = np.clip(dot_product / magnitude, -1, 1)
    return np.degrees(np.arccos(cosine_angle))


def get_keypoint_safe(keypoints, idx):
    """Safely get keypoint, return None if invalid."""
    if keypoints is None or len(keypoints) <= idx:
        return None
    kp = keypoints[idx]
    if len(kp) < 2 or (kp[0] == 0 and kp[1] == 0):
        return None
    return [float(kp[0]), float(kp[1])]


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    if p1 is None or p2 is None:
        return float('inf')
    return np.linalg.norm(np.array(p1) - np.array(p2))


class HeartAttackPoseDetector:
    """Heart attack pose detection using body keypoints."""
    
    def __init__(self, config):
        self.config = config
        
    def detect(self, keypoints):
        """Detect heart attack pose from keypoints."""
        result = {
            "detected": False,
            "confidence": 0.0,
            "signs": [],
            "details": {}
        }
        
        if keypoints is None or len(keypoints) < 11:
            return result
        
        # Extract keypoints
        nose = get_keypoint_safe(keypoints, 0)
        l_shoulder = get_keypoint_safe(keypoints, 5)
        r_shoulder = get_keypoint_safe(keypoints, 6)
        l_elbow = get_keypoint_safe(keypoints, 7)
        r_elbow = get_keypoint_safe(keypoints, 8)
        l_wrist = get_keypoint_safe(keypoints, 9)
        r_wrist = get_keypoint_safe(keypoints, 10)
        l_hip = get_keypoint_safe(keypoints, 11)
        r_hip = get_keypoint_safe(keypoints, 12)
        
        if not all([l_shoulder, r_shoulder]):
            return result
        
        # Calculate chest midpoint
        chest = [
            (l_shoulder[0] + r_shoulder[0]) / 2,
            (l_shoulder[1] + r_shoulder[1]) / 2 + 30
        ]
        
        signs_detected = []
        confidence_scores = []
        
        # Check 1: Hand near chest
        left_hand_dist = calculate_distance(l_wrist, chest)
        right_hand_dist = calculate_distance(r_wrist, chest)
        
        left_hand_on_chest = left_hand_dist < self.config.hand_chest_distance
        right_hand_on_chest = right_hand_dist < self.config.hand_chest_distance
        
        if left_hand_on_chest:
            signs_detected.append("Left hand on chest")
            conf = 1 - (left_hand_dist / self.config.hand_chest_distance)
            confidence_scores.append(conf * 0.4)
            
        if right_hand_on_chest:
            signs_detected.append("Right hand on chest")
            conf = 1 - (right_hand_dist / self.config.hand_chest_distance)
            confidence_scores.append(conf * 0.4)
        
        # Check 2: Elbow bent
        left_elbow_bent = False
        right_elbow_bent = False
        
        if all([l_shoulder, l_elbow, l_wrist]):
            left_angle = angle_between_points(l_shoulder, l_elbow, l_wrist)
            if self.config.elbow_angle_min < left_angle < self.config.elbow_angle_max:
                left_elbow_bent = True
                signs_detected.append(f"Left arm bent ({left_angle:.0f}°)")
                angle_conf = 1 - abs(left_angle - 90) / 60
                confidence_scores.append(max(0, angle_conf) * 0.2)
        
        if all([r_shoulder, r_elbow, r_wrist]):
            right_angle = angle_between_points(r_shoulder, r_elbow, r_wrist)
            if self.config.elbow_angle_min < right_angle < self.config.elbow_angle_max:
                right_elbow_bent = True
                signs_detected.append(f"Right arm bent ({right_angle:.0f}°)")
                angle_conf = 1 - abs(right_angle - 90) / 60
                confidence_scores.append(max(0, angle_conf) * 0.2)
        
        # Check 3: Hunched posture
        hunched = False
        if all([nose, l_shoulder, r_shoulder, l_hip, r_hip]):
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            body_height = abs(hip_y - shoulder_y)
            
            if body_height > 0:
                head_forward = nose[1] > shoulder_y - (body_height * 0.3)
                if head_forward:
                    hunched = True
                    signs_detected.append("Hunched posture")
                    confidence_scores.append(0.15)
        
        # Check 4: Both hands on chest
        both_hands = left_hand_on_chest and right_hand_on_chest
        if both_hands:
            signs_detected.append("BOTH hands on chest")
            confidence_scores.append(0.3)
        
        # Compile results
        hand_on_chest = left_hand_on_chest or right_hand_on_chest
        elbow_bent = left_elbow_bent or right_elbow_bent
        
        result["detected"] = hand_on_chest and (elbow_bent or hunched or both_hands)
        result["signs"] = signs_detected
        result["details"] = {
            "left_hand_dist": left_hand_dist,
            "right_hand_dist": right_hand_dist,
            "chest_position": chest,
            "hand_on_chest": hand_on_chest,
            "elbow_bent": elbow_bent,
            "hunched": hunched,
            "both_hands": both_hands
        }
        
        if confidence_scores:
            result["confidence"] = min(1.0, sum(confidence_scores))
        
        return result


class FallDetector:
    """Fall detection with multiple indicators."""
    
    def __init__(self, config):
        self.config = config
        self.prev_positions = []
        self.max_history = 10
        self.prev_time = time.time()
        
    def detect(self, keypoints, frame_height, frame_width):
        """Detect falls using multiple indicators."""
        result = {
            "detected": False,
            "confidence": 0.0,
            "signs": [],
            "type": None
        }
        
        if keypoints is None or len(keypoints) < 15:
            return result
        
        # Extract keypoints
        nose = get_keypoint_safe(keypoints, 0)
        l_shoulder = get_keypoint_safe(keypoints, 5)
        r_shoulder = get_keypoint_safe(keypoints, 6)
        l_hip = get_keypoint_safe(keypoints, 11)
        r_hip = get_keypoint_safe(keypoints, 12)
        l_ankle = get_keypoint_safe(keypoints, 15)
        r_ankle = get_keypoint_safe(keypoints, 16)
        
        signs_detected = []
        confidence_scores = []
        fall_type = None
        
        # Calculate key positions
        shoulder_center = None
        hip_center = None
        ankle_center = None
        
        if all([l_shoulder, r_shoulder]):
            shoulder_center = [(l_shoulder[0] + r_shoulder[0]) / 2, 
                              (l_shoulder[1] + r_shoulder[1]) / 2]
        
        if all([l_hip, r_hip]):
            hip_center = [(l_hip[0] + r_hip[0]) / 2,
                         (l_hip[1] + r_hip[1]) / 2]
        
        if all([l_ankle, r_ankle]):
            ankle_center = [(l_ankle[0] + r_ankle[0]) / 2,
                           (l_ankle[1] + r_ankle[1]) / 2]
        
        # Check 1: Body collapse
        if shoulder_center and hip_center:
            vertical_distance = abs(hip_center[1] - shoulder_center[1])
            horizontal_distance = abs(hip_center[0] - shoulder_center[0])
            
            if vertical_distance < self.config.shoulder_hip_collapse:
                signs_detected.append("Body collapsed vertically")
                confidence_scores.append(0.4)
                fall_type = "collapse"
            
            if vertical_distance > 0:
                ratio = horizontal_distance / vertical_distance
                if ratio > self.config.horizontal_body_ratio:
                    signs_detected.append("Horizontal body position")
                    confidence_scores.append(0.5)
                    fall_type = "lying_down"
        
        # Check 2: Head near ground
        if nose:
            head_ratio = nose[1] / frame_height
            if head_ratio > self.config.head_ground_ratio:
                signs_detected.append(f"Head low ({head_ratio:.0%} of frame)")
                confidence_scores.append(0.35)
                fall_type = fall_type or "ground_level"
        
        # Check 3: Fast downward velocity
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if shoulder_center and len(self.prev_positions) > 0 and dt > 0:
            prev_shoulder = self.prev_positions[-1]
            velocity_y = (shoulder_center[1] - prev_shoulder[1]) / dt
            
            if velocity_y > self.config.fall_velocity_threshold:
                signs_detected.append(f"Rapid descent (v={velocity_y:.0f})")
                confidence_scores.append(0.45)
                fall_type = "rapid_fall"
        
        # Check 4: Ankles above hips
        if ankle_center and hip_center:
            if ankle_center[1] < hip_center[1] - 20:
                signs_detected.append("Ankles above hips")
                confidence_scores.append(0.4)
                fall_type = fall_type or "backward_fall"
        
        # Check 5: Shoulders at same level as hips
        if shoulder_center and hip_center:
            y_diff = abs(shoulder_center[1] - hip_center[1])
            if y_diff < 30:
                signs_detected.append("Shoulders level with hips")
                confidence_scores.append(0.35)
                fall_type = fall_type or "side_lying"
        
        # Update position history
        if shoulder_center:
            self.prev_positions.append(shoulder_center)
            if len(self.prev_positions) > self.max_history:
                self.prev_positions.pop(0)
        
        self.prev_time = current_time
        
        # Compile results
        total_confidence = sum(confidence_scores) if confidence_scores else 0
        
        result["detected"] = total_confidence >= 0.5 and len(signs_detected) >= 2
        result["signs"] = signs_detected
        result["confidence"] = min(1.0, total_confidence)
        result["type"] = fall_type
        
        return result
    
    def reset(self):
        """Reset detector state."""
        self.prev_positions = []
        self.prev_time = time.time()


class HeartAttackDetectionSystem:
    """Main detection system for heart attack and fall detection."""
    
    _instance = None
    _pose_model = None
    
    def __init__(self, config_dict):
        self.config = DetectionConfig(config_dict)
        
        # Load pose model (singleton to save memory)
        if HeartAttackDetectionSystem._pose_model is None:
            HeartAttackDetectionSystem._pose_model = YOLO(self.config.pose_model_path)
        self.pose_model = HeartAttackDetectionSystem._pose_model
        
        # Initialize detectors
        self.heart_pose_detector = HeartAttackPoseDetector(self.config)
        self.fall_detector = FallDetector(self.config)
        
        # State tracking
        self.state = self._get_initial_state()
    
    def _get_initial_state(self):
        """Get fresh initial state."""
        return {
            "heart_attack_confirmed": False,
            "heart_attack_detecting": False,
            "heart_attack_confidence": 0.0,
            "heart_attack_start_time": None,
            "heart_attack_confirm_time": None,
            "heart_attack_display_until": None,
            "fall_confirmed": False,
            "fall_detecting": False,
            "fall_consecutive_frames": 0,
            "fall_confirm_time": None,
            "fall_display_until": None,
            "emergency": False,
            "emergency_display_until": None,
            "heart_attack_screenshot_taken": False,
            "fall_screenshot_taken": False,
            "all_signs": [],
            "screenshot_frame": None
        }
    
    def process_frame(self, frame, screenshot_path=None, live=True):
        """Process a single frame through all detection pipelines."""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        current_time = time.time()
        
        results = {
            "pose_detection": {"detected": False},
            "fall_detection": {"detected": False},
            "combined": {
                "heart_attack": False,
                "fall": False,
                "emergency": False,
                "confidence": 0.0,
                "status": "normal"
            },
            "screenshot_path": None
        }
        
        # 1. Pose estimation
        pose_results = self.pose_model(frame, conf=self.config.pose_conf, verbose=False)
        
        keypoints = None
        if pose_results and pose_results[0].keypoints is not None:
            kp_data = pose_results[0].keypoints.xy
            if len(kp_data) > 0:
                keypoints = kp_data[0].cpu().numpy()
                annotated = self._draw_skeleton(annotated, keypoints)
        
        # 2. Heart attack pose detection
        pose_result = self.heart_pose_detector.detect(keypoints)
        results["pose_detection"] = pose_result
        
        # 3. Fall detection
        fall_result = self.fall_detector.detect(keypoints, h, w)
        results["fall_detection"] = fall_result
        
        # 4. Update state with timing logic
        screenshot_taken = self._update_state(pose_result, fall_result, current_time, annotated, screenshot_path, live)
        if screenshot_taken:
            results["screenshot_path"] = screenshot_taken
        
        # 5. Compile combined results
        results["combined"]["heart_attack"] = self.state["heart_attack_confirmed"]
        results["combined"]["fall"] = self.state["fall_confirmed"]
        results["combined"]["emergency"] = self.state["emergency"]
        results["combined"]["confidence"] = self.state["heart_attack_confidence"]
        results["combined"]["signs"] = self.state["all_signs"]
        
        # Determine status
        if self.state["emergency"]:
            results["combined"]["status"] = "EMERGENCY"
        elif self.state["heart_attack_confirmed"]:
            results["combined"]["status"] = "HEART_ATTACK"
        elif self.state["fall_confirmed"]:
            results["combined"]["status"] = "FALL"
        elif self.state["heart_attack_detecting"]:
            results["combined"]["status"] = "detecting"
        else:
            results["combined"]["status"] = "normal"
        
        # 6. Draw alerts
        annotated = self._draw_alerts(annotated, results, current_time)
        
        return annotated, results
    
    def _update_state(self, pose_result, fall_result, current_time, frame, screenshot_path, live=True):
        """Update detection state with proper timing logic."""
        screenshot_taken = None
        
        # Collect all signs
        all_signs = []
        if pose_result.get("signs"):
            all_signs.extend(pose_result["signs"])
        if fall_result.get("signs"):
            all_signs.extend(fall_result["signs"])
        self.state["all_signs"] = all_signs
        
        # Heart attack state machine
        if self.state["heart_attack_display_until"]:
            if current_time >= self.state["heart_attack_display_until"]:
                self._reset_heart_attack_state()
        
        if not self.state["heart_attack_confirmed"]:
            if pose_result["detected"]:
                if not live:
                    # Confirm immediately for non-live detection
                    self.state["heart_attack_confirmed"] = True
                    self.state["heart_attack_confirm_time"] = current_time
                    self.state["heart_attack_display_until"] = current_time + self.config.alert_display_seconds
                    self.state["heart_attack_confidence"] = pose_result["confidence"]
                    
                    # Take screenshot
                    if screenshot_path and not self.state["heart_attack_screenshot_taken"]:
                        screenshot_taken = self._take_screenshot(frame, "HEART_ATTACK", screenshot_path)
                        self.state["heart_attack_screenshot_taken"] = True
                        self.state["screenshot_frame"] = frame.copy()
                else:
                    # Live detection with timing
                    if not self.state["heart_attack_detecting"]:
                        self.state["heart_attack_detecting"] = True
                        self.state["heart_attack_start_time"] = current_time
                    
                    self.state["heart_attack_confidence"] = pose_result["confidence"]
                    detection_duration = current_time - self.state["heart_attack_start_time"]
                    
                    if detection_duration >= self.config.heart_attack_confirm_seconds:
                        self.state["heart_attack_confirmed"] = True
                        self.state["heart_attack_confirm_time"] = current_time
                        self.state["heart_attack_display_until"] = current_time + self.config.alert_display_seconds
                        
                        # Take screenshot
                        if screenshot_path and not self.state["heart_attack_screenshot_taken"]:
                            screenshot_taken = self._take_screenshot(frame, "HEART_ATTACK", screenshot_path)
                            self.state["heart_attack_screenshot_taken"] = True
                            self.state["screenshot_frame"] = frame.copy()
            else:
                if live:
                    self.state["heart_attack_detecting"] = False
                    self.state["heart_attack_start_time"] = None
                    self.state["heart_attack_confidence"] = 0.0
        
        # Fall state machine
        if self.state["fall_display_until"]:
            if current_time >= self.state["fall_display_until"]:
                self._reset_fall_state()
        
        if not self.state["fall_confirmed"]:
            if fall_result["detected"]:
                self.state["fall_consecutive_frames"] += 1
                self.state["fall_detecting"] = True
                
                if self.state["fall_consecutive_frames"] >= self.config.fall_confirm_frames:
                    self.state["fall_confirmed"] = True
                    self.state["fall_confirm_time"] = current_time
                    self.state["fall_display_until"] = current_time + self.config.alert_display_seconds
                    
                    if screenshot_path and not self.state["fall_screenshot_taken"]:
                        screenshot_taken = self._take_screenshot(frame, "FALL", screenshot_path)
                        self.state["fall_screenshot_taken"] = True
            else:
                self.state["fall_consecutive_frames"] = max(0, self.state["fall_consecutive_frames"] - 1)
                if self.state["fall_consecutive_frames"] == 0:
                    self.state["fall_detecting"] = False
        
        # Emergency check
        if self.state["heart_attack_confirmed"] and self.state["fall_confirmed"]:
            if not self.state["emergency"]:
                self.state["emergency"] = True
                self.state["emergency_display_until"] = max(
                    self.state["heart_attack_display_until"] or 0,
                    self.state["fall_display_until"] or 0
                )
                
                if screenshot_path:
                    screenshot_taken = self._take_screenshot(frame, "EMERGENCY", screenshot_path)
        
        if self.state["emergency"] and self.state["emergency_display_until"]:
            if current_time >= self.state["emergency_display_until"]:
                self.state["emergency"] = False
        
        return screenshot_taken
    
    def _take_screenshot(self, frame, detection_type, save_path):
        """Save screenshot with timestamp."""
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{detection_type}_{timestamp}.jpg"
        filepath = os.path.join(save_path, filename)
        cv2.imwrite(filepath, frame)
        return filepath
    
    def _reset_heart_attack_state(self):
        """Reset heart attack detection state."""
        self.state["heart_attack_confirmed"] = False
        self.state["heart_attack_detecting"] = False
        self.state["heart_attack_confidence"] = 0.0
        self.state["heart_attack_start_time"] = None
        self.state["heart_attack_confirm_time"] = None
        self.state["heart_attack_display_until"] = None
        self.state["heart_attack_screenshot_taken"] = False
    
    def _reset_fall_state(self):
        """Reset fall detection state."""
        self.state["fall_confirmed"] = False
        self.state["fall_detecting"] = False
        self.state["fall_consecutive_frames"] = 0
        self.state["fall_confirm_time"] = None
        self.state["fall_display_until"] = None
        self.state["fall_screenshot_taken"] = False
    
    def _draw_skeleton(self, frame, keypoints):
        """Draw pose skeleton on frame."""
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        for start_idx, end_idx in connections:
            start = get_keypoint_safe(keypoints, start_idx)
            end = get_keypoint_safe(keypoints, end_idx)
            if start and end:
                cv2.line(frame, 
                        (int(start[0]), int(start[1])),
                        (int(end[0]), int(end[1])),
                        (0, 255, 0), 2)
        
        for i, kp in enumerate(keypoints):
            if kp[0] != 0 or kp[1] != 0:
                if i in [9, 10]:
                    color = (0, 0, 255)
                elif i in [5, 6]:
                    color = (255, 0, 0)
                elif i in [11, 12]:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, color, -1)
        
        return frame
    
    def _draw_alerts(self, frame, results, current_time):
        """Draw alert overlays on frame."""
        h, w = frame.shape[:2]
        
        # Status panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        y_offset = 35
        
        # Heart Attack Status
        if self.state["heart_attack_confirmed"]:
            remaining = max(0, self.state["heart_attack_display_until"] - current_time)
            color = (0, 0, 255)
            text = f"HEART ATTACK DETECTED ({remaining:.1f}s)"
        elif self.state["heart_attack_detecting"]:
            elapsed = current_time - self.state["heart_attack_start_time"]
            color = (0, 165, 255)
            text = f"Detecting... ({elapsed:.1f}/{self.config.heart_attack_confirm_seconds}s)"
        else:
            color = (0, 255, 0)
            text = "Heart Monitor: Active"
        
        cv2.putText(frame, text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_offset += 35
        
        # Fall Status
        if self.state["fall_confirmed"]:
            remaining = max(0, self.state["fall_display_until"] - current_time)
            color = (0, 0, 255)
            text = f"FALL DETECTED ({remaining:.1f}s)"
        elif self.state["fall_detecting"]:
            color = (0, 255, 255)
            text = f"Fall detecting... ({self.state['fall_consecutive_frames']}/{self.config.fall_confirm_frames})"
        else:
            color = (0, 255, 0)
            text = "Fall Monitor: Active"
        
        cv2.putText(frame, text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_offset += 35
        
        # Confidence
        conf = self.state["heart_attack_confidence"]
        cv2.putText(frame, f"Confidence: {conf:.1%}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Emergency Banner
        if self.state["emergency"]:
            if int(current_time * 3) % 2 == 0:
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (0, 0), (w, h), (0, 0, 150), -1)
                frame = cv2.addWeighted(overlay2, 0.3, frame, 0.7, 0)
            
            cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 255), -1)
            cv2.putText(frame, "EMERGENCY: HEART ATTACK + FALL",
                       (w // 2 - 250, h - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "CALL EMERGENCY SERVICES!",
                       (w // 2 - 180, h - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif self.state["heart_attack_confirmed"]:
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 100, 255), -1)
            cv2.putText(frame, "WARNING: Heart Attack Signs Detected",
                       (w // 2 - 200, h - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        elif self.state["fall_confirmed"]:
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 165, 255), -1)
            cv2.putText(frame, "ALERT: Fall Detected",
                       (w // 2 - 120, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (w - 200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def reset(self):
        """Reset all detection states."""
        self.state = self._get_initial_state()
        self.fall_detector.reset()


def process_image(image_path, config_dict, screenshot_path):
    """Process a single image for detection."""
    system = HeartAttackDetectionSystem(config_dict)
    
    img = cv2.imread(image_path)
    if img is None:
        return None, {"error": "Cannot read image"}
    
    # Process multiple times to build confidence
    confirm_seconds = config_dict.get('heart_attack_confirm_seconds', 3)
    fps = 30
    total_frames = int(fps * (confirm_seconds + 1))
    
    final_results = None
    annotated = None
    
    for _ in range(total_frames):
        annotated, final_results = system.process_frame(img.copy(), screenshot_path, live=False)
    
    # Save output
    output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    output_path = os.path.join(os.path.dirname(screenshot_path), 'outputs', output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    
    return output_path, final_results


def process_video(video_path, config_dict, screenshot_path, output_folder):
    """Process a video file for detection."""
    system = HeartAttackDetectionSystem(config_dict)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"error": "Cannot open video"}
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(output_folder, output_filename)
    os.makedirs(output_folder, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    final_results = None
    frame_count = 0
    detection_occurred = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, results = system.process_frame(frame, screenshot_path, live=False)
        writer.write(annotated)
        
        if results["combined"]["heart_attack"] or results["combined"]["fall"]:
            detection_occurred = True
            final_results = results
        
        frame_count += 1
    
    cap.release()
    writer.release()
    
    if not detection_occurred:
        final_results = {
            "combined": {
                "heart_attack": False,
                "fall": False,
                "emergency": False,
                "confidence": 0.0,
                "status": "normal",
                "signs": []
            }
        }
    
    return output_path, final_results