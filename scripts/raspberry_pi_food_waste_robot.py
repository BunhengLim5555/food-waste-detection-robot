#!/usr/bin/env python3
"""
Food Waste Detection Robot for Raspberry Pi 4
Uses TFLite model to detect food waste and control robot actions
Optimized for 12-20 FPS with class-specific confidence thresholds and NMS
"""

import cv2
import numpy as np
import time
import threading
try:
    import RPi.GPIO as GPIO
    from auppbot import AUPPBot
    from flask import Flask, Response
    RPI_AVAILABLE = True
except ImportError:
    print("⚠️ Running in simulation mode (RPi libraries not available)")
    RPI_AVAILABLE = False

try:
    import tensorflow as tf
except ImportError:
    print("❌ TensorFlow not installed! Run: pip install tensorflow")
    exit(1)

# ============= MODEL CONFIGURATION =============
MODEL_PATH = "food_waste_model_float16.tflite"
LABELS = ["Watermelon rind", "Apple core", "fish", "Egg shell"]

# Class-specific confidence thresholds
CLASS_THRESHOLDS = {
    "Watermelon rind": 0.5,  # High confidence - works well
    "Apple core": 0.3,        # Lower threshold - needs help
    "fish": 0.5,              # High confidence - works well
    "Egg shell": 0.5          # High confidence - works well
}

# ============= CLASS TO ACTION MAPPING =============
# Based on project guidelines
CLASS_ACTIONS = {
    "Watermelon rind": "TURN_LEFT",
    "Apple core": "TURN_RIGHT",
    "fish": "MOVE_FORWARD",
    "Egg shell": "MOVE_BACKWARD"
}

# ============= CAMERA CONFIGURATION =============
CAM_INDEX = 0
FRAME_WIDTH = 320  # Reduced for better FPS on Pi 4
FRAME_HEIGHT = 320
ROTATE_180 = True  # Set based on your camera orientation

# ============= MOTOR CONTROL PARAMETERS =============
PORT = "/dev/ttyUSB0"
BAUD = 115200
BASE_SPEED = 18       # Base motor speed
TURN_SPEED = 22       # Speed for turning
APPROACH_SPEED = 15   # Speed when approaching object

# PID Control for steering
KP = 0.5  # Proportional gain
DEADBAND = 0.15  # Center deadband (ignore small offsets)

LEFT_SIGN = +1
RIGHT_SIGN = +1

# ============= DETECTION PARAMETERS =============
IOU_THRESHOLD = 0.3  # Non-Maximum Suppression threshold
MIN_DETECTION_AREA = 0.02  # Minimum box area (as fraction of frame)
TARGET_AREA_MIN = 0.05  # Too far - move forward
TARGET_AREA_MAX = 0.40  # Too close - move backward
TARGET_AREA_CRUISE = 0.15  # Ideal distance

# Frame smoothing (exponential moving average)
SMOOTHING_FRAMES = 3
CONFIDENCE_GATE = 0.3  # Minimum confidence to act

# Loss handling
MAX_FRAMES_LOST = 15  # Frames before entering scan mode
SCAN_SPEED = 18       # Rotation speed during scan

# ============= WEB STREAMING =============
if RPI_AVAILABLE:
    app = Flask(__name__)
current_frame = None
frame_lock = threading.Lock()

# ============= GLOBAL STATE =============
detection_history = []  # For frame smoothing
frames_without_detection = 0

# =============================================================================
# TFLITE DETECTOR CLASS
# =============================================================================
class FoodWasteDetector:
    def __init__(self, model_path):
        """Initialize TFLite interpreter with Flex ops support"""
        print(f"Loading TFLite model from {model_path}...")

        try:
            # Try with Flex delegate for TF Select ops
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
            )
        except Exception as e:
            print(f"⚠️ Flex delegate failed: {e}")
            print("Trying standard interpreter...")
            self.interpreter = tf.lite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

        print(f"✅ Model loaded! Input size: {self.width}x{self.height}")
        print(f"   Input dtype: {self.input_details[0]['dtype']}")

    def preprocess_image(self, image):
        """Preprocess image for TFLite model"""
        # Resize to model input size
        input_image = cv2.resize(image, (self.width, self.height))

        # Convert BGR to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Normalize if model expects float32
        if self.input_details[0]['dtype'] == np.float32:
            input_image = input_image.astype(np.float32) / 255.0

        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)

        return input_image

    def detect(self, image):
        """Run detection on image"""
        # Preprocess
        input_data = self.preprocess_image(image)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms

        # Get results (SSD MobileNet output format)
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        return boxes, classes, scores, inference_time

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        ymin1, xmin1, ymax1, xmax1 = box1
        ymin2, xmin2, ymax2, xmax2 = box2

        # Intersection area
        intersect_ymin = max(ymin1, ymin2)
        intersect_xmin = max(xmin1, xmin2)
        intersect_ymax = min(ymax1, ymax2)
        intersect_xmax = min(xmax1, xmax2)

        intersect_width = max(0, intersect_xmax - intersect_xmin)
        intersect_height = max(0, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        # Union area
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        union_area = area1 + area2 - intersect_area

        if union_area == 0:
            return 0
        return intersect_area / union_area

    def process_detections(self, boxes, classes, scores):
        """Apply NMS with Apple core priority and return best detection"""
        # Collect valid detections above threshold
        valid_detections = []
        for i in range(len(scores)):
            raw_class_id = int(classes[i])
            class_id = raw_class_id - 1  # Convert from 1-indexed to 0-indexed
            if 0 <= class_id < len(LABELS):
                label = LABELS[class_id]
                threshold = CLASS_THRESHOLDS.get(label, 0.5)

                if scores[i] > threshold:
                    valid_detections.append({
                        'box': boxes[i],
                        'score': scores[i],
                        'label': label,
                        'class_id': class_id
                    })

        if not valid_detections:
            return None

        # Sort by PRIORITY: Apple core first, then by confidence
        def detection_priority(det):
            if det['label'] == 'Apple core':
                return (0, -det['score'])  # Highest priority
            else:
                return (1, -det['score'])

        valid_detections.sort(key=detection_priority)

        # Non-Maximum Suppression
        final_detections = []
        for detection in valid_detections:
            keep = True
            for selected in final_detections:
                iou = self.calculate_iou(detection['box'], selected['box'])
                if iou > IOU_THRESHOLD:
                    keep = False
                    break

            if keep:
                final_detections.append(detection)

        # Return highest priority detection
        return final_detections[0] if final_detections else None

# =============================================================================
# MOTOR CONTROL FUNCTIONS
# =============================================================================
def clamp99(x):
    """Clamp motor speed to [-99, 99]"""
    return int(max(-99, min(99, x)))

def set_tank(bot, left, right):
    """Set tank drive motor speeds"""
    if not RPI_AVAILABLE:
        print(f"  → [SIM] Motors: L={left}, R={right}")
        return

    try:
        l = clamp99(LEFT_SIGN * left)
        r = clamp99(RIGHT_SIGN * right)
        bot.motor1.speed(l)
        bot.motor2.speed(l)
        bot.motor3.speed(r)
        bot.motor4.speed(r)
    except Exception as e:
        print(f"❌ Motor error: {e}")

def turn_left(bot, duration=1.0):
    """Turn robot left"""
    print(f"🔄 TURNING LEFT...")
    set_tank(bot, -TURN_SPEED, TURN_SPEED)
    time.sleep(duration)
    set_tank(bot, 0, 0)

def turn_right(bot, duration=1.0):
    """Turn robot right"""
    print(f"🔄 TURNING RIGHT...")
    set_tank(bot, TURN_SPEED, -TURN_SPEED)
    time.sleep(duration)
    set_tank(bot, 0, 0)

def move_forward(bot, duration=1.0, speed=BASE_SPEED):
    """Move robot forward"""
    print(f"➡️ MOVING FORWARD...")
    set_tank(bot, speed, speed)
    time.sleep(duration)
    set_tank(bot, 0, 0)

def move_backward(bot, duration=1.0, speed=BASE_SPEED):
    """Move robot backward"""
    print(f"⬅️ MOVING BACKWARD...")
    set_tank(bot, -speed, -speed)
    time.sleep(duration)
    set_tank(bot, 0, 0)

def stop_robot(bot):
    """Stop all motors"""
    set_tank(bot, 0, 0)

# =============================================================================
# STEERING CONTROL WITH PID
# =============================================================================
def calculate_steering(box, frame_width):
    """Calculate steering adjustment based on box position (PID control)"""
    ymin, xmin, ymax, xmax = box

    # Calculate box center X position (normalized 0-1)
    box_center_x = (xmin + xmax) / 2.0
    frame_center_x = 0.5

    # Calculate offset from center
    offset = box_center_x - frame_center_x

    # Apply deadband (ignore small offsets)
    if abs(offset) < DEADBAND:
        return 0  # Go straight

    # Proportional control
    angular_speed = KP * offset

    return angular_speed

def calculate_distance_action(box):
    """Determine action based on box size (area)"""
    ymin, xmin, ymax, xmax = box
    box_area = (xmax - xmin) * (ymax - ymin)

    if box_area < TARGET_AREA_MIN:
        return "FORWARD"  # Too far, move closer
    elif box_area > TARGET_AREA_MAX:
        return "BACKWARD"  # Too close, back up
    else:
        return "CRUISE"  # Good distance

# =============================================================================
# FRAME SMOOTHING (Exponential Moving Average)
# =============================================================================
def smooth_detection(current_detection):
    """Smooth detections over multiple frames"""
    global detection_history

    if current_detection is None:
        return None

    detection_history.append(current_detection)
    if len(detection_history) > SMOOTHING_FRAMES:
        detection_history.pop(0)

    # Return most common class in recent detections
    if detection_history:
        labels = [d['label'] for d in detection_history]
        most_common = max(set(labels), key=labels.count)

        # Return detection with most common label
        for d in reversed(detection_history):
            if d['label'] == most_common:
                return d

    return current_detection

# =============================================================================
# WEB STREAMING
# =============================================================================
def generate_frames():
    """Generator for streaming frames"""
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

if RPI_AVAILABLE:
    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        return '''
        <html>
          <head>
            <title>Food Waste Detection Robot</title>
            <style>
              body { background: #000; display: flex; justify-content: center;
                     align-items: center; height: 100vh; margin: 0; font-family: Arial; }
              .container { text-align: center; }
              h1 { color: #0f0; font-size: 24px; margin-bottom: 10px; }
              img { border: 3px solid #0f0; max-width: 95vw; }
              .info { color: #0f0; margin-top: 10px; font-size: 14px; }
            </style>
          </head>
          <body>
            <div class="container">
              <h1>🗑️ Food Waste Detection Robot</h1>
              <img src="/video_feed" width="640" height="480">
              <div class="info">Real-time TFLite Object Detection</div>
            </div>
          </body>
        </html>
        '''

# =============================================================================
# MAIN CONTROL LOOP
# =============================================================================
def robot_control_loop(bot, cap, detector):
    """Main control loop: detect food waste and control robot"""
    global current_frame, frames_without_detection

    last_print = 0.0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    print("\n" + "="*70)
    print("🤖 ROBOT CONTROL ACTIVE - Waiting for food waste detection...")
    print("="*70)
    print("Class-to-Action Mapping:")
    for label, action in CLASS_ACTIONS.items():
        print(f"  • {label} → {action}")
    print("="*70 + "\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ Failed to grab frame")
                break

            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            h, w = frame.shape[:2]

            # ===== RUN DETECTION =====
            boxes, classes, scores, inference_time = detector.detect(frame)

            # ===== PROCESS DETECTIONS (NMS + Priority) =====
            best_detection = detector.process_detections(boxes, classes, scores)

            # ===== SMOOTH DETECTIONS =====
            smoothed_detection = smooth_detection(best_detection)

            # ===== ROBOT CONTROL LOGIC =====
            if smoothed_detection and smoothed_detection['score'] >= CONFIDENCE_GATE:
                frames_without_detection = 0

                label = smoothed_detection['label']
                confidence = smoothed_detection['score']
                box = smoothed_detection['box']
                action = CLASS_ACTIONS.get(label, "STOP")

                print(f"✅ DETECTED: {label} ({confidence:.0%}) → ACTION: {action}")

                # Calculate steering offset
                steering = calculate_steering(box, w)
                distance_action = calculate_distance_action(box)

                # Execute action based on detected class
                if action == "TURN_LEFT":
                    set_tank(bot, -TURN_SPEED, TURN_SPEED)

                elif action == "TURN_RIGHT":
                    set_tank(bot, TURN_SPEED, -TURN_SPEED)

                elif action == "MOVE_FORWARD":
                    # Apply PID steering while moving forward
                    if distance_action == "FORWARD":
                        left_speed = APPROACH_SPEED - int(steering * TURN_SPEED)
                        right_speed = APPROACH_SPEED + int(steering * TURN_SPEED)
                        set_tank(bot, left_speed, right_speed)
                    elif distance_action == "CRUISE":
                        # Maintain distance with steering
                        left_speed = BASE_SPEED - int(steering * TURN_SPEED)
                        right_speed = BASE_SPEED + int(steering * TURN_SPEED)
                        set_tank(bot, left_speed, right_speed)
                    else:  # BACKWARD
                        set_tank(bot, 0, 0)

                elif action == "MOVE_BACKWARD":
                    set_tank(bot, -APPROACH_SPEED, -APPROACH_SPEED)

                else:
                    stop_robot(bot)

            else:
                # No detection - enter scan mode if lost for too long
                frames_without_detection += 1

                if frames_without_detection > MAX_FRAMES_LOST:
                    print(f"⚠️ Target lost! Scanning... ({frames_without_detection} frames)")
                    # Slow rotation to search for target
                    set_tank(bot, SCAN_SPEED, -SCAN_SPEED)
                else:
                    stop_robot(bot)

            # ===== VISUALIZATION =====
            vis = frame.copy()

            # Draw detection if available
            if best_detection:
                ymin, xmin, ymax, xmax = best_detection['box']
                left = int(xmin * w)
                top = int(ymin * h)
                right = int(xmax * w)
                bottom = int(ymax * h)

                label = best_detection['label']
                confidence = best_detection['score']
                action = CLASS_ACTIONS.get(label, "UNKNOWN")

                # Color based on class
                color_map = {
                    "Watermelon rind": (0, 255, 0),
                    "Apple core": (0, 165, 255),
                    "fish": (255, 0, 255),
                    "Egg shell": (255, 255, 0)
                }
                color = color_map.get(label, (255, 255, 255))

                # Draw box
                cv2.rectangle(vis, (left, top), (right, bottom), color, 2)

                # Draw label + action
                label_text = f"{label}: {confidence:.0%}"
                action_text = f"Action: {action}"

                cv2.putText(vis, label_text, (left, top - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(vis, action_text, (left, top - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw center crosshair
                box_center_x = (left + right) // 2
                box_center_y = (top + bottom) // 2
                cv2.circle(vis, (box_center_x, box_center_y), 5, (0, 0, 255), -1)

            # Draw frame center
            cv2.line(vis, (w//2, 0), (w//2, h), (0, 255, 255), 1)
            cv2.line(vis, (0, h//2), (w, h//2), (0, 255, 255), 1)

            # FPS counter
            fps_counter += 1
            if fps_counter % 10 == 0:
                current_fps = 10 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            # Display info
            cv2.putText(vis, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis, f"Inference: {inference_time:.1f}ms", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Update web stream
            with frame_lock:
                current_frame = vis.copy()

            # Throttled console output
            if time.time() - last_print > 1.0:
                status = f"FPS: {current_fps:.1f} | Inference: {inference_time:.1f}ms"
                if best_detection:
                    status += f" | {best_detection['label']} ({best_detection['score']:.0%})"
                else:
                    status += " | No detection"
                print(status)
                last_print = time.time()

    finally:
        stop_robot(bot)

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    print("🚀 Starting Food Waste Detection Robot...")

    # Initialize detector
    try:
        detector = FoodWasteDetector(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"Make sure '{MODEL_PATH}' exists!")
        return

    # Initialize robot
    bot = None
    if RPI_AVAILABLE:
        try:
            bot = AUPPBot(PORT, BAUD, auto_safe=True)
            print("✅ Robot connected!")
        except Exception as e:
            print(f"❌ Failed to connect to robot: {e}")
            return

    # Initialize camera
    print("Opening camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print("✅ Camera opened!")

    # Start control thread
    control_thread = threading.Thread(
        target=robot_control_loop,
        args=(bot, cap, detector),
        daemon=True
    )
    control_thread.start()

    if RPI_AVAILABLE:
        # Start web server
        import socket
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        print("\n" + "="*70)
        print("🎥 WEB STREAM STARTED!")
        print("="*70)
        print(f"\n📱 Open browser: http://{ip_address}:5000")
        print("="*70 + "\n")

        try:
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping robot...")
    else:
        # Simulation mode - just run control loop
        try:
            control_thread.join()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping...")

    # Cleanup
    if RPI_AVAILABLE and bot:
        try:
            stop_robot(bot)
            bot.stop_all()
            bot.close()
        except:
            pass
    cap.release()
    if RPI_AVAILABLE:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
