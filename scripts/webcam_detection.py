#!/usr/bin/env python3
"""
Food Waste Detection - Webcam Test
Test your trained model on laptop webcam
Detects 4 classes: Watermelon Rind, Egg Shell, Apple Core, Fish Bone
"""

import numpy as np
import cv2
import time
import tensorflow as tf

# Configuration
MODEL_PATH = "food_waste_model_float16.tflite"
LABELS = ["Watermelon rind", "Apple core", "fish", "Egg shell"]
CONFIDENCE_THRESHOLD = 0.5

# Colors for each class (BGR format)
COLORS = {
    "Watermelon rind": (0, 255, 0),    # Green
    "Apple core": (0, 165, 255),       # Orange
    "fish": (255, 0, 255),             # Magenta
    "Egg shell": (255, 255, 0)         # Cyan
}

class FoodWasteDetector:
    def __init__(self, model_path):
        """Initialize the TFLite interpreter"""
        print(f"Loading model from {model_path}...")

        # Load TFLite model with Flex delegate support
        try:
            # Try loading with experimental delegates for Flex ops
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
            )
        except Exception as e:
            print(f"Note: Using standard interpreter (Flex ops may not work): {e}")
            self.interpreter = tf.lite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

        print(f"✅ Model loaded! Input size: {self.width}x{self.height}")
        print(f"   Input dtype: {self.input_details[0]['dtype']}")
        print(f"   Number of outputs: {len(self.output_details)}")

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to model input size (320x320)
        input_image = cv2.resize(image, (self.width, self.height))

        # Convert BGR to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] if model expects float input
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

    def draw_detections(self, image, boxes, classes, scores):
        """Draw bounding boxes and labels on image"""
        height, width = image.shape[:2]

        detections_found = 0
        detection_summary = {label: 0 for label in LABELS}

        for i in range(len(scores)):
            if scores[i] > CONFIDENCE_THRESHOLD:
                # Get box coordinates (normalized 0-1)
                ymin, xmin, ymax, xmax = boxes[i]

                # Convert to pixel coordinates
                left = int(xmin * width)
                top = int(ymin * height)
                right = int(xmax * width)
                bottom = int(ymax * height)

                # Get class info
                class_id = int(classes[i])
                if class_id < len(LABELS):
                    label = LABELS[class_id]
                    color = COLORS.get(label, (255, 255, 255))
                    confidence = scores[i]

                    # Draw bounding box
                    cv2.rectangle(image, (left, top), (right, bottom), color, 3)

                    # Draw label background
                    label_text = f"{label}: {confidence:.0%}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        image,
                        (left, top - text_height - 10),
                        (left + text_width, top),
                        color,
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        image,
                        label_text,
                        (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    detections_found += 1
                    detection_summary[label] += 1

        return image, detections_found, detection_summary


def main():
    """Main function for webcam detection"""
    print("\n" + "=" * 60)
    print("🗑️  FOOD WASTE DETECTION SYSTEM - WEBCAM TEST")
    print("=" * 60)
    print("\nClasses:")
    for label, color in COLORS.items():
        print(f"  • {label} - {color}")
    print(f"\nConfidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print("=" * 60 + "\n")

    # Initialize detector
    try:
        detector = FoodWasteDetector(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure 'food_waste_model_float16.tflite' is in the same directory!")
        return

    # Initialize camera
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam!")
        print("Make sure your webcam is connected and not being used by another app.")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Webcam initialized! Resolution: {actual_width}x{actual_height}")
    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  • Press 'q' to quit")
    print("  • Press 's' to save screenshot")
    print("  • Press 'r' to reset statistics")
    print("=" * 60 + "\n")

    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    total_detections = {label: 0 for label in LABELS}
    screenshot_count = 0

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to grab frame")
                break

            # Run detection
            boxes, classes, scores, inference_time = detector.detect(frame)

            # Draw results
            output_frame, num_detections, detection_summary = detector.draw_detections(
                frame.copy(), boxes, classes, scores
            )

            # Update total statistics
            for label, count in detection_summary.items():
                total_detections[label] += count

            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            # Display stats panel
            panel_height = 180
            panel = np.zeros((panel_height, output_frame.shape[1], 3), dtype=np.uint8)

            # Title
            cv2.putText(panel, "FOOD WASTE DETECTION", (10, 30),
                       cv2.FONT_HERSHEY_BOLD, 0.8, (255, 255, 255), 2)

            # Performance stats
            stats_text = f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms | Detections: {num_detections}"
            cv2.putText(panel, stats_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Current frame detections
            y_pos = 90
            cv2.putText(panel, "Current Frame:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 20
            for label in LABELS:
                count = detection_summary[label]
                color = COLORS[label]
                text = f"  {label}: {count}"
                cv2.putText(panel, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 20

            # Total detections (right side)
            y_pos = 90
            cv2.putText(panel, "Total Detected:", (500, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 20
            for label in LABELS:
                count = total_detections[label]
                color = COLORS[label]
                text = f"  {label}: {count}"
                cv2.putText(panel, text, (500, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 20

            # Combine panel and output frame
            final_display = np.vstack([panel, output_frame])

            # Display result
            cv2.imshow('Food Waste Detection - Webcam Test', final_display)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n🛑 Stopping detection...")
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_count += 1
                filename = f"detection_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, final_display)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('r'):
                # Reset statistics
                total_detections = {label: 0 for label in LABELS}
                print("🔄 Statistics reset")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Print final summary
        print("\n" + "=" * 60)
        print("DETECTION SUMMARY")
        print("=" * 60)
        total = sum(total_detections.values())
        print(f"Total detections: {total}")
        for label, count in total_detections.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  • {label}: {count} ({percentage:.1f}%)")
        print("=" * 60)
        print("👋 Goodbye!\n")


if __name__ == "__main__":
    main()
