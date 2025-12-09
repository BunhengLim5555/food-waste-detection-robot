#!/usr/bin/env python3
"""
Food Waste Detection - Webcam Test (using SavedModel)
Test your trained model on laptop webcam
Detects 4 classes: Watermelon Rind, Egg Shell, Apple Core, Fish Bone
"""

import numpy as np
import cv2
import time
import tensorflow as tf

# Configuration
MODEL_PATH = "../exported_model/saved_model"
LABELS = ["Watermelon rind", "Apple core", "fish", "Egg shell"]

# Class-specific confidence thresholds (tune for each class)
CLASS_THRESHOLDS = {
    "Watermelon rind": 0.5,  # High confidence - works well
    "Apple core": 0.3,        # Lower threshold - needs help
    "fish": 0.5,              # High confidence - works well
    "Egg shell": 0.5          # High confidence - works well
}

# Colors for each class (BGR format)
COLORS = {
    "Watermelon rind": (0, 255, 0),    # Green
    "Apple core": (0, 165, 255),       # Orange
    "fish": (255, 0, 255),             # Magenta
    "Egg shell": (255, 255, 0)         # Cyan
}

class FoodWasteDetector:
    def __init__(self, model_path):
        """Initialize the SavedModel"""
        print(f"Loading model from {model_path}...")
        self.model = tf.saved_model.load(model_path)
        self.detect_fn = self.model.signatures['serving_default']

        print(f"✅ Model loaded successfully!")

    def detect(self, image):
        """Run detection on image"""
        # Resize for faster inference (640x480 is good balance)
        resized = cv2.resize(image, (640, 480))

        # Convert to RGB and uint8
        input_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_image)
        input_tensor = input_tensor[tf.newaxis, ...]

        start_time = time.time()
        detections = self.detect_fn(input_tensor)
        inference_time = (time.time() - start_time) * 1000  # ms

        # Extract detection results
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        return boxes, classes, scores, inference_time

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        ymin1, xmin1, ymax1, xmax1 = box1
        ymin2, xmin2, ymax2, xmax2 = box2

        # Calculate intersection area
        intersect_ymin = max(ymin1, ymin2)
        intersect_xmin = max(xmin1, xmin2)
        intersect_ymax = min(ymax1, ymax2)
        intersect_xmax = min(xmax1, xmax2)

        intersect_width = max(0, intersect_xmax - intersect_xmin)
        intersect_height = max(0, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        # Calculate union area
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        union_area = area1 + area2 - intersect_area

        # Calculate IoU
        if union_area == 0:
            return 0
        return intersect_area / union_area

    def draw_detections(self, image, boxes, classes, scores):
        """Draw bounding boxes and labels on image"""
        height, width = image.shape[:2]

        detections_found = 0
        detection_summary = {label: 0 for label in LABELS}

        # First, collect all valid detections above threshold
        valid_detections = []
        for i in range(len(scores)):
            class_id = int(classes[i]) - 1
            if 0 <= class_id < len(LABELS):
                label = LABELS[class_id]
                threshold = CLASS_THRESHOLDS.get(label, 0.5)

                if scores[i] > threshold:
                    valid_detections.append({
                        'box': boxes[i],
                        'score': scores[i],
                        'class_id': class_id,
                        'label': label,
                        'raw_class_id': int(classes[i])
                    })

        # Sort by PRIORITY: Apple core first (it needs help), then by confidence
        # This ensures Apple core wins even if it has slightly lower confidence
        def detection_priority(det):
            # Apple core gets highest priority (0), others by confidence
            if det['label'] == 'Apple core':
                return (0, -det['score'])  # Sort by priority 0, then by -score (highest first)
            else:
                return (1, -det['score'])  # Sort by priority 1, then by -score

        valid_detections.sort(key=detection_priority)

        # Non-Maximum Suppression: Remove overlapping boxes
        final_detections = []
        IOU_THRESHOLD = 0.3  # Lowered to 30% - more aggressive at removing overlaps

        for detection in valid_detections:
            # Check if this detection overlaps with any already selected detection
            keep = True
            for selected in final_detections:
                iou = self.calculate_iou(detection['box'], selected['box'])
                if iou > IOU_THRESHOLD:
                    # Overlaps too much, don't keep it
                    keep = False
                    break

            if keep:
                final_detections.append(detection)

        # Now draw only the final detections
        for detection in final_detections:
            # Get box coordinates (normalized 0-1)
            ymin, xmin, ymax, xmax = detection['box']

            # Convert to pixel coordinates
            left = int(xmin * width)
            top = int(ymin * height)
            right = int(xmax * width)
            bottom = int(ymax * height)

            # Get class info
            label = detection['label']
            raw_class_id = detection['raw_class_id']
            class_id = detection['class_id']
            color = COLORS.get(label, (255, 255, 255))
            confidence = detection['score']
            threshold = CLASS_THRESHOLDS.get(label, 0.5)

            # Debug: Print raw class ID
            print(f"✅ Final Detection: raw_id={raw_class_id}, array_index={class_id}, label={label}, conf={confidence:.2f}, threshold={threshold}")

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
    print("\nClasses and Confidence Thresholds:")
    for label in LABELS:
        threshold = CLASS_THRESHOLDS.get(label, 0.5)
        print(f"  • {label}: {threshold:.0%}")
    print("=" * 60 + "\n")

    # Initialize detector
    try:
        detector = FoodWasteDetector(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"\nMake sure '{MODEL_PATH}' exists!")
        return

    # Initialize camera
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)

    # Set camera resolution (lower for better FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam!")
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
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

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
