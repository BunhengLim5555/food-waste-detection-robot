import cv2
import tensorflow as tf
import numpy as np

# Load model
detect_fn = tf.saved_model.load(r"C:\Users\Bunheng Lim\Downloads\food_waste_ssd_project\food_waste_ssd\food_waste_ssd\exported_model\saved_model")

# Initialize camera (try 0 first, then 1 or 2 if you have multiple)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)

    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:  # confidence threshold
            y1, x1, y2, x2 = boxes[i]
            y1, x1, y2, x2 = int(y1*h), int(x1*w), int(y2*h), int(x2*w)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {classes[i]}: {scores[i]:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
