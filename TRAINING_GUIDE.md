# Food Waste Detector - Complete Training Guide
## SSD-MobileNetV2 with TensorFlow 2 Object Detection API

---

## 📋 Dataset Summary

Your dataset has been converted to TFRecord format:

| Split | Images | File Size |
|-------|--------|-----------|
| Train | 901 | 12.10 MB |
| Valid | 257 | 3.48 MB |
| Test | 98 | 1.4 MB |

**Classes:**
| ID | Class Name | Training Samples |
|----|------------|------------------|
| 1 | Watermelon rind | 327 |
| 2 | Apple core | 181 |
| 3 | fish | 186 |
| 4 | Egg shell | 631 |

---

## 🚀 Step-by-Step Training Instructions

### Step 1: Set Up Your VS Code Project

1. Copy the entire `food_waste_ssd` folder to your local machine
2. Open the folder in VS Code

Your folder structure should look like:
```
food_waste_ssd/
├── data/
│   ├── label_map.pbtxt
│   ├── train.tfrecord
│   ├── valid.tfrecord
│   └── test.tfrecord
├── training/
│   └── pipeline.config
├── scripts/
│   ├── voc_to_tfrecord.py
│   └── verify_tfrecord.py
├── models/
│   └── ssd_mobilenet_v2/
├── exported_model/
└── tflite_model/
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv tf_env

# Activate it
# Windows:
tf_env\Scripts\activate
# Mac/Linux:
source tf_env/bin/activate

# Install TensorFlow 2.15.0 (compatible with Python 3.11 and Object Detection API)
pip install tensorflow==2.15.0

# Note: For WSL2/Linux GPU support, ensure CUDA 12.3 and cuDNN are installed
# GPU detection is automatic once drivers are properly configured
```

### Step 3: Install TensorFlow Object Detection API

```bash
# Clone the TensorFlow models repository
git clone https://github.com/tensorflow/models.git

# Install the Object Detection API
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .

# Verify installation
python -c "from object_detection.utils import label_map_util; print('Success!')"
```

### Step 4: Download Pretrained Model

```bash
# Navigate to models directory
cd food_waste_ssd/models/ssd_mobilenet_v2

# Download pretrained checkpoint
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

# Extract it
tar -xzf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
```

### Step 5: Update Pipeline Config Paths

Edit `training/pipeline.config` and update these paths to match YOUR system:

```
fine_tune_checkpoint: "YOUR_PATH/models/ssd_mobilenet_v2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0"

train_input_reader {
  label_map_path: "YOUR_PATH/data/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "YOUR_PATH/data/train.tfrecord"
  }
}

eval_input_reader {
  label_map_path: "YOUR_PATH/data/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "YOUR_PATH/data/valid.tfrecord"
  }
}
```

### Step 6: Start Training

```bash
# From your project root directory
python models/research/object_detection/model_main_tf2.py \
    --model_dir=training \
    --pipeline_config_path=training/pipeline.config \
    --alsologtostderr
```

**What to Expect:**
- Training will show loss values every 100 steps
- Checkpoints saved every 1000 steps in `training/` folder
- With GPU: ~2-4 hours for 25,000 steps
- Target: total_loss < 1.0

### Step 7: Monitor with TensorBoard

Open a new terminal:
```bash
tensorboard --logdir=training
```
Then open http://localhost:6006 in your browser.

### Step 8: Export the Model

After training completes:
```bash
python models/research/object_detection/exporter_main_v2.py \
    --input_type=image_tensor \
    --pipeline_config_path=training/pipeline.config \
    --trained_checkpoint_dir=training \
    --output_directory=exported_model
```

### Step 9: Convert to TFLite

Create `convert_to_tflite.py`:

```python
import tensorflow as tf

# Load saved model
saved_model_dir = 'exported_model/saved_model'

# Convert with float16 quantization
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save
with open('tflite_model/food_waste_model_float16.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model saved! Size: {len(tflite_model)/1024/1024:.2f} MB")
```

Run it:
```bash
python convert_to_tflite.py
```

---

## 🔧 Training Tips

### If You Get Out of Memory (OOM) Error:
Edit `pipeline.config`:
```
batch_size: 8  # Reduce from 16
```

### If Loss is Not Decreasing:
Edit `pipeline.config`:
```
learning_rate_base: 0.04  # Reduce from 0.08
```

### For Better Accuracy:
Edit `pipeline.config`:
```
num_steps: 40000  # Increase from 25000
```

---

## 📱 Raspberry Pi Deployment

After getting your TFLite model, copy it to your Pi and use this test script:

```python
import tensorflow as tf
import cv2
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path='food_waste_model_float16.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class mapping
CLASS_NAMES = {1: 'Watermelon_rind', 2: 'Apple_core', 3: 'fish', 4: 'Egg_shell'}
ACTIONS = {1: 'TURN LEFT', 2: 'TURN RIGHT', 3: 'MOVE FORWARD', 4: 'MOVE BACKWARD'}

# Capture frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Preprocess
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (320, 320))
input_data = np.expand_dims(image_resized, axis=0).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get results
boxes = interpreter.get_tensor(output_details[0]['index'])
classes = interpreter.get_tensor(output_details[1]['index'])
scores = interpreter.get_tensor(output_details[2]['index'])

# Find best detection
for i in range(len(scores[0])):
    if scores[0][i] > 0.5:
        class_id = int(classes[0][i])
        print(f"Detected: {CLASS_NAMES.get(class_id)} -> Action: {ACTIONS.get(class_id)}")

cap.release()
```

---

## 🎯 Expected Results

After training, you should achieve:
- mAP@0.5: > 0.70
- Per-class precision: > 0.75
- Inference speed on Pi 4: 12-20 FPS

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `data/train.tfrecord` | Training data (901 images) |
| `data/valid.tfrecord` | Validation data (257 images) |
| `data/test.tfrecord` | Test data (98 images) |
| `data/label_map.pbtxt` | Class label mapping |
| `training/pipeline.config` | Model configuration |
| `scripts/voc_to_tfrecord.py` | Conversion script |
| `scripts/verify_tfrecord.py` | Verification script |

---
