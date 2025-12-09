# Food Waste Detection Robot with SSD MobileNet

Real-time food waste detection system using TensorFlow Object Detection API and Raspberry Pi 4. The robot can detect 4 types of food waste and respond with specific actions.

## Project Overview

This project implements an autonomous robot that:
- Detects 4 food waste classes using a custom-trained SSD MobileNet V2 model
- Runs real-time inference on Raspberry Pi 4 (12-20 FPS)
- Controls robot movement based on detected objects
- Streams live video feed via web interface

### Detected Classes & Robot Actions

| Class | Action | Color |
|-------|--------|-------|
| Watermelon rind | Turn Left | Green |
| Apple core | Turn Right | Orange |
| Fish | Move Forward | Magenta |
| Egg shell | Move Backward | Cyan |

## Features

- **Custom Object Detection Model**: SSD MobileNet V2 trained on food waste dataset
- **Class-Specific Thresholds**: Optimized confidence levels for each class
- **Priority-Based NMS**: Apple core detection gets highest priority
- **PID Steering Control**: Smooth tracking and centering of detected objects
- **Web Streaming**: Real-time video feed accessible via browser
- **Frame Smoothing**: Exponential moving average for stable detections
- **Scan Mode**: Automatic search when target is lost

## Hardware Requirements

- **Raspberry Pi 4** (4GB or 8GB RAM recommended)
- **Pi Camera Module** or USB Webcam
- **Robot Chassis** with motor controller
- **AUPP Robot Board** (or compatible motor driver)
- **Power Supply** for Pi and motors

## Software Requirements

### For Training (PC/Laptop)
- Python 3.11
- TensorFlow 2.15.0
- CUDA 12.2 + cuDNN 9.0 (for GPU training)
- TensorFlow Object Detection API

### For Deployment (Raspberry Pi 4)
- Python 3.9+
- TensorFlow Lite 2.15.0
- OpenCV
- RPi.GPIO
- Flask

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/food_waste_ssd_project.git
cd food_waste_ssd_project/food_waste_ssd/food_waste_ssd
```

### 2. Training Setup (PC/Laptop)

```bash
# Install dependencies
pip install -r requirements.txt

# Install TensorFlow Object Detection API
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

### 3. Raspberry Pi Setup

```bash
# Install dependencies
pip install -r requirements_pi.txt

# Install AUPP Robot library
pip install auppbot

# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
```

## Usage

### Training the Model

1. **Prepare Dataset**: Place images and annotations in `data/` directory
2. **Generate TFRecords**:
   ```bash
   python scripts/create_tfrecords.py
   ```
3. **Start Training**:
   ```bash
   python models/research/object_detection/model_main_tf2.py \
       --model_dir=training/ \
       --pipeline_config_path=training/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config
   ```
4. **Export Model**:
   ```bash
   python models/research/object_detection/exporter_main_v2.py \
       --input_type image_tensor \
       --pipeline_config_path training/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config \
       --trained_checkpoint_dir training/ \
       --output_directory exported_model
   ```

### Convert to TFLite (for Raspberry Pi)

```bash
cd scripts
python convert_to_tflite.py
```

This creates `food_waste_model_float16.tflite` (6.5 MB) optimized for Pi 4.

### Testing on Laptop Webcam

**Option 1: Using SavedModel (Full TensorFlow)**
```bash
cd scripts
python webcam_detection_savedmodel.py
```

**Option 2: Using TFLite Model**
```bash
cd scripts
python webcam_detection.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot
- Press `r` to reset statistics

### Running on Raspberry Pi

1. **Transfer files to Pi**:
   ```bash
   scp food_waste_model_float16.tflite pi@raspberrypi.local:~/
   scp scripts/raspberry_pi_food_waste_robot.py pi@raspberrypi.local:~/
   ```

2. **SSH into Pi and run**:
   ```bash
   ssh pi@raspberrypi.local
   python3 raspberry_pi_food_waste_robot.py
   ```

3. **View web stream**: Open browser and navigate to:
   ```
   http://<raspberry-pi-ip>:5000
   ```

## Project Structure

```
food_waste_ssd/
├── data/
│   ├── train.tfrecord          # Training data
│   ├── valid.tfrecord          # Validation data
│   ├── test.tfrecord           # Test data
│   └── label_map.pbtxt         # Class labels
├── scripts/
│   ├── webcam_detection.py                    # Webcam test (TFLite)
│   ├── webcam_detection_savedmodel.py         # Webcam test (SavedModel)
│   ├── raspberry_pi_food_waste_robot.py       # Main Pi robot script
│   └── convert_to_tflite.py                   # Model conversion
├── training/
│   └── ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config
├── exported_model/
│   └── saved_model/            # Exported TensorFlow model
├── food_waste_model_float16.tflite  # TFLite model (6.5 MB)
├── requirements.txt            # Python dependencies (PC)
├── requirements_pi.txt         # Python dependencies (Pi)
├── TRAINING_GUIDE.md          # Detailed training instructions
└── README.md                  # This file
```

## Configuration

### Raspberry Pi Robot Configuration

Edit `scripts/raspberry_pi_food_waste_robot.py`:

```python
# Camera settings
CAM_INDEX = 0
FRAME_WIDTH = 320
FRAME_HEIGHT = 320
ROTATE_180 = True  # Flip if camera is upside down

# Motor settings
PORT = "/dev/ttyUSB0"
BAUD = 115200
BASE_SPEED = 18
TURN_SPEED = 22

# Detection thresholds
CLASS_THRESHOLDS = {
    "Watermelon rind": 0.5,
    "Apple core": 0.3,      # Lower for harder-to-detect class
    "fish": 0.5,
    "Egg shell": 0.5
}
```

## Performance

- **Training**: ~10,000 steps, 2-3 hours on RTX 3060
- **Inference Speed**:
  - Laptop (CPU): 30-50 FPS
  - Raspberry Pi 4: 12-20 FPS
- **Model Size**: 6.5 MB (TFLite float16)
- **Memory Usage**: ~500 MB on Pi 4

## Troubleshooting

### TFLite Conversion Issues

If you get "SELECT_TF_OPS" errors:
```bash
pip install tensorflow==2.15.0 --force-reinstall
```

### Camera Not Working on Pi

```bash
# Check camera detection
vcgencmd get_camera

# Test camera
raspistill -o test.jpg
```

### Low FPS on Raspberry Pi

1. Reduce frame resolution (320x320 recommended)
2. Use TFLite float16 model (not full TensorFlow)
3. Disable desktop GUI: `sudo raspi-config` > Boot Options > Console

### Robot Not Moving

1. Check motor connections
2. Verify USB port: `ls /dev/ttyUSB*`
3. Test motors: `python -c "from auppbot import AUPPBot; bot = AUPPBot('/dev/ttyUSB0', 115200)"`

## Model Details

- **Architecture**: SSD MobileNet V2 FPNLite 320x320
- **Pretrained**: COCO17 TPU-8
- **Input Size**: 320x320x3
- **Output**: Bounding boxes, class IDs, confidence scores
- **Quantization**: Float16 for speed/accuracy balance

## Future Improvements

- [ ] Add more food waste classes
- [ ] Implement object tracking (SORT/DeepSORT)
- [ ] Add distance estimation using depth camera
- [ ] Implement pick-and-place mechanism
- [ ] Create mobile app for remote control
- [ ] Add voice feedback for detections

## License

This project is for educational purposes.

## Acknowledgments

- TensorFlow Object Detection API
- COCO Dataset (pretrained model)
- AUPP Robotics Lab
- Raspberry Pi Foundation

## Contact

For questions or issues, please open an issue on GitHub.

## Citation

If you use this project, please cite:

```bibtex
@misc{food_waste_detection_robot,
  author = {Your Name},
  title = {Food Waste Detection Robot with SSD MobileNet},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/food_waste_ssd_project}
}
```
