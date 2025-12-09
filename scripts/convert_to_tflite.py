import tensorflow as tf

# Path to exported model (WSL Linux path)
saved_model_dir = "/mnt/c/Users/Bunheng Lim/Downloads/food_waste_ssd_project/food_waste_ssd/food_waste_ssd/exported_model/saved_model"

# Path where you want to save your .tflite
tflite_model_path = "food_waste_model_float16.tflite"

# Convert the model to TFLite (float16 for speed & accuracy balance)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Enable TF Select ops (required for object detection models)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow ops (for StridedSlice, etc.)
]

tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete! Saved as", tflite_model_path)
