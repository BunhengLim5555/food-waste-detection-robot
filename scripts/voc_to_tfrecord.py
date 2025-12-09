"""
Pascal VOC to TFRecord Converter for Food Waste Detection
=========================================================
This script converts Pascal VOC format annotations (from Roboflow) to TFRecord format
for training with TensorFlow Object Detection API.

Usage:
    python voc_to_tfrecord.py --data_dir=data/train --output_path=data/train.tfrecord --label_map_path=data/label_map.pbtxt
"""

import os
import io
import glob
import hashlib
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
import tensorflow as tf


def create_label_map_dict(label_map_path):
    """Read label map and return a dictionary mapping class names to IDs."""
    label_map_dict = {}
    with open(label_map_path, 'r') as f:
        content = f.read()
    
    import re
    items = re.findall(r"item\s*\{[^}]*id:\s*(\d+)[^}]*name:\s*['\"]([^'\"]+)['\"][^}]*\}", content)
    
    for item_id, name in items:
        label_map_dict[name] = int(item_id)
    
    print(f"Label map loaded: {label_map_dict}")
    return label_map_dict


def parse_voc_xml(xml_path):
    """Parse Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image filename
    filename = root.find('filename').text
    
    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get all objects (bounding boxes)
    objects = []
    for obj in root.findall('object'):
        # Try 'name' first, then 'n' (Roboflow uses 'n')
        name_elem = obj.find('name')
        if name_elem is None:
            name_elem = obj.find('n')
        name = name_elem.text if name_elem is not None else None
        if name is None:
            continue
        
        # Handle difficult flag (optional)
        difficult = obj.find('difficult')
        difficult = int(difficult.text) if difficult is not None else 0
        
        # Get bounding box
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'difficult': difficult,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def create_tf_example(annotation, image_dir, label_map_dict):
    """Create a TF Example from a single image and its annotation."""
    
    # Find the image file
    image_filename = annotation['filename']
    
    # Try different image extensions
    image_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        potential_path = os.path.join(image_dir, os.path.splitext(image_filename)[0] + ext)
        if os.path.exists(potential_path):
            image_path = potential_path
            break
    
    # Also try the exact filename
    if image_path is None:
        exact_path = os.path.join(image_dir, image_filename)
        if os.path.exists(exact_path):
            image_path = exact_path
    
    if image_path is None:
        print(f"Warning: Image not found for {image_filename}")
        return None
    
    # Read image
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    
    # Get image format
    image_format = os.path.splitext(image_path)[1].lower().replace('.', '')
    if image_format == 'jpg':
        image_format = 'jpeg'
    
    # Verify image dimensions using PIL
    image = Image.open(io.BytesIO(encoded_image))
    width, height = image.size
    
    # Update dimensions if they differ from XML
    if width != annotation['width'] or height != annotation['height']:
        print(f"Warning: Image dimensions mismatch for {image_filename}. Using actual: {width}x{height}")
    
    # Create lists for TFRecord
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    difficult = []
    
    for obj in annotation['objects']:
        class_name = obj['name']
        
        # Skip if class not in label map
        if class_name not in label_map_dict:
            print(f"Warning: Class '{class_name}' not in label map, skipping...")
            continue
        
        # Normalize coordinates to [0, 1]
        xmins.append(obj['xmin'] / width)
        xmaxs.append(obj['xmax'] / width)
        ymins.append(obj['ymin'] / height)
        ymaxs.append(obj['ymax'] / height)
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])
        difficult.append(obj['difficult'])
    
    # Skip if no valid objects
    if len(classes) == 0:
        print(f"Warning: No valid objects in {image_filename}")
        return None
    
    # Create TF Example
    key = hashlib.sha256(encoded_image).hexdigest()
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_filename.encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult)),
    }))
    
    return tf_example


def convert_voc_to_tfrecord(data_dir, output_path, label_map_path):
    """Convert all Pascal VOC annotations in a directory to TFRecord."""
    
    # Load label map
    label_map_dict = create_label_map_dict(label_map_path)
    
    # Find all XML files
    xml_pattern = os.path.join(data_dir, '*.xml')
    xml_files = glob.glob(xml_pattern)
    
    if len(xml_files) == 0:
        print(f"No XML files found in {data_dir}")
        print("Looking for XML files in subdirectories...")
        xml_pattern = os.path.join(data_dir, '**', '*.xml')
        xml_files = glob.glob(xml_pattern, recursive=True)
    
    print(f"Found {len(xml_files)} XML annotation files")
    
    # Create TFRecord writer
    writer = tf.io.TFRecordWriter(output_path)
    
    success_count = 0
    error_count = 0
    
    for xml_path in xml_files:
        try:
            # Parse XML
            annotation = parse_voc_xml(xml_path)
            
            # Determine image directory (same as XML or parent directory)
            xml_dir = os.path.dirname(xml_path)
            
            # Create TF Example
            tf_example = create_tf_example(annotation, xml_dir, label_map_dict)
            
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            error_count += 1
    
    writer.close()
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {success_count} images")
    print(f"Errors/Skipped: {error_count} images")
    print(f"Output saved to: {output_path}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Convert Pascal VOC to TFRecord')
    parser.add_argument('--data_dir', required=True, help='Directory containing images and XML annotations')
    parser.add_argument('--output_path', required=True, help='Output TFRecord file path')
    parser.add_argument('--label_map_path', required=True, help='Path to label_map.pbtxt file')
    
    args = parser.parse_args()
    
    convert_voc_to_tfrecord(args.data_dir, args.output_path, args.label_map_path)


if __name__ == '__main__':
    main()
