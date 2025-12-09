"""
TFRecord Verification Script
============================
Use this to verify your TFRecord files are correctly formatted.
"""

import tensorflow as tf
import os

def count_records(tfrecord_path):
    """Count records in a TFRecord file."""
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_path):
        count += 1
    return count

def inspect_record(tfrecord_path, num_samples=3):
    """Inspect sample records from a TFRecord file."""
    # Updated to use modern TensorFlow API (TF 2.17.0+)
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.RaggedFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.RaggedFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.RaggedFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.RaggedFeature(tf.float32),
        'image/object/class/text': tf.io.RaggedFeature(tf.string),
        'image/object/class/label': tf.io.RaggedFeature(tf.int64),
    }

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    print(f"\n{'='*60}")
    print(f"Inspecting: {tfrecord_path}")
    print(f"{'='*60}")

    for i, raw_record in enumerate(dataset.take(num_samples)):
        example = tf.io.parse_single_example(raw_record, feature_description)

        filename = example['image/filename'].numpy().decode('utf-8')
        height = example['image/height'].numpy()
        width = example['image/width'].numpy()
        labels = example['image/object/class/label'].numpy()
        classes = example['image/object/class/text'].numpy()
        xmin = example['image/object/bbox/xmin'].numpy()
        
        print(f"\nSample {i+1}:")
        print(f"  Filename: {filename}")
        print(f"  Size: {width}x{height}")
        print(f"  Num objects: {len(labels)}")
        print(f"  Classes: {[c.decode('utf-8') for c in classes]}")
        print(f"  Labels: {labels}")

def main():
    data_dir = "data"
    
    print("\n" + "="*60)
    print("TFRecord Verification Report")
    print("="*60)
    
    for split in ['train', 'valid', 'test']:
        tfrecord_path = os.path.join(data_dir, f'{split}.tfrecord')
        if os.path.exists(tfrecord_path):
            count = count_records(tfrecord_path)
            size_mb = os.path.getsize(tfrecord_path) / (1024*1024)
            print(f"\n{split.upper()}: {count} records ({size_mb:.2f} MB)")
            inspect_record(tfrecord_path, num_samples=2)
        else:
            print(f"\n{split.upper()}: File not found!")
    
    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)

if __name__ == '__main__':
    main()
