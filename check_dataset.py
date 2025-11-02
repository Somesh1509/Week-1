#!/usr/bin/env python3
"""
Diagnostic script to check your dataset structure
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

def check_dataset():
    print("🔍 Checking dataset structure...")
    
    # Check what's in your data directory
    data_dir = "data/raw"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return
    
    # List all classes (subdirectories)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"📁 Found {len(classes)} classes: {classes}")
    
    # Count images in each class
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"   {class_name}: {len(images)} images")
    
    # Test data generator
    try:
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        print(f"✅ Data generator found {generator.samples} samples")
        print(f"✅ Data generator found {len(generator.class_indices)} classes: {list(generator.class_indices.keys())}")
        
        return list(generator.class_indices.keys())
        
    except Exception as e:
        print(f"❌ Error with data generator: {e}")
        return None

if __name__ == "__main__":
    check_dataset()