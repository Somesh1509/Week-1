#!/usr/bin/env python3
"""
SOLAR PANEL FAULT DETECTION - COMPLETE WORKING VERSION
"""
import os
import sys
import argparse
import glob
sys.path.append('src')

def main():
    parser = argparse.ArgumentParser(description='Solar Panel Fault Detection')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict', 'demo', 'setup', 'list-images'],
                       help='Mode: train, predict, demo, setup, or list-images')
    parser.add_argument('--image', type=str, help='Path to image file for prediction')
    
    args = parser.parse_args()
    
    print("🔧 SOLAR PANEL FAULT DETECTION SYSTEM")
    print("=" * 60)
    
    try:
        if args.mode == 'train':
            train_model()
            
        elif args.mode == 'predict':
            if not args.image:
                print("❌ ERROR: Please provide an image file with --image")
                print("💡 Example: python main.py --mode predict --image my_image.jpg")
                print("💡 Or use: python main.py --mode list-images to see available images")
                return
            predict_image(args.image)
            
        elif args.mode == 'demo':
            run_demo()
            
        elif args.mode == 'setup':
            check_project_setup()
            
        elif args.mode == 'list-images':
            list_available_images()
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def train_model():
    """Train the solar panel fault detection model"""
    from train import ModelTrainer
    print("🚀 Starting training process...")
    trainer = ModelTrainer()
    trainer.train_model()
    
    if trainer.history is not None:
        trainer.plot_training_history()
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        
        # Show where model was saved
        model_path = 'models/trained_models/solar_fault_model.h5'
        if os.path.exists(model_path):
            print(f"💾 Model saved: {model_path}")
    else:
        print("⚠️ Training finished but no real data was used.")

def predict_image(image_path):
    """Predict fault type for a single image"""
    try:
        from tensorflow.keras.preprocessing import image
        import numpy as np
        import tensorflow as tf
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ ERROR: File not found: {image_path}")
            print("💡 Available images:")
            list_available_images()
            return
        
        # Check if model exists
        model_path = 'models/trained_models/solar_fault_model.h5'
        if not os.path.exists(model_path):
            print("❌ No trained model found.")
            print("💡 Please train the model first:")
            print("   python main.py --mode train")
            return
        
        print(f"📷 Loading image: {image_path}")
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print("🤖 Loading trained model...")
        model = tf.keras.models.load_model(model_path)
        
        print("🔮 Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        
        # Define class names
        class_names = ['Dust', 'Snow', 'Bird Droppings', 'Crack', 'Healthy']
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = np.max(predictions[0])
        
        print("\n" + "=" * 50)
        print("🎯 PREDICTION RESULTS")
        print("=" * 50)
        print(f"✅ Fault Type: {predicted_class}")
        print(f"📊 Confidence: {confidence:.2%}")
        print("\n📈 All probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
            print(f"   {class_name}: {prob:.2%}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")

def run_demo():
    """Run demo with synthetic data"""
    from train import ModelTrainer
    print("🎭 Running demo with synthetic data...")
    trainer = ModelTrainer()
    trainer._train_with_demo_data()
    trainer.plot_training_history()

def list_available_images():
    """List all available images in the project"""
    print("📸 AVAILABLE IMAGES:")
    print("=" * 40)
    
    # Check data directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    found_images = False
    
    for root, dirs, files in os.walk('data'):
        for file in files:
            if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                print(f"📁 {os.path.join(root, file)}")
                found_images = True
    
    if not found_images:
        print("❌ No images found in data directory")
        print("💡 You can:")
        print("   1. Add images to data/raw/ folders")
        print("   2. Use any image file: python main.py --mode predict --image path/to/your/image.jpg")
        print("   3. Run demo: python main.py --mode demo")

def check_project_setup():
    """Check project structure"""
    print("🔍 PROJECT SETUP CHECK")
    print("=" * 50)
    
    required = {
        'Files': ['main.py', 'train_model.py', 'requirements.txt', 'config/config.yaml'],
        'Directories': ['src/', 'data/raw/', 'models/']
    }
    
    all_ok = True
    
    for category, items in required.items():
        print(f"\n📁 {category}:")
        for item in items:
            if os.path.exists(item):
                print(f"   ✅ {item}")
            else:
                print(f"   ❌ {item}")
                all_ok = False
    
    print("\n" + "=" * 50)
    if all_ok:
        print("🎉 PROJECT SETUP IS COMPLETE!")
    else:
        print("⚠️ Some files/directories are missing")

if __name__ == "__main__":
    main()