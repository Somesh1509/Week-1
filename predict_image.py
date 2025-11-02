#!/usr/bin/env python3
"""
Prediction script for Solar Panel Fault Detection
"""

from src.predict import FaultPredictor
from src.data_loader import DataLoader
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Solar Panel Fault')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image for prediction')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model to use for prediction')
    parser.add_argument('--grad_cam', action='store_true',
                       help='Show Grad-CAM heatmap')
    
    args = parser.parse_args()
    
    # Load class names
    data_loader = DataLoader()
    train_gen, _ = data_loader.create_data_generators()
    class_names = data_loader.get_class_names(train_gen)
    
    # Load model and predict
    model_path = f"models/trained_models/{args.model}_final.h5"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        exit(1)
    
    predictor = FaultPredictor(model_path, class_names)
    predictor.predict_image(args.image, show_grad_cam=args.grad_cam)