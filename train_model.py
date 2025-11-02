#!/usr/bin/env python3
"""
Training script for Solar Panel Fault Detection
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import ModelTrainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Solar Panel Fault Detection Model')
    parser.add_argument('--model', type=str, default='simple',
                      choices=['simple', 'resnet'],
                      help='Model architecture: simple or resnet')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("🔧 Solar Panel Fault Detection - Training Started")
    print("=" * 50)
    
    try:
        trainer = ModelTrainer()
        history = trainer.train_model(args.model)
        
        if history is not None:
            trainer.plot_training_history()
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your config file and data directory are set up correctly.")