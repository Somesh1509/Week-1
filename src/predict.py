import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from .utils import preprocess_image, get_grad_cam
import matplotlib.pyplot as plt

class FaultPredictor:
    def __init__(self, model_path: str, class_names: list):
        self.model = load_model(model_path)
        self.class_names = class_names
        
    def predict_image(self, image_path: str, show_grad_cam: bool = False):
        """Predict fault type for a single image."""
        
        # Preprocess image
        processed_image = preprocess_image(image_path, target_size=(224, 224))
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = np.max(predictions[0])
        
        # Display results
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2 if show_grad_cam else 1, figsize=(12, 5))
        
        if show_grad_cam:
            # Generate Grad-CAM
            gradcam = get_grad_cam(self.model, processed_image, original_image)
            
            axes[0].imshow(original_image)
            axes[0].set_title(f'Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(gradcam)
            axes[1].set_title(f'Grad-CAM Heatmap')
            axes[1].axis('off')
        else:
            axes.imshow(original_image)
            axes.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}')
            axes.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Predicted Fault: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print("\nAll predictions:")
        for i, (class_name, prob) in enumerate(zip(self.class_names, predictions[0])):
            print(f"  {class_name}: {prob:.2%}")
        
        return predicted_class, confidence
    
    def predict_batch(self, image_dir: str):
        """Predict fault types for all images in a directory."""
        
        results = []
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            processed_image = preprocess_image(image_path)
            
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = np.max(predictions[0])
            
            results.append({
                'image': image_file,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': predictions[0]
            })
            
            print(f"{image_file}: {predicted_class} ({confidence:.2%})")
        
        return results