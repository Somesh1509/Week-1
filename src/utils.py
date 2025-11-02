import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess image for model prediction."""
    
    # Load image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0
    
    return image_array

def get_grad_cam(model, processed_image, original_image, layer_name: str = "conv5_block3_out"):
    """Generate Grad-CAM heatmap for model interpretability."""
    
    # Create a model that maps the input image to the activations
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Compute gradient of top predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_image)
        loss = predictions[:, np.argmax(predictions[0])]
    
    # Extract gradients and compute importance weights
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the convolution outputs with the computed gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(
        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0
    )
    
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

def save_training_history(history, filepath: str = "training_history.json"):
    """Save training history to JSON file."""
    
    import json
    
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(value) for value in values]
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=4)

def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 8)):
    """Plot confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return cm