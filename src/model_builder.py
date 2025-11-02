import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.optimizers import Adam
import yaml
from typing import Dict

class ModelBuilder:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.img_height = self.config['data']['image_height']
        self.img_width = self.config['data']['image_width']
        self.num_classes = self.config['model']['num_classes']
        self.learning_rate = self.config['model']['learning_rate']
    
    def build_custom_cnn(self) -> tf.keras.Model:
        """Build a custom CNN model."""
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_resnet50(self, fine_tune: bool = False) -> tf.keras.Model:
        """Build ResNet50 model with transfer learning."""
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = not fine_tune
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_efficientnet(self, fine_tune: bool = False) -> tf.keras.Model:
        """Build EfficientNetB0 model."""
        
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        base_model.trainable = not fine_tune
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model(self, model_name: str = 'resnet50', fine_tune: bool = False) -> tf.keras.Model:
        """Get model by name."""
        
        model_builders = {
            'custom_cnn': self.build_custom_cnn,
            'resnet50': self.build_resnet50,
            'efficientnet': self.build_efficientnet
        }
        
        if model_name not in model_builders:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(model_builders.keys())}")
        
        return model_builders[model_name](fine_tune)