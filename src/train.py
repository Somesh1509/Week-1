import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"❌ Config file not found at {config_path}")
            raise
        
        # Auto-detect number of classes
        self.num_classes = self._detect_num_classes()
        print(f"🎯 Auto-detected {self.num_classes} classes")
        
        self.img_height = self.config['data']['image_height']
        self.img_width = self.config['data']['image_width']
        self.batch_size = self.config['data']['batch_size']
        self.model = None
        self.history = None
    
    def _detect_num_classes(self):
        """Automatically detect number of classes from data directory"""
        data_dir = self.config['paths']['data_dir']
        
        if not os.path.exists(data_dir):
            print(f"⚠️  Data directory {data_dir} not found. Using default 5 classes.")
            return 5
        
        # Count subdirectories (each is a class)
        classes = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        
        num_classes = len(classes)
        print(f"📁 Found {num_classes} classes: {classes}")
        return num_classes
    
    def create_data_generators(self):
        """Create data generators with auto-detected classes"""
        data_dir = self.config['paths']['data_dir']
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} not found")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        # Data generator for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Update num_classes based on actual data
        actual_classes = len(train_generator.class_indices)
        if actual_classes != self.num_classes:
            print(f"🔄 Updating num_classes from {self.num_classes} to {actual_classes}")
            self.num_classes = actual_classes
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {validation_generator.samples}")
        print(f"✅ Classes: {list(train_generator.class_indices.keys())}")
        
        return train_generator, validation_generator
    
    def create_model(self, model_type='simple'):
        """Create model with correct number of output classes"""
        
        if model_type == 'simple':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                      input_shape=(self.img_height, self.img_width, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])
        elif model_type == 'resnet':
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            base_model.trainable = False
            
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Created {model_type} model with {self.num_classes} output classes")
        return model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        os.makedirs(self.config['paths']['model_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.config['paths']['model_dir'], 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, model_type='simple', fine_tune=False):
        """Train the model"""
        print(f"🚀 Starting {model_type} model training...")
        print(f"📊 Number of classes: {self.num_classes}")
        
        try:
            # Create data generators
            train_generator, validation_generator = self.create_data_generators()
            
            # Create model
            self.model = self.create_model(model_type)
            
            # Setup callbacks
            callbacks = self.setup_callbacks()
            
            print("\n📈 Starting training...")
            self.history = self.model.fit(
                train_generator,
                epochs=self.config['model']['epochs'],
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save final model
            model_path = os.path.join(self.config['paths']['model_dir'], 'final_model.h5')
            self.model.save(model_path)
            print(f"✅ Model saved to {model_path}")
            
            return self.history
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            print("💡 Creating demo with synthetic data...")
            return self._train_with_demo_data()
    
    def _train_with_demo_data(self):
        """Train with synthetic data for demonstration"""
        print("🎭 Training with synthetic data for demonstration...")
        
        # Create synthetic data matching the detected number of classes
        num_samples = 200
        X_train = np.random.random((num_samples, self.img_height, self.img_width, 3))
        y_train = tf.keras.utils.to_categorical(
            np.random.randint(0, self.num_classes, num_samples), 
            num_classes=self.num_classes
        )
        
        # Create model
        self.model = self.create_model('simple')
        
        # Train on synthetic data
        self.history = self.model.fit(
            X_train, y_train,
            epochs=5,
            validation_split=0.2,
            verbose=1,
            batch_size=32
        )
        
        print("✅ Demo training completed!")
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available.")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function"""
    print("🔧 Solar Panel Fault Detection - Training")
    trainer = ModelTrainer()
    trainer.train_model('simple')
    trainer.plot_training_history()

if __name__ == "__main__":
    main()