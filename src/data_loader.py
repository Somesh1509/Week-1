import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import yaml

class DataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"❌ Config file not found at {config_path}")
            raise
        
        self.img_height = self.config['data']['image_height']
        self.img_width = self.config['data']['image_width']
        self.batch_size = self.config['data']['batch_size']
        self.validation_split = self.config['data']['validation_split']
        
    def create_data_generators(self):
        """Create training and validation data generators with safe config access"""
        
        # Get augmentation parameters with safe defaults
        augmentation_config = self.config.get('augmentation', {})
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=augmentation_config.get('rotation_range', 20),
            width_shift_range=augmentation_config.get('width_shift_range', 0.2),
            height_shift_range=augmentation_config.get('height_shift_range', 0.2),
            horizontal_flip=augmentation_config.get('horizontal_flip', True),
            zoom_range=augmentation_config.get('zoom_range', 0.2),
            brightness_range=augmentation_config.get('brightness_range', [0.8, 1.2]),
            shear_range=augmentation_config.get('shear_range', 0.1),
            fill_mode=augmentation_config.get('fill_mode', 'nearest'),
            validation_split=self.validation_split
        )
        
        # Data generator for validation (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split
        )
        
        data_dir = self.config['paths']['data_dir']
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"❌ Data directory {data_dir} not found!")
            print("💡 Creating sample directory structure...")
            self._create_sample_directory()
            return None, None
        
        # Create generators
        try:
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
            
            print(f"✅ Training samples: {train_generator.samples}")
            print(f"✅ Validation samples: {validation_generator.samples}")
            print(f"✅ Classes found: {list(train_generator.class_indices.keys())}")
            
            return train_generator, validation_generator
            
        except Exception as e:
            print(f"❌ Error creating data generators: {e}")
            return None, None
    
    def _create_sample_directory(self):
        """Create sample directory structure"""
        data_dir = self.config['paths']['data_dir']
        classes = ['dust', 'snow', 'bird_drop', 'crack', 'healthy']
        
        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            print(f"📁 Created: {class_dir}")
        
        print(f"\n💡 Please add your images to the directories in: {data_dir}")
        print("   Each class should have its own folder with images.")
    
    def get_class_names(self, generator):
        """Get class names from data generator"""
        if generator is None:
            return ['dust', 'snow', 'bird_drop', 'crack', 'healthy']  # Default fallback
        return list(generator.class_indices.keys())