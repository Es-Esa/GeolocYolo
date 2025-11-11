#!/usr/bin/env python3
"""
YOLOv11n to TFLite Conversion Script for Human Detection
======================================================

This script converts YOLOv11n models to TensorFlow Lite format with optimizations
for human detection on Android devices. Supports multiple quantization modes
(FP16, INT8) and optional non-maximum suppression (NMS) bundling.

Usage:
    python sample_conversion_script.py [OPTIONS]

Examples:
    # Basic FP16 conversion with NMS
    python sample_conversion_script.py --output yolov11n_human_fp16.tflite
    
    # INT8 quantization with calibration
    python sample_conversion_script.py --int8 --calibration-dir ./calibration_images --output yolov11n_human_int8.tflite
    
    # Custom input size and settings
    python sample_conversion_script.py --input-size 416 --confidence 0.3 --iou 0.45 --output custom_yolo.tflite

Author: YOLOv11n Integration Guide
Version: 1.0.0
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
except ImportError as e:
    logger.error("Ultralytics not installed. Please install with: pip install ultralytics")
    raise e

try:
    import tensorflow as tf
except ImportError as e:
    logger.error("TensorFlow not installed. Please install with: pip install tensorflow")
    raise e

class YOLOv11nTFLiteConverter:
    """
    YOLOv11n to TFLite converter with optimization options for human detection.
    """
    
    def __init__(self):
        self.model = None
        self.input_size = 640
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
        self.num_classes = 80  # COCO default
        
    def load_model(self, model_path: str) -> YOLO:
        """
        Load YOLOv11n model from path.
        
        Args:
            model_path: Path to the YOLOv11n model file
            
        Returns:
            Loaded YOLO model
        """
        logger.info(f"Loading YOLOv11n model from: {model_path}")
        self.model = YOLO(model_path)
        logger.info("Model loaded successfully")
        return self.model
    
    def validate_calibration_data(self, calibration_dir: str) -> bool:
        """
        Validate calibration dataset for INT8 quantization.
        
        Args:
            calibration_dir: Directory containing calibration images
            
        Returns:
            True if validation passes
        """
        calibration_path = Path(calibration_dir)
        if not calibration_path.exists():
            logger.warning(f"Calibration directory not found: {calibration_dir}")
            return False
        
        # Check for common image formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = list(calibration_path.glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in supported_formats]
        
        if len(image_files) < 100:
            logger.warning(f"Only {len(image_files)} calibration images found. Recommended: 100-500 images")
        
        if len(image_files) == 0:
            logger.error("No valid image files found in calibration directory")
            return False
        
        logger.info(f"Found {len(image_files)} calibration images")
        return True
    
    def prepare_human_dataset_config(self, calibration_dir: str = None) -> str:
        """
        Create dataset configuration for human-focused training/calibration.
        
        Args:
            calibration_dir: Directory with human images for calibration
            
        Returns:
            Path to created dataset YAML file
        """
        if not calibration_dir or not os.path.exists(calibration_dir):
            # Use default COCO configuration
            return 'coco8.yaml'
        
        config_path = 'human_calibration_dataset.yaml'
        
        calibration_path = Path(calibration_dir)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(calibration_path.glob(ext))
        
        config_content = f"""
# Human-focused calibration dataset configuration
train: {calibration_dir}
val: {calibration_dir}
test: {calibration_dir}

# Focus on human class (class 0 in COCO)
nc: 1  # number of classes
names: ['person']

# Augmentations for better calibration
augment: true
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7    # image HSV-Saturation augmentation (fraction) 
hsv_v: 0.4    # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.2  # image translation (+/- fraction)
scale: 0.5     # image scale (+/- gain)
shear: 0.0     # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0   # image flip up-down (probability)
fliplr: 0.5   # image flip left-right (probability)
mosaic: 0.0   # image mosaic (probability)
mixup: 0.0    # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content.strip())
        
        logger.info(f"Created dataset configuration: {config_path}")
        return config_path
    
    def generate_calibration_dataset(self, source_dir: str, output_dir: str, num_samples: int = 500) -> str:
        """
        Generate a calibration dataset by sampling and preprocessing images.
        
        Args:
            source_dir: Source directory with images
            output_dir: Output directory for calibration samples
            num_samples: Number of samples to include
            
        Returns:
            Path to calibration dataset directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        source_path = Path(source_dir)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(source_path.glob(ext))
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {source_dir}")
        
        # Sample images
        np.random.seed(42)  # For reproducible results
        selected_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        # Preprocess and save images
        logger.info(f"Generating calibration dataset with {len(selected_files)} samples")
        
        for i, img_path in enumerate(selected_files):
            try:
                # Load and preprocess image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Resize to input size while maintaining aspect ratio
                h, w = image.shape[:2]
                scale = self.input_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                
                resized = cv2.resize(image, (new_w, new_h))
                
                # Center crop or pad to square
                if new_h != new_h or new_w != self.input_size:
                    padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
                    y_offset = (self.input_size - new_h) // 2
                    x_offset = (self.input_size - new_w) // 2
                    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                    image = padded
                else:
                    image = resized
                
                # Save preprocessed image
                output_path = Path(output_dir) / f"calib_{i:04d}.jpg"
                cv2.imwrite(str(output_path), image)
                
            except Exception as e:
                logger.warning(f"Failed to process image {img_path}: {e}")
                continue
        
        logger.info(f"Calibration dataset created in: {output_dir}")
        return output_dir
    
    def convert_to_tflite(
        self,
        output_path: str,
        quantization: str = 'fp16',
        calibration_dir: str = None,
        include_nms: bool = True,
        input_size: int = 640,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5
    ) -> str:
        """
        Convert YOLOv11n to TFLite with specified options.
        
        Args:
            output_path: Output TFLite model path
            quantization: 'fp16', 'int8', or 'fp32'
            calibration_dir: Directory with calibration images for INT8
            include_nms: Whether to bundle NMS in the model
            input_size: Input image size for the model
            confidence_threshold: Confidence threshold for NMS
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Path to converted TFLite model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Prepare export arguments
        export_args = {
            'format': 'tflite',
            'imgsz': input_size,
            'half': quantization == 'fp16',
            'nms': include_nms,
            'batch': 1,
            'verbose': True
        }
        
        # Handle INT8 quantization
        if quantization == 'int8':
            export_args['int8'] = True
            
            if calibration_dir:
                if not self.validate_calibration_data(calibration_dir):
                    raise ValueError("Calibration data validation failed")
                
                # Create or use existing calibration dataset config
                dataset_config = self.prepare_human_dataset_config(calibration_dir)
                export_args['data'] = dataset_config
                
                # Generate calibration samples
                calib_dir = self.generate_calibration_dataset(calibration_dir, 'temp_calibration')
                export_args['data'] = calib_dir
            else:
                logger.warning("No calibration directory provided for INT8 quantization. Using default.")
                export_args['data'] = 'coco8.yaml'
        
        logger.info(f"Exporting model with arguments: {export_args}")
        
        try:
            # Export the model
            self.model.export(**export_args)
            
            # The exported model will be in the same directory as the source model
            # with a .tflite extension
            source_model_path = self.model.model_name
            exported_model_path = source_model_path.replace('.pt', '.tflite')
            
            # Move to desired output location
            if exported_model_path != output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                os.rename(exported_model_path, output_path)
            
            logger.info(f"Model successfully converted and saved to: {output_path}")
            
            # Clean up temporary files
            if quantization == 'int8' and 'temp_calibration' in str(export_args.get('data', '')):
                import shutil
                if os.path.exists('temp_calibration'):
                    shutil.rmtree('temp_calibration')
                if os.path.exists('human_calibration_dataset.yaml'):
                    os.remove('human_calibration_dataset.yaml')
            
            return output_path
            
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            raise
    
    def get_model_info(self, tflite_path: str) -> Dict[str, Any]:
        """
        Analyze converted TFLite model and return information.
        
        Args:
            tflite_path: Path to TFLite model
            
        Returns:
            Dictionary with model information
        """
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            # Get input details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Get model size
            model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
            
            # Analyze input tensor
            input_shape = input_details[0]['shape']
            input_type = input_details[0]['dtype']
            
            # Analyze output tensors
            output_shapes = [out['shape'] for out in output_details]
            
            info = {
                'model_size_mb': round(model_size, 2),
                'input_shape': input_shape.tolist(),
                'input_dtype': str(input_type),
                'output_shapes': [shape.tolist() for shape in output_shapes],
                'output_dtypes': [str(out['dtype']) for out in output_details],
                'num_outputs': len(output_details),
                'input_size': self.input_size,
                'quantization_type': 'fp32' if input_type == tf.float32 else 'int8'
            }
            
            logger.info("Model Information:")
            for key, value in info.items():
                logger.info(f"  {key}: {value}")
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to analyze model: {e}")
            return {}
    
    def validate_model(self, tflite_path: str, test_images_dir: str = None) -> bool:
        """
        Basic validation of converted TFLite model.
        
        Args:
            tflite_path: Path to TFLite model
            test_images_dir: Directory with test images (optional)
            
        Returns:
            True if validation passes
        """
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()
            
            # Test with dummy data
            dummy_input = np.random.random(input_details['shape']).astype(input_details['dtype'])
            interpreter.set_tensor(input_details['index'], dummy_input)
            interpreter.invoke()
            
            # Check output
            outputs = []
            for output_detail in output_details:
                output = interpreter.get_tensor(output_detail['index'])
                outputs.append(output)
                logger.debug(f"Output shape: {output.shape}, dtype: {output.dtype}")
            
            # Test with real images if provided
            if test_images_dir and os.path.exists(test_images_dir):
                self._test_with_images(interpreter, input_details, test_images_dir)
            
            logger.info("Model validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _test_with_images(self, interpreter, input_details, test_images_dir: str):
        """Test model with real images."""
        test_files = list(Path(test_images_dir).glob('*'))
        test_files = [f for f in test_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        
        if len(test_files) == 0:
            logger.warning("No test images found for validation")
            return
        
        logger.info(f"Testing model with {len(test_files)} images")
        
        for i, img_path in enumerate(test_files[:5]):  # Test first 5 images
            try:
                # Load and preprocess image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Resize to model input size
                resized = cv2.resize(image, (self.input_size, self.input_size))
                
                # Normalize to [0, 1]
                normalized = resized.astype(np.float32) / 255.0
                
                # Add batch dimension
                batch_input = np.expand_dims(normalized, axis=0)
                
                # Run inference
                interpreter.set_tensor(input_details['index'], batch_input)
                interpreter.invoke()
                
                # Check outputs
                for j, output_detail in enumerate(interpreter.get_output_details()):
                    output = interpreter.get_tensor(output_detail['index'])
                    logger.debug(f"Image {i+1}, Output {j+1}: shape={output.shape}, mean={np.mean(output):.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to test image {img_path}: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert YOLOv11n to TFLite for human detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        'model_path',
        help='Path to YOLOv11n model (.pt file)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output TFLite model path'
    )
    
    # Model options
    parser.add_argument(
        '--input-size', '-s',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    
    parser.add_argument(
        '--quantization', '-q',
        choices=['fp32', 'fp16', 'int8'],
        default='fp16',
        help='Quantization type (default: fp16)'
    )
    
    parser.add_argument(
        '--calibration-dir', '-c',
        help='Directory with calibration images for INT8 quantization'
    )
    
    parser.add_argument(
        '--confidence', '-conf',
        type=float,
        default=0.5,
        help='Confidence threshold for NMS (default: 0.5)'
    )
    
    parser.add_argument(
        '--iou', '-iou',
        type=float,
        default=0.5,
        help='IoU threshold for NMS (default: 0.5)'
    )
    
    parser.add_argument(
        '--no-nms',
        action='store_true',
        help='Do not bundle NMS in the exported model'
    )
    
    # Testing options
    parser.add_argument(
        '--test-images-dir',
        help='Directory with test images for validation'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the existing TFLite model (skip conversion)'
    )
    
    # Output options
    parser.add_argument(
        '--save-info',
        help='Save model information to JSON file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize converter
    converter = YOLOv11nTFLiteConverter()
    
    if args.validate_only:
        if not os.path.exists(args.model_path):
            logger.error(f"Model file not found: {args.model_path}")
            return 1
        
        logger.info("Validating TFLite model...")
        success = converter.validate_model(args.model_path, args.test_images_dir)
        return 0 if success else 1
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    try:
        # Load model
        converter.load_model(args.model_path)
        
        # Convert model
        tflite_path = converter.convert_to_tflite(
            output_path=args.output,
            quantization=args.quantization,
            calibration_dir=args.calibration_dir,
            include_nms=not args.no_nms,
            input_size=args.input_size,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou
        )
        
        # Get model information
        model_info = converter.get_model_info(tflite_path)
        
        # Save model info if requested
        if args.save_info:
            with open(args.save_info, 'w') as f:
                json.dump(model_info, f, indent=2)
            logger.info(f"Model information saved to: {args.save_info}")
        
        # Validate model
        logger.info("Validating converted model...")
        if converter.validate_model(tflite_path, args.test_images_dir):
            logger.info("Conversion and validation completed successfully!")
            return 0
        else:
            logger.error("Validation failed")
            return 1
            
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())