# Model Comparison Suite
# Different YOLO variants for testing human detection on Android

## Available Model Variants

### 1. YOLOv11n (Nano) - Ultra Lightweight
- **File**: `yolov11n.tflite`
- **Size**: ~6MB
- **Input Size**: 320x320
- **Target**: Low-end devices, real-time performance
- **Use Case**: Basic human detection, resource-constrained environments

### 2. YOLOv11s (Small) - Balanced Performance
- **File**: `yolov11s.tflite`
- **Size**: ~14MB
- **Input Size**: 416x416
- **Target**: Mid-range devices, balance of speed and accuracy
- **Use Case**: General purpose human detection, mobile apps

### 3. YOLOv11m (Medium) - High Accuracy
- **File**: `yolov11m.tflite`
- **Size**: ~25MB
- **Input Size**: 416x416
- **Target**: High-end devices, accuracy focused
- **Use Case**: Professional applications, detailed analysis

### 4. YOLOv11l (Large) - Maximum Accuracy
- **File**: `yolov11l.tflite`
- **Size**: ~45MB
- **Input Size**: 512x512
- **Target**: Flagship devices, cloud processing
- **Use Case**: Research, high-precision applications

## Model Configuration

### Model Metadata
```json
{
  "model_variants": {
    "yolov11n": {
      "model_file": "yolov11n.tflite",
      "input_size": [320, 320, 3],
      "output_names": ["output0"],
      "confidence_threshold": 0.4,
      "nms_threshold": 0.5,
      "max_detections": 50,
      "target_classes": ["person"],
      "optimization": "int8"
    },
    "yolov11s": {
      "model_file": "yolov11s.tflite", 
      "input_size": [416, 416, 3],
      "output_names": ["output0"],
      "confidence_threshold": 0.3,
      "nms_threshold": 0.4,
      "max_detections": 100,
      "target_classes": ["person"],
      "optimization": "fp16"
    },
    "yolov11m": {
      "model_file": "yolov11m.tflite",
      "input_size": [416, 416, 3], 
      "output_names": ["output0"],
      "confidence_threshold": 0.25,
      "nms_threshold": 0.4,
      "max_detections": 200,
      "target_classes": ["person"],
      "optimization": "fp32"
    },
    "yolov11l": {
      "model_file": "yolov11l.tflite",
      "input_size": [512, 512, 3],
      "output_names": ["output0"],
      "confidence_threshold": 0.2,
      "nms_threshold": 0.3,
      "max_detections": 300,
      "target_classes": ["person"],
      "optimization": "fp32"
    }
  }
}
```

## Performance Expectations

| Model | Model Size | Inference Time | Battery Impact | Memory Usage | Accuracy (mAP) |
|-------|------------|----------------|----------------|--------------|----------------|
| YOLOv11n | ~6MB | 20-30ms | Low | 50-100MB | 0.65 |
| YOLOv11s | ~14MB | 40-60ms | Medium | 100-200MB | 0.75 |
| YOLOv11m | ~25MB | 80-120ms | High | 200-400MB | 0.85 |
| YOLOv11l | ~45MB | 150-200ms | Very High | 400-800MB | 0.90 |

## Testing Matrix

### Device Classes
- **Low-end**: 2GB RAM, quad-core, Android 8+
- **Mid-range**: 4GB RAM, octa-core, Android 9+
- **High-end**: 8GB RAM, flagship processor, Android 10+

### Test Scenarios
1. **Single Person Detection**
2. **Multiple People Detection (2-5 people)**
3. **Crowd Detection (6+ people)**
4. **Low Light Conditions**
5. **Fast Motion Detection**
6. **Occlusion Handling**

### Metrics to Compare
- **Speed**: FPS, inference time per image
- **Accuracy**: Precision, recall, mAP
- **Efficiency**: Battery drain, memory usage
- **Robustness**: Performance across different conditions

## Usage Instructions

1. **Select appropriate model** based on target device capabilities
2. **Run comparison benchmarks** using performance test suite
3. **Evaluate trade-offs** between speed and accuracy
4. **Choose optimal configuration** for your use case

## Optimization Tips

- Use INT8 quantization for smaller models
- Consider YOLOv11n for real-time applications
- Use larger models for static image analysis
- Test on actual target devices for accurate results