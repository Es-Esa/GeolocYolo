# YOLOv11n Human Detection Model Assets

This directory contains the TensorFlow Lite models and related files for YOLOv11n human detection.

## Model Files

### yolov11n_fp16.tflite
- **Quantization**: Float16
- **Size**: Reduced by ~50% from FP32
- **Accuracy**: Minimal impact on human detection
- **Use Case**: Recommended for most Android devices
- **Performance**: Good balance of speed and accuracy

### yolov11n_int8.tflite
- **Quantization**: Integer 8-bit
- **Size**: Reduced by ~75% from FP32
- **Accuracy**: Some impact on small person detection
- **Use Case**: For devices with strong NNAPI support
- **Performance**: Best for power efficiency

### yolov11n_fp32.tflite
- **Quantization**: Float32 (baseline)
- **Size**: Full precision model
- **Accuracy**: Reference accuracy
- **Use Case**: For comparison and debugging
- **Performance**: Slowest but most accurate

## Labels

### labels.txt
Contains the class labels for the COCO dataset used by YOLOv11n:
- person
- bicycle
- car
- motorcycle
- airplane
- bus
- train
- truck
- boat
- traffic light
- fire hydrant
- stop sign
- parking meter
- bench
- bird
- cat
- dog
- horse
- sheep
- cow
- ... (80 total classes)

## Model Metadata

### metadata.json
Contains model metadata including:
- Input shape and preprocessing parameters
- Output format and post-processing requirements
- Class names and confidence thresholds
- Performance characteristics

## Usage Instructions

1. **Recommended Model**: Use `yolov11n_fp16.tflite` for best overall performance
2. **Fallback**: If FP16 fails, try `yolov11n_fp32.tflite`
3. **Battery Optimization**: Use `yolov11n_int8.tflite` on NNAPI-capable devices
4. **Model Selection**: The app will automatically select the best model based on device capabilities

## Performance Expectations

### YOLOv11n Characteristics:
- **Parameters**: 2.6M
- **FLOPs**: 6.5B
- **Input Size**: 640x640 (configurable)
- **Classes**: 80 (COCO dataset)
- **Target**: Real-time human detection at 30 FPS

### Expected Performance:
- **Modern High-end Devices**: 30+ FPS with GPU delegate
- **Mid-range Devices**: 15-25 FPS with CPU/NNAPI
- **Low-end Devices**: 8-15 FPS with optimizations

## Model Conversion Notes

These models were converted from YOLOv11n using Ultralytics with the following settings:
- **Input Size**: 640x640
- **Quantization**: Applied during export
- **NMS**: Embedded for simplified post-processing
- **Format**: TensorFlow Lite FlatBuffer

For model conversion instructions, see the main documentation.