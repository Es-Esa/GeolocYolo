# YOLOv11n Human Detection Android App

A real-time human detection Android application built with YOLOv11n and TensorFlow Lite, optimized for mobile performance and battery efficiency.

## Features

- 🚀 **Real-time Human Detection** - Detect humans in real-time using device camera
- 🎯 **YOLOv11n Model** - Ultra-lightweight, fast detection model optimized for mobile
- ⚡ **Hardware Acceleration** - Support for CPU, GPU, and NNAPI delegates
- 📊 **Performance Monitoring** - Real-time FPS, memory usage, and thermal monitoring
- 🔧 **Customizable Settings** - Adjust confidence thresholds, input sizes, and detection options
- 📱 **Modern UI** - Material Design 3 with dark theme optimized for camera use
- 🔋 **Battery Optimization** - Multiple performance profiles for different power needs
- 📈 **Statistics Dashboard** - Comprehensive performance analytics

## Project Structure

```
android/
├── app/
│   ├── src/main/
│   │   ├── java/com/yolodetection/app/
│   │   │   ├── ui/                    # UI Activities
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── SettingsActivity.kt
│   │   │   │   └── StatisticsActivity.kt
│   │   │   ├── detection/             # YOLO Detection Logic
│   │   │   │   ├── YoloDetector.kt
│   │   │   │   └── models/
│   │   │   │       ├── Detection.kt
│   │   │   │       ├── BoundingBox.kt
│   │   │   │       └── ModelInfo.kt
│   │   │   ├── overlay/               # Visualization
│   │   │   │   └── OverlayView.kt
│   │   │   ├── utils/                 # Utilities
│   │   │   │   ├── Constants.kt
│   │   │   │   ├── FrameProcessor.kt
│   │   │   │   ├── PermissionUtils.kt
│   │   │   │   └── PerformanceManager.kt
│   │   │   └── YoloDetectionApp.kt    # Application class
│   │   ├── res/                       # Resources
│   │   │   ├── layout/                # Layout files
│   │   │   ├── values/                # Colors, strings, themes
│   │   │   └── ...
│   │   └── assets/models/             # Model files
│   │       ├── yolov11n_fp16.tflite
│   │       ├── yolov11n_int8.tflite
│   │       └── labels.txt
│   └── build.gradle
├── build.gradle
└── settings.gradle
```

## Model Information

### YOLOv11n Characteristics
- **Parameters**: 2.6M (ultra-lightweight)
- **FLOPs**: 6.5B (optimized for mobile)
- **Input Size**: 640x640 (configurable: 320, 416, 512, 640, 832, 1024)
- **Classes**: 80 (COCO dataset)
- **Target FPS**: 30 FPS on modern devices
- **Human Detection**: Optimized for person class with high accuracy

### Model Variants
1. **yolov11n_fp16.tflite** (Recommended)
   - Float16 quantization
   - 50% size reduction
   - Minimal accuracy impact
   - Best overall performance

2. **yolov11n_int8.tflite** (For NNAPI)
   - Integer 8-bit quantization
   - 75% size reduction
   - Best for power efficiency
   - Requires good NNAPI support

3. **yolov11n_fp32.tflite** (Reference)
   - Full precision
   - Baseline accuracy
   - Use for debugging

## Hardware Acceleration

### Supported Delegates
- **CPU (XNNPACK)** - Universal fallback, good for small models
- **GPU** - Significant speedup for convolution-heavy workloads
- **NNAPI** - System-level acceleration, device-dependent
- **Hexagon DSP** - Qualcomm-specific low-power acceleration

### Performance Matrix
| Device Type | Expected FPS | Recommended Delegate |
|-------------|--------------|---------------------|
| High-end (Snapdragon 8 Gen 2, A16) | 30-45 | GPU |
| Mid-range (Snapdragon 7 Gen 1) | 20-30 | NNAPI/GPU |
| Low-end (Snapdragon 4 Gen 2) | 10-20 | CPU |
| Older devices | 5-15 | CPU with optimizations |

## Installation & Setup

### Prerequisites
- Android Studio Arctic Fox or later
- Android SDK 24+ (Android 7.0+)
- Android NDK (for native code)
- Gradle 8.0+

### Build Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd android
   ```

2. **Open in Android Studio**
   - Open Android Studio
   - Select "Open an existing Android Studio project"
   - Navigate to the `android` directory

3. **Sync Project**
   - Android Studio will prompt to sync Gradle
   - Click "Sync Now"
   - Wait for synchronization to complete

4. **Build Variants**
   - Debug: Full logging and debugging features
   - Release: Optimized for production with performance monitoring

5. **Run the App**
   - Connect Android device or start emulator
   - Click "Run" or use `Shift+F10`

### Model Asset Setup

1. **Download Models** (if not included)
   - Place model files in `app/src/main/assets/models/`
   - Ensure proper file permissions

2. **Model Conversion** (Optional)
   - Use Ultralytics to convert YOLOv11n to TFLite
   - Apply quantization as needed
   - See model conversion guide

## Configuration

### Key Constants (`Constants.kt`)
```kotlin
// Model configuration
const val MODEL_INPUT_SIZE = 640
const val CONFIDENCE_THRESHOLD = 0.5f
const val IOU_THRESHOLD = 0.5f
const val MAX_DETECTIONS = 300

// Performance settings
const val TARGET_FPS = 30
const val MAX_QUEUE_SIZE = 3
const val THREAD_COUNT = 4
```

### Build Configuration (`build.gradle`)
```gradle
// Enable/disable delegates
buildConfigField "boolean", "USE_GPU", "true"
buildConfigField "boolean", "USE_NNAPI", "true"
```

### Runtime Settings
- **Input Size**: 320x320 to 1024x1024
- **Confidence Threshold**: 0.1 to 0.9
- **IoU Threshold**: 0.1 to 0.9
- **Max Detections**: 10 to 300
- **Performance Profile**: High Performance, Balanced, Battery Saver

## Performance Optimization

### Memory Management
- Automatic garbage collection
- Memory pressure handling
- Buffer reuse
- Bitmap optimization
- Native memory tracking

### Thermal Management
- Thermal throttling detection
- Adaptive performance scaling
- Battery-aware optimization
- Frame skipping under load

### Pipeline Optimization
- YUV to RGB conversion optimization
- Tensor allocation reuse
- Asynchronous processing
- Backpressure handling
- Multi-threading with coroutines

## Monitoring & Analytics

### Real-time Metrics
- **FPS**: Frames per second
- **Inference Time**: Model execution time
- **Memory Usage**: Heap and native memory
- **Thermal State**: Device throttling status
- **Detection Count**: Current objects detected
- **CPU/GPU Usage**: Processing load

### Performance Dashboard
- Historical performance data
- Memory usage graphs
- FPS trend analysis
- Thermal state monitoring
- Battery consumption tracking

## Troubleshooting

### Common Issues

**Camera Permission Denied**
```
Solution: Grant camera permission in device settings
```

**Model Loading Failed**
```
Solution: Check model file exists in assets/
Verify model format and quantization
```

**Low FPS**
```
Solutions:
- Reduce input size (416x416)
- Enable GPU delegate
- Increase frame skipping
- Check thermal throttling
```

**Out of Memory**
```
Solutions:
- Reduce model input size
- Clear application cache
- Enable aggressive memory management
- Check for memory leaks
```

**NNAPI Not Working**
```
Solutions:
- Check device NNAPI support
- Fall back to GPU/CPU
- Verify model compatibility
- Update Android system
```

### Debug Mode
Enable debug logging and performance metrics:
```kotlin
// In debug builds
Timber.d("Debug information")
PerformanceManager.recordFPS(currentFPS)
```

## Development Guidelines

### Code Architecture
- **MVVM Pattern**: Clear separation of concerns
- **Repository Pattern**: Data management abstraction
- **Dependency Injection**: For better testability
- **Coroutines**: Asynchronous programming
- **Flow**: Reactive programming

### Best Practices
- Use ViewBinding instead of findViewById
- Implement proper lifecycle management
- Use safe args for navigation
- Follow Material Design guidelines
- Implement proper error handling
- Add comprehensive logging

### Testing
- Unit tests for business logic
- Integration tests for camera
- UI tests for user interactions
- Performance benchmarking
- Memory leak detection

## Deployment

### Release Checklist
- [ ] Enable ProGuard/R8
- [ ] Remove debug logging
- [ ] Test on multiple devices
- [ ] Validate model accuracy
- [ ] Performance testing
- [ ] Battery usage testing
- [ ] Memory profiling
- [ ] Thermal testing

### Distribution
- Google Play Store
- Direct APK installation
- Enterprise distribution

## Contributing

### Development Setup
1. Follow coding standards
2. Add unit tests
3. Update documentation
4. Test on multiple devices
5. Performance validation

### Code Review Process
- All changes require review
- Performance impact assessment
- Memory usage validation
- Battery consumption testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Android Camera2](https://developer.android.com/training/camera2)
- [Android Performance Guidelines](https://developer.android.com/topic/performance)

## Support

For issues and questions:
- GitHub Issues
- Stack Overflow
- Android Developer Community

---

**Built with ❤️ for real-time human detection on Android**