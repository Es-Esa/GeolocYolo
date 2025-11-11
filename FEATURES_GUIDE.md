# YOLOv11n Human Detection - Features Guide

## 🎯 Complete Feature Documentation

This guide provides detailed information about all features, settings, and capabilities of the YOLOv11n Human Detection Android app.

## 📱 Application Overview

### Core Functionality
The YOLOv11n Human Detection app is a real-time AI-powered mobile application that uses the YOLOv11n (You Only Look Once version 11 nano) model to detect and track humans in camera feeds. The app is optimized for mobile devices and provides real-time performance with minimal battery impact.

### Key Capabilities
- **Real-time Human Detection**: Up to 30 FPS on modern devices
- **Multi-format Model Support**: FP32, FP16, and INT8 quantized models
- **Hardware Acceleration**: GPU, NNAPI, and CPU delegates
- **Adaptive Performance**: Automatic optimization based on device capabilities
- **Battery-Aware Design**: Power management for extended operation
- **Modern UI/UX**: Material Design 3 interface

## 🖥️ User Interface Components

### Main Activity Interface
The primary screen where camera feed and detection results are displayed.

#### Camera Preview Area
- **Location**: Center of screen
- **Function**: Displays real-time camera feed
- **Features**:
  - Full-screen preview with aspect ratio preservation
  - Smooth 30 FPS display (device dependent)
  - Auto-rotation support (portrait/landscape)
  - Zoom gesture support (pinch to zoom)

#### Control Buttons
```
┌─────────────────────────────────────┐
│  [⚙️]  [📊]  [🔄]  [🎯]         │ <- Top control bar
│                                     │
│                                     │
│         Camera Preview              │
│                                     │
│                                     │
│    [Enable Detection] [Switch]      │ <- Bottom control bar
└─────────────────────────────────────┘
```

**Top Control Bar (Left to Right)**:
- **Settings Button (⚙️)**: Access configuration panel
- **Statistics Button (📊)**: View performance metrics
- **Camera Switch Button (🔄)**: Toggle between front/back camera
- **Detection Toggle Button (🎯)**: Enable/disable AI detection

**Bottom Control Bar (Left to Right)**:
- **Detection Status**: Shows current detection state (ON/OFF)
- **Switch Camera**: Quick camera switching
- **Settings Access**: Quick settings menu

#### Status Indicators
- **Detection Status Indicator**: 
  - 🟢 Green: Detection active, humans detected
  - 🟡 Yellow: Detection active, no humans
  - 🔴 Red: Detection error or disabled
  - ⚪ Gray: Camera not active

- **Performance Indicator**:
  - Shows current FPS (Frames Per Second)
  - Updates every 2 seconds
  - Color-coded: Green (>20 FPS), Yellow (10-20 FPS), Red (<10 FPS)

### Settings Activity Interface
Comprehensive configuration panel for customizing detection behavior.

#### Detection Settings Section
```
┌─ Detection Settings ─────────────────┐
│ Confidence Threshold                │
│ ┌─────────────────────────────────┐ │
│ │ 0.50                            │ │ <- Slider: 0.1-0.9
│ └─────────────────────────────────┘ │
│ [Low] ← [Optimal] → [High]          │
│ Controls minimum detection confidence│
│                                     │
│ IoU Threshold                      │
│ ┌─────────────────────────────────┐ │
│ │ 0.50                            │ │ <- Slider: 0.1-0.9
│ └─────────────────────────────────┘ │
│ [Conservative] ← [Balanced] → [Aggressive] │
│ Controls detection overlap tolerance│
└─────────────────────────────────────┘
```

**Confidence Threshold**:
- **Range**: 0.1 to 0.9
- **Default**: 0.5
- **Effect**: Higher values = fewer but more accurate detections
- **Recommended**: 0.3 for crowded scenes, 0.7 for clean detection

**IoU (Intersection over Union) Threshold**:
- **Range**: 0.1 to 0.9
- **Default**: 0.5
- **Effect**: Controls when multiple detections merge
- **Recommended**: 0.3 for multiple people, 0.7 for single person

#### Model Settings Section
```
┌─ Model Settings ────────────────────┐
│ Input Resolution                    │
│ ○ 320x320 (Fastest)                 │
│ ● 416x416 (Balanced)                │
│ ○ 640x640 (Highest Quality)         │
│                                     │
│ Model Type                          │
│ ○ FP32 (Highest Accuracy)          │
│ ● FP16 (Balanced)                   │
│ ○ INT8 (Smallest Size)              │
│                                     │
│ Hardware Delegate                   │
│ ○ CPU (Universal)                  │
│ ● GPU (High-end devices)            │
│ ○ NNAPI (Mid-range devices)         │
└─────────────────────────────────────┘
```

**Input Resolution Options**:
- **320x320**: 
  - Fastest inference (50-100ms)
  - Best for low-end devices
  - Reduced accuracy at distance
  
- **416x416**:
  - Balanced performance (40-60ms)
  - Good for mid-range devices
  - Recommended for most use cases
  
- **640x640**:
  - Highest accuracy (20-30ms)
  - Best for high-end devices
  - Maximum detail detection

**Model Type Options**:
- **FP32 (32-bit Float)**:
  - Highest accuracy
  - Largest model size (~15MB)
  - Universal compatibility
  
- **FP16 (16-bit Float)**:
  - 50% size reduction
  - Minimal accuracy loss
  - GPU accelerated
  
- **INT8 (8-bit Integer)**:
  - 75% size reduction
  - NNAPI accelerated
  - Slight accuracy trade-off

#### Detection Behavior Section
```
┌─ Detection Behavior ────────────────┐
│ ☑ Person Only Mode                   │
│   (Detect humans only, ignore other) │
│                                     │
│ ☑ Show Confidence Scores            │
│   (Display confidence percentage)   │
│                                     │
│ ☑ Smooth Bounding Boxes             │
│   (Reduce box jitter)               │
│                                     │
│ ☑ Adaptive Frame Rate               │
│   (Auto-adjust FPS based on load)   │
└─────────────────────────────────────┘
```

**Person Only Mode**:
- When enabled: Filters to show only human detections
- When disabled: Shows all 80 COCO dataset classes
- Classes filtered when enabled: person (class 0)

**Show Confidence Scores**:
- Displays percentage confidence above each bounding box
- Range: 0-100%
- Color coding: Green (>70%), Yellow (40-70%), Red (<40%)

**Smooth Bounding Boxes**:
- Applies temporal smoothing to box positions
- Reduces visual jitter
- Slight increase in latency (5-10ms)

**Adaptive Frame Rate**:
- Automatically reduces FPS under heavy load
- Maintains smooth operation
- Prevents thermal throttling

### Statistics Activity Interface
Real-time performance monitoring dashboard.

#### Performance Metrics Display
```
┌─ Performance Statistics ────────────┐
│ FPS:                    28.5        │
│ Frame Time:             35ms        │
│ Memory Usage:           145 MB      │
│ CPU Usage:              45%         │
│ Model Load Time:        2.3s        │
│ Detections/Second:      15.2        │
│                                     │
│ Thermal Status:         Cool        │
│ Battery Impact:         Moderate    │
│ Uptime:                 00:15:42    │
│                                     │
│ [Export Data] [Reset Stats]         │
└─────────────────────────────────────┘
```

#### Detailed Metrics
**Frame Rate Metrics**:
- **Current FPS**: Real-time frame rate
- **Average FPS**: Rolling 30-second average
- **Minimum FPS**: Lowest recorded frame rate
- **Frame Time**: Average time per frame in milliseconds

**System Resources**:
- **Memory Usage**: Current app memory consumption
- **CPU Usage**: Percentage of CPU utilization
- **GPU Usage**: Graphics processing usage (if available)
- **Thermal Status**: Device temperature indicator

**Detection Metrics**:
- **Detections/Second**: Number of successful detections
- **Average Confidence**: Mean confidence score of current detections
- **False Positives**: Rate of incorrect detections (estimated)
- **Detection Range**: Maximum effective detection distance

## 🔧 Advanced Features

### Hardware Acceleration Support
The app supports multiple hardware delegates for optimal performance:

#### CPU Delegate
- **Compatibility**: Universal (all devices)
- **Performance**: Baseline performance
- **Usage**: Automatic fallback when other delegates unavailable
- **Optimization**: Multi-threading support

#### GPU Delegate
- **Compatibility**: High-end devices with OpenGL ES 3.0+
- **Performance**: 2-3x faster than CPU
- **Usage**: Automatic selection for flagship devices
- **Models Supported**: FP32, FP16

#### NNAPI Delegate (Neural Networks API)
- **Compatibility**: Android 8.1+ with NNAPI support
- **Performance**: Hardware-specific acceleration
- **Usage**: Automatic selection for supported devices
- **Models Supported**: INT8 quantized models

### Adaptive Performance System
The app includes intelligent performance adaptation:

#### Automatic Model Selection
```kotlin
fun selectOptimalModel(): String {
    return when {
        hasNNAPI() && hasInt8Model() -> "yolov11n_int8.tflite"
        hasGPU() && hasFP16Model() -> "yolov11n_fp16.tflite"
        else -> "yolov11n_fp32.tflite"
    }
}
```

#### Dynamic Input Size Adjustment
- Monitors frame rate performance
- Automatically adjusts input resolution
- Maintains target FPS while optimizing accuracy
- User override available in settings

#### Thermal Management
- Monitors device temperature
- Reduces performance under thermal stress
- Prevents device damage
- Maintains functionality during cooling

### Memory Management System
Advanced memory optimization for stable operation:

#### Buffer Pool Management
- Pre-allocates memory buffers
- Reduces garbage collection pressure
- Prevents memory fragmentation
- Automatic cleanup on app closure

#### Model Caching
- Loads models into memory efficiently
- Supports multiple model variants
- Smart model switching
- Memory pressure detection

#### Frame Buffer Optimization
- Reuses camera frame buffers
- Minimizes memory copy operations
- Efficient image preprocessing
- Smart resolution scaling

## 🎨 Visual Features

### Bounding Box Rendering
Real-time visualization of detection results:

#### Box Appearance
- **Color Coding**:
  - 🟢 Green: High confidence (>70%)
  - 🟡 Yellow: Medium confidence (40-70%)
  - 🔴 Red: Low confidence (<40%)

- **Box Style**:
  - 3px solid border with rounded corners
  - Semi-transparent fill (20% opacity)
  - 1px outer glow for visibility

#### Animation Effects
- **Box Appearance**: Fade-in animation (300ms)
- **Box Updates**: Smooth position interpolation
- **Box Disappearance**: Fade-out animation (200ms)
- **Confidence Changes**: Color transition

#### Information Overlay
Each detection displays:
- **Class Label**: "Person" (or "Human")
- **Confidence Score**: Percentage (e.g., "85%")
- **Detection ID**: Unique identifier for tracking
- **Distance Estimate**: Approximate distance (if available)

### Statistics Visualization
Real-time charts and graphs:

#### Performance Charts
- **FPS Chart**: Line graph showing frame rate over time
- **Memory Usage**: Area chart of memory consumption
- **Detection Rate**: Bar chart of detections per second
- **Thermal Status**: Temperature gauge display

#### Historical Data
- **Session Statistics**: 5-minute rolling window
- **Peak Performance**: Best performance metrics
- **Average Metrics**: Overall session averages
- **Trend Analysis**: Performance improvement/decline

## ⚙️ Configuration Options

### Detection Parameters
Comprehensive control over detection behavior:

#### Threshold Configuration
```kotlin
data class DetectionConfig(
    val confidenceThreshold: Float = 0.5f,
    val iouThreshold: Float = 0.5f,
    val maxDetections: Int = 10,
    val personOnly: Boolean = true
)
```

#### Input Parameters
```kotlin
data class InputConfig(
    val inputSize: Int = 416,
    val inputFormat: String = "RGB",
    val normalizeInput: Boolean = true,
    val aspectRatioMode: AspectRatioMode = AspectRatioMode.CENTER_CROP
)
```

#### Performance Parameters
```kotlin
data class PerformanceConfig(
    val targetFPS: Int = 30,
    val frameSkipStrategy: FrameSkipStrategy = FrameSkipStrategy.ADAPTIVE,
    val maxInferenceTime: Long = 33L, // 33ms for 30 FPS
    val memoryLimit: Long = 300L * 1024L * 1024L // 300MB
)
```

### Model Configuration
Flexible model loading and management:

#### Model File Structure
```
assets/models/
├── yolov11n_fp32.tflite    // Full precision model
├── yolov11n_fp16.tflite    // Half precision model
├── yolov11n_int8.tflite    // Quantized model
└── labels.txt              // Class labels
```

#### Model Loading Options
- **Automatic Selection**: Best model for current device
- **Manual Override**: User-specified model
- **Fallback Strategy**: Multiple model support
- **Update Detection**: Reload models on file changes

#### Preprocessing Configuration
```kotlin
data class PreprocessingConfig(
    val resizeAlgorithm: ResizeAlgorithm = ResizeAlgorithm.LANCZOS,
    val normalize: Boolean = true,
    val mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
    val std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f),
    val dataFormat: DataFormat = DataFormat.NHWC
)
```

## 🔄 Usage Scenarios

### Single Person Detection
**Best Configuration**:
- Confidence Threshold: 0.7
- IoU Threshold: 0.7
- Input Size: 640x640
- Person Only: Enabled

**Expected Results**:
- High accuracy detection
- Stable bounding box
- Low false positives
- 20-30 FPS performance

### Multiple People Detection
**Best Configuration**:
- Confidence Threshold: 0.5
- IoU Threshold: 0.3
- Input Size: 416x416
- Person Only: Enabled

**Expected Results**:
- Multiple person detection
- Reduced box overlap
- Maintained FPS
- 15-25 FPS performance

### Crowd Detection
**Best Configuration**:
- Confidence Threshold: 0.3
- IoU Threshold: 0.2
- Input Size: 416x416
- Person Only: Enabled

**Expected Results**:
- Maximum person detection
- More false positives
- Lower FPS (10-15)
- Dense crowd coverage

### Low Light Detection
**Best Configuration**:
- Confidence Threshold: 0.4
- Input Size: 640x640
- Person Only: Enabled
- Smoothing: Enabled

**Expected Results**:
- Improved low-light sensitivity
- More stable detections
- Higher accuracy
- Moderate FPS (15-20)

### Battery Conservation
**Best Configuration**:
- Confidence Threshold: 0.7
- Input Size: 320x320
- Frame Skip: 2
- Adaptive FPS: Enabled

**Expected Results**:
- Extended battery life
- Reduced heat generation
- Lower performance (8-12 FPS)
- Sustainable operation

## 🛠️ Developer Features

### Logging and Debugging
Comprehensive logging system for development and troubleshooting:

#### Log Levels
```kotlin
enum class LogLevel {
    VERBOSE,    // Detailed debug information
    DEBUG,      // General debug information
    INFO,       // General information
    WARNING,    // Warning messages
    ERROR       // Error messages
}
```

#### Log Categories
- **Camera**: Camera initialization and frame processing
- **Detection**: Model inference and detection results
- **Performance**: Frame rate and memory monitoring
- **UI**: User interface events and interactions
- **Model**: Model loading and management
- **Error**: Error conditions and recovery

#### Debug Features
- **Performance Overlay**: Real-time FPS and memory display
- **Detection Visualization**: Bounding box debug information
- **Model Information**: Current model details and statistics
- **Frame Analysis**: Frame processing timing breakdown

### Testing and Quality Assurance
Built-in testing features for development and validation:

#### Unit Testing Support
```kotlin
@Test
fun testDetectionAccuracy() {
    val detector = YoloDetector()
    val testImage = loadTestImage("single_person.jpg")
    val results = detector.detect(testImage)
    
    assertEquals(1, results.size)
    assertTrue(results[0].confidence > 0.8f)
    assertEquals("Person", results[0].label)
}
```

#### Integration Testing
- Camera pipeline testing
- Model loading validation
- UI interaction testing
- Performance benchmarking

#### Continuous Monitoring
- Real-time performance metrics
- Error rate monitoring
- Memory leak detection
- Battery usage tracking

## 📊 Data and Analytics

### Detection Analytics
Comprehensive tracking of detection performance:

#### Metrics Tracked
- **Detection Count**: Total and per-session detections
- **Confidence Distribution**: Histogram of confidence scores
- **Spatial Distribution**: Where detections occur in frame
- **Temporal Patterns**: Detection frequency over time
- **Performance Correlation**: Accuracy vs. speed analysis

#### Export Options
- **JSON Format**: Structured data export
- **CSV Format**: Spreadsheet-compatible data
- **Real-time Streaming**: Live data to external systems
- **Visualization**: Charts and graphs

### Performance Analytics
Detailed performance monitoring and analysis:

#### Real-time Metrics
- Frame rate monitoring
- Memory usage tracking
- CPU utilization analysis
- Thermal performance monitoring
- Battery consumption measurement

#### Historical Analysis
- Performance trends over time
- Peak performance identification
- Resource utilization patterns
- Bottleneck analysis
- Optimization recommendations

## 🔒 Privacy and Security

### Data Handling
- **Local Processing**: All detection performed on-device
- **No Cloud Upload**: Images never leave the device
- **Temporary Storage**: Frames processed in memory only
- **User Control**: All features user-controllable
- **Permission Management**: Granular permission control

### Security Features
- **Signed Release**: Production builds are digitally signed
- **Obfuscation**: Code protection for release builds
- **Certificate Pinning**: Secure network communication (if applicable)
- **Sandboxed Execution**: App runs in secure Android sandbox
- **Permission Validation**: Runtime permission checks

## 📚 API Reference

### Core Classes

#### YoloDetector
```kotlin
class YoloDetector {
    suspend fun loadModel(modelPath: String): Boolean
    suspend fun detect(image: Image): List<Detection>
    fun getModelInfo(): ModelInfo
    fun setConfiguration(config: DetectionConfig)
    fun getPerformanceStats(): PerformanceStats
}
```

#### Detection
```kotlin
data class Detection(
    val boundingBox: BoundingBox,
    val confidence: Float,
    val label: String,
    val classId: Int,
    val trackingId: Int? = null
)
```

#### BoundingBox
```kotlin
data class BoundingBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    val centerX: Float get() = (left + right) / 2
    val centerY: Float get() = (top + bottom) / 2
    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val area: Float get() = width * height
}
```

### Configuration Classes

#### DetectionConfig
```kotlin
data class DetectionConfig(
    val confidenceThreshold: Float = 0.5f,
    val iouThreshold: Float = 0.5f,
    val maxDetections: Int = 10,
    val personOnly: Boolean = true,
    val inputSize: Int = 416,
    val modelType: ModelType = ModelType.FP16,
    val delegate: DelegateType = DelegateType.AUTO
)
```

#### PerformanceConfig
```kotlin
data class PerformanceConfig(
    val targetFPS: Int = 30,
    val frameSkipStrategy: FrameSkipStrategy = FrameSkipStrategy.ADAPTIVE,
    val enableProfiling: Boolean = false,
    val memoryLimit: Long = 300L * 1024L * 1024L
)
```

## 🎯 Best Practices

### Performance Optimization
- Use appropriate input size for device capabilities
- Enable hardware acceleration when available
- Implement frame skipping for lower-end devices
- Monitor and adapt to thermal conditions
- Use model quantization for better performance

### Battery Life
- Enable adaptive performance modes
- Use smaller input sizes on battery
- Implement smart frame skipping
- Monitor battery level and adjust performance
- Provide power-saving configuration options

### User Experience
- Provide clear visual feedback
- Implement smooth animations
- Use appropriate confidence thresholds
- Offer detailed but not overwhelming statistics
- Maintain consistent UI behavior

### Development
- Use comprehensive logging for debugging
- Implement proper error handling
- Test on various device classes
- Monitor performance metrics continuously
- Document all configuration options

This comprehensive features guide provides complete documentation of all capabilities, settings, and usage scenarios for the YOLOv11n Human Detection Android app, enabling users and developers to fully utilize and understand the application's functionality.