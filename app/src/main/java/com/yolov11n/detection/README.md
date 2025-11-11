# YOLOv11n TFLite Android Integration

A complete, production-ready TensorFlow Lite integration for YOLOv11n human detection on Android, implementing the core optimization techniques from mobile research.

## Features

### 🚀 **Performance Optimizations**
- **Hardware Acceleration**: GPU, NNAPI, and Hexagon DSP delegate support with intelligent fallback
- **Memory Management**: Object pooling, buffer reuse, and optimized memory allocation
- **Frame Skipping**: Adaptive frame skipping based on performance monitoring
- **Threading**: Kotlin coroutines with structured concurrency for main-safety
- **Quantization Support**: FP16 and INT8 quantization with automatic delegate optimization

### 🎯 **Human Detection Focused**
- **Single Class**: Optimized for human/person detection only (class ID 0)
- **Configurable Thresholds**: Adjustable confidence and NMS thresholds
- **Real-time Processing**: Target 30 FPS on mid-range devices
- **Efficient Preprocessing**: Letterboxing with aspect ratio preservation

### 📊 **Performance Monitoring**
- **FPS Tracking**: Real-time frame rate monitoring
- **Memory Monitoring**: Heap and native memory usage tracking
- **Latency Analysis**: End-to-end and per-stage latency measurement
- **Energy Monitoring**: Battery level and thermal state tracking
- **Performance Alerts**: Automatic degradation detection

### 🎨 **Optimized Rendering**
- **Hardware Accelerated**: GPU-based rendering for smooth overlays
- **Object Pooling**: Reusable drawing objects to prevent GC pressure
- **Smooth Animations**: Interpolated detection transitions
- **Customizable**: Color schemes, styling, and label configuration

## Architecture

```
TFLiteInterpreter          - Core TFLite wrapper with hardware acceleration
HumanDetectionProcessor    - Preprocessing and postprocessing pipeline  
ModelConfig               - Configuration and optimization settings
PerformanceMonitor        - FPS, memory, and energy monitoring
BoundingBoxRenderer       - Real-time bounding box visualization
```

## Quick Start

### 1. Add Dependencies

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-metadata:0.4.4'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

### 2. Add Model Files to Assets

```
app/src/main/assets/
├── yolov11n_human.tflite      # YOLOv11n model file
├── labels.txt                  # Class labels (optional)
├── yolov11n_config.json       # Model configuration
└── yolov11n_metadata.json     # Model metadata (optional)
```

### 3. Initialize Detection System

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var interpreter: TFLiteInterpreter
    private lateinit var processor: HumanDetectionProcessor  
    private lateinit var monitor: PerformanceMonitor
    private lateinit var renderer: BoundingBoxRenderer
    private lateinit var config: ModelConfig
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize configuration
        config = ModelConfig(this)
        
        // Initialize components
        interpreter = TFLiteInterpreter(this, config)
        processor = HumanDetectionProcessor(config)
        monitor = PerformanceMonitor(this)
        renderer = findViewById(R.id.bounding_box_renderer)
        
        // Setup camera
        setupCamera()
        
        // Start monitoring
        monitor.startMonitoring()
        
        // Initialize interpreter
        lifecycleScope.launch {
            interpreter.initialize().fold(
                onSuccess = { 
                    Log.i("MainActivity", "YOLOv11n initialized successfully")
                    startDetection()
                },
                onFailure = { 
                    Log.e("MainActivity", "Failed to initialize: ${it.message}")
                }
            )
        }
    }
}
```

### 4. Process Camera Frames

```kotlin
private fun startDetection() {
    cameraView.post(object : CameraSurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            // Start camera preview
            startCameraPreview()
            
            // Start detection loop
            lifecycleScope.launch {
                while (isActive) {
                    val bitmap = captureLatestFrame()
                    if (bitmap != null) {
                        processFrame(bitmap)
                    }
                    delay(16) // ~60 FPS
                }
            }
        }
    })
}

private suspend fun processFrame(bitmap: Bitmap) {
    // Check if we should skip frame for performance
    if (processor.shouldSkipFrame()) return
    
    val startTime = System.nanoTime()
    
    // Process frame
    val result = interpreter.processFrame(bitmap)
    result.fold(
        onSuccess = { detectionResult ->
            val detections = detectionResult.detections
            
            // Update renderer
            renderer.setDetections(detections)
            
            // Update performance metrics
            val endTime = System.nanoTime()
            monitor.recordEndToEndLatency(endTime - startTime)
            monitor.recordInferenceLatency(detectionResult.inferenceTimeMs * 1_000_000)
            
            Log.d("Detection", "Found ${detections.size} humans, FPS: ${monitor.getCurrentMetrics().currentFPS}")
        },
        onFailure = { error ->
            Log.w("Detection", "Frame processing failed: ${error.message}")
        }
    )
}
```

## Configuration

### Model Configuration (yolov11n_config.json)

```json
{
  "model_file_name": "yolov11n_human.tflite",
  "input_image_size": 640,
  "confidence_threshold": 0.5,
  "preferred_delegate": "GPU",
  "target_fps": 30,
  "quantization_type": "FP16",
  "enable_memory_optimization": true,
  "frame_skipping_enabled": true
}
```

### Available Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `input_image_size` | int | 640 | Model input resolution (320, 480, 640) |
| `confidence_threshold` | float | 0.5 | Minimum confidence for human detection |
| `preferred_delegate` | string | GPU | Hardware acceleration preference |
| `target_fps` | int | 30 | Target frame rate for optimization |
| `quantization_type` | string | FP16 | Model quantization (FP32, FP16, INT8) |
| `frame_skipping_enabled` | bool | true | Enable adaptive frame skipping |
| `max_detections` | int | 100 | Maximum number of detections per frame |

## Hardware Optimization

### Delegate Selection Strategy

The system automatically selects the best available delegate:

1. **GPU Delegate** (Preferred for high-end devices)
   - Best for Adreno and Mali GPUs
   - Significant speedup for convolution-heavy models
   - Automatic fallback to CPU if unsupported

2. **NNAPI** (Universal Android support)
   - Available on Android 8.1+
   - Access to CPU, GPU, and DSP accelerators
   - Automatic device discovery and partitioning

3. **CPU** (Universal fallback)
   - XNNPACK optimization enabled
   - Thread count optimized for device cores
   - Most compatible option

### Performance Tiers

| Device Performance | Input Size | Delegate | Threads | Expected FPS |
|-------------------|------------|----------|---------|--------------|
| High-end (8+ cores) | 640x640 | GPU | 4 | 30+ |
| Mid-range (4-6 cores) | 480x480 | NNAPI/CPU | 2 | 20-30 |
| Low-end (2-4 cores) | 320x320 | CPU | 1 | 10-20 |

## Performance Monitoring

### Real-time Metrics

```kotlin
// Get current performance metrics
val metrics = monitor.getCurrentMetrics()
Log.i("Performance", """
    FPS: ${metrics.currentFPS}
    Inference: ${metrics.averageInferenceLatencyMs.toInt()}ms  
    Memory: ${metrics.currentMemoryUsageMB}MB
    Battery: ${metrics.batteryLevel}%
    Status: ${if (metrics.isPerformanceDegraded) "DEGRADED" else "NORMAL"}
""".trimIndent())

// Get detailed statistics
val stats = monitor.getPerformanceStatistics()
val avgFPS = stats.fps.average
val p95Latency = stats.latency.inference.p95
```

### Performance Optimization

The system automatically adapts based on monitoring:

- **Frame Skipping**: Increases when FPS drops below target
- **Resolution Scaling**: Reduces input size under memory pressure
- **Delegate Switching**: Falls back to CPU when GPU performance degrades
- **Thermal Management**: Reduces workload when device overheats

## Memory Management

### Built-in Optimizations

- **Object Pooling**: Reusable ByteBuffer and Bitmap objects
- **Buffer Reuse**: Pre-allocated tensors for inference
- **GC Pressure Reduction**: Minimize per-frame allocations
- **Memory Monitoring**: Automatic cleanup on memory pressure

### Memory Limits

```kotlin
// Default memory constraints
val maxHeapMB = 256
val maxBufferPool = 5
val maxDetections = 100
```

## Customization

### Rendering Styles

```kotlin
val customConfig = RenderingConfig(
    showLabels = true,
    showConfidence = true,
    boxStrokeWidth = 3.0f,
    textSize = 16.0f,
    cornerRadius = 6.0f,
    humanBoxColor = Color.CYAN,
    humanTextColor = Color.BLACK,
    enableAnimations = true,
    reuseObjects = true
)

renderer.updateConfiguration(customConfig)
```

### Detection Thresholds

```kotlin
// Runtime threshold updates
config.updateConfiguration(mapOf(
    "confidence_threshold" to 0.7f,
    "nms_threshold" to 0.3f,
    "target_fps" to 20
))
```

## Troubleshooting

### Common Issues

1. **Low FPS**
   - Check delegate compatibility
   - Reduce input image size
   - Enable frame skipping
   - Monitor memory usage

2. **Memory Errors**
   - Increase heap size limit
   - Reduce buffer pool size
   - Check for memory leaks
   - Enable memory optimization

3. **Delegate Failures**
   - Automatic fallback to CPU
   - Check device compatibility
   - Verify model quantization
   - Enable detailed logging

4. **Accuracy Issues**
   - Verify preprocessing matches model requirements
   - Check confidence threshold
   - Ensure proper NMS configuration
   - Validate label mappings

### Debug Logging

```kotlin
// Enable detailed logging
config.updateConfiguration(mapOf(
    "enable_performance_logging" to true,
    "enable_detailed_profiling" to true,
    "debug_output_enabled" to true
))
```

## Advanced Usage

### Custom Model Integration

```kotlin
// For different YOLO variants or custom models
val customConfig = ModelConfig(context, "custom_config.json")
customConfig.updateConfiguration(mapOf(
    "human_class_id" to 2,  // Person class ID
    "max_detections" to 50,
    "input_image_size" to 416
))
```

### Energy-Aware Processing

```kotlin
// Monitor battery level and adapt performance
val batteryLevel = monitor.getCurrentMetrics().batteryLevel
if (batteryLevel < 20) {
    // Low battery - reduce performance
    config.updateConfiguration(mapOf(
        "target_fps" to 15,
        "preferred_delegate" to "CPU",
        "frame_skipping_enabled" to true
    ))
}
```

## Dependencies

- **TensorFlow Lite**: Core inference engine
- **TensorFlow Lite GPU**: GPU acceleration delegate  
- **TensorFlow Lite Support**: Image preprocessing utilities
- **Kotlin Coroutines**: Async processing and threading
- **Android Camera2**: Camera frame capture

## License

This implementation follows the research findings and best practices from mobile optimization studies for real-time YOLO human detection on Android.

## Research Basis

This implementation is based on the comprehensive mobile optimization research documented in:
- Mobile YOLO optimization techniques
- Android hardware acceleration best practices  
- Memory management for real-time inference
- Frame skipping and performance adaptation strategies

For detailed technical background, refer to the research documents in the `docs/` directory.