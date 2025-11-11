# YOLOv11n TFLite Model Integration Guide

## Overview

This guide provides a comprehensive, step-by-step approach to integrate YOLOv11n TFLite models into Android applications for human detection. YOLOv11n is Ultralytics' smallest and most efficient detection variant, optimized for edge and mobile deployment with approximately 2.6M parameters.

## Prerequisites

- Android Studio 4.0 or higher
- Android SDK API 24+ (Android 7.0+)
- Basic knowledge of Kotlin and TensorFlow Lite
- Python environment for model conversion (if needed)

## Model Export from YOLOv11n to TFLite

### Step 1: Install Required Dependencies

```bash
pip install ultralytics
```

### Step 2: Direct Export to TFLite (Recommended)

Use Ultralytics' built-in export functionality for the most straightforward path:

```python
from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO('yolo11n.pt')  # or yolo11n.pt for human detection

# Export to TFLite with optimized settings for mobile
model.export(
    format='tflite',           # TFLite format
    imgsz=640,                 # Input size (balance accuracy/latency)
    half=True,                 # FP16 quantization
    nms=True,                  # Bundle NMS in model
    int8=False,                # Use False for FP16, True for INT8
    batch=1,                   # Single frame for real-time
    data='person_dataset.yaml' # Required for INT8 calibration
)
```

### Step 3: Export with INT8 Quantization

For maximum performance and smallest model size:

```python
model.export(
    format='tflite',
    imgsz=416,                 # Smaller input for speed
    half=False,                # No FP16
    nms=True,
    int8=True,                 # Full integer quantization
    batch=1,
    data='representative_dataset.yaml'  # Critical for INT8
)
```

## Model Integration in Android

### Step 1: Add Dependencies to build.gradle

```gradle
dependencies {
    // TensorFlow Lite Core
    implementation 'org.tensorflow:tensorflow-lite:0.4.4'
    
    // TensorFlow Lite Support Library
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    
    // TensorFlow Lite Task Vision Library (for easy preprocessing)
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.4'
    
    // CameraX for camera integration
    implementation 'androidx.camera:camera-core:1.2.2'
    implementation 'androidx.camera:camera-camera2:1.2.2'
    implementation 'androidx.camera:camera-lifecycle:1.2.2'
    implementation 'androidx.camera:camera-view:1.2.2'
    
    // Kotlin Coroutines for async processing
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

### Step 2: Add Model to Assets

1. Create an `assets/models` directory in your app module
2. Copy your exported `.tflite` file to this directory
3. Ensure the model file is named appropriately (e.g., `yolov11n_human.tflite`)

### Step 3: Create TFLite Model Wrapper Class

```kotlin
class YOLOv11nHumanDetector(
    assetManager: AssetManager,
    modelFileName: String = "yolov11n_human.tflite",
    numThreads: Int = 4
) {
    
    private val options = Interpreter.Options().apply {
        setNumThreads(numThreads)
        setUseXNNPack(true)
        // Enable GPU delegate (optional, test on your device)
        // addDelegate(GpuDelegate())
    }
    
    private val model: Interpreter by lazy {
        val buffer = readModelBuffer(assetManager, modelFileName)
        Interpreter(buffer, options)
    }
    
    private fun readModelBuffer(assetManager: AssetManager, fileName: String): ByteBuffer {
        assetManager.open("models/$fileName").use { inputStream ->
            val bytes = inputStream.readBytes()
            val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            buffer.put(bytes)
            buffer.flip()
            return buffer
        }
    }
    
    @OptIn(ExperimentalGetImage::class)
    fun detectHumans(image: Image, imageRotation: Int): List<DetectionResult> {
        // Preprocess image
        val inputBuffer = preprocessImage(image, imageRotation, 640, 640)
        
        // Prepare output buffer
        val outputBuffer = Array(1) { Array(300) { FloatArray(6) } }
        model.run(inputBuffer, outputBuffer)
        
        // Postprocess results
        return postprocessResults(outputBuffer[0])
    }
    
    private fun preprocessImage(
        image: Image, 
        rotation: Int, 
        targetWidth: Int, 
        targetHeight: Int
    ): ByteBuffer {
        val yuvToRgbConverter = YuvToRgbConverter()
        val bitmap = Bitmap.createBitmap(
            image.width, 
            image.height, 
            Bitmap.Config.ARGB_8888
        )
        yuvToRgbConverter.yuvToRgb(image, bitmap)
        
        // Resize and normalize
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        val buffer = ByteBuffer.allocateDirect(1 * targetHeight * targetWidth * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())
        
        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val pixel = resizedBitmap.getPixel(x, y)
                // Normalize to [0, 1]
                buffer.putFloat(((pixel shr 16 and 0xFF) / 255.0f))
                buffer.putFloat(((pixel shr 8 and 0xFF) / 255.0f))
                buffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }
        buffer.rewind()
        return buffer
    }
    
    private fun postprocessResults(rawResults: Array<FloatArray>): List<DetectionResult> {
        val detections = mutableListOf<DetectionResult>()
        
        for (result in rawResults) {
            if (result[4] > 0.5f) { // Confidence threshold
                detections.add(
                    DetectionResult(
                        boundingBox = RectF(
                            result[0], result[1], 
                            result[2], result[3]
                        ),
                        confidence = result[4],
                        classId = result[5].toInt()
                    )
                )
            }
        }
        
        return applyNMS(detections) // Non-Maximum Suppression
    }
    
    private fun applyNMS(detections: List<DetectionResult>, iouThreshold: Float = 0.5f): List<DetectionResult> {
        // Implement NMS logic or use the bundled NMS from export
        return detections // Return if NMS is bundled in model
    }
    
    fun close() {
        model.close()
    }
}

data class DetectionResult(
    val boundingBox: RectF,
    val confidence: Float,
    val classId: Int
)
```

### Step 4: Camera2 Integration

```kotlin
class HumanDetectionActivity : AppCompatActivity() {
    
    private lateinit var cameraExecutor: CameraExecutor
    private lateinit var yoloDetector: YOLOv11nHumanDetector
    private lateinit var detectionOverlay: DetectionOverlayView
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection)
        
        detectionOverlay = findViewById(R.id.overlay)
        
        // Initialize detector
        yoloDetector = YOLOv11nHumanDetector(assets)
        
        // Setup camera
        cameraExecutor = CameraExecutor(this)
        cameraExecutor.startCamera { image, rotation ->
            detectHumans(image, rotation)
        }
    }
    
    private fun detectHumans(image: Image, rotation: Int) {
        lifecycleScope.launch {
            val detections = yoloDetector.detectHumans(image, rotation)
            detectionOverlay.setDetections(detections)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        yoloDetector.close()
    }
}

class CameraExecutor(
    private val activity: HumanDetectionActivity,
    private val onFrame: (Image, Int) -> Unit
) {
    
    private val cameraExecutor = ContextCompat.getMainExecutor(activity)
    
    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(activity)
        
        cameraProviderFuture.addListener(cameraExecutor) {
            val cameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(Surface.ROTATION_0)
                .build()
                .also {
                    it.setSurfaceProvider(findViewById<PreviewView>(R.id.viewFinder).surfaceProvider)
                }
            
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        imageProxy.image?.let { image ->
                            onFrame(image, imageProxy.imageInfo.rotationDegrees)
                        }
                    }
                }
            
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                activity,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalyzer
            )
        }
    }
}
```

### Step 5: Detection Overlay View

```kotlin
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    private val detections = mutableListOf<DetectionResult>()
    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        typeface = Typeface.DEFAULT_BOLD
    }
    
    fun setDetections(newDetections: List<DetectionResult>) {
        detections.clear()
        detections.addAll(newDetections)
        invalidate()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        detections.forEach { detection ->
            val rect = detection.boundingBox
            
            // Draw bounding box
            canvas.drawRect(rect, paint)
            
            // Draw confidence score
            val text = String.format("Human: %.2f", detection.confidence * 100)
            val textBounds = Rect()
            textPaint.getTextBounds(text, 0, text.length, textBounds)
            
            canvas.drawRect(
                rect.left,
                rect.top - textBounds.height() - 8f,
                rect.left + textBounds.width() + 16f,
                rect.top,
                Paint().apply { color = Color.BLACK; style = Paint.Style.FILL }
            )
            
            canvas.drawText(text, rect.left + 8f, rect.top - 8f, textPaint)
        }
    }
}
```

## Performance Optimization

### Delegate Selection

Test different delegates on your target devices:

```kotlin
// CPU (always available)
val options = Interpreter.Options().apply {
    setNumThreads(Runtime.getRuntime().availableProcessors())
}

// GPU (optional)
if (GpuDelegateHelper.isGpuDelegateSupported()) {
    options.addDelegate(GpuDelegate())
}

// NNAPI (Android 8.1+)
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
    options.addDelegate(NnApiDelegate())
}
```

### Threading and Async Processing

```kotlin
class HumanDetectionThreadManager {
    
    private val detectionScope = CoroutineScope(Dispatchers.Default)
    private var isProcessing = false
    
    fun detectInBackground(image: Image, rotation: Int, onComplete: (List<DetectionResult>) -> Unit) {
        if (isProcessing) return // Skip frame if still processing
        
        detectionScope.launch {
            isProcessing = true
            try {
                val detections = yoloDetector.detectHumans(image, rotation)
                withContext(Dispatchers.Main) {
                    onComplete(detections)
                }
            } finally {
                isProcessing = false
            }
        }
    }
}
```

### Memory Management

```kotlin
class EfficientBitmapProcessor {
    private val bitmapPool = mutableMapOf<Pair<Int, Int>, Bitmap>()
    
    fun getBitmap(width: Int, height: Int): Bitmap {
        val key = width to height
        return bitmapPool[key] ?: Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
            bitmapPool[key] = it
        }
    }
    
    fun recycle() {
        bitmapPool.values.forEach { it.recycle() }
        bitmapPool.clear()
    }
}
```

## Model Variants and Performance Targets

| Model Variant | Input Size | Parameters | Expected mAP | Recommended Use |
|--------------|------------|------------|--------------|-----------------|
| YOLOv11n     | 320×320    | 2.6M       | ~35-39       | High-speed, low-resource |
| YOLOv11n     | 416×416    | 2.6M       | ~39-42       | Balanced accuracy/speed |
| YOLOv11n     | 640×640    | 2.6M       | ~42-45       | High accuracy, powerful devices |

## Common Integration Patterns

### 1. Real-time Human Detection (Camera Stream)
```kotlin
// Use when processing every frame
val detector = YOLOv11nHumanDetector(assets, "yolov11n_human.tflite", numThreads = 2)
```

### 2. Static Image Analysis
```kotlin
// Use for single image processing
val detector = YOLOv11nHumanDetector(assets, "yolov11n_human.tflite", numThreads = 4)
val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test_image)
val results = detector.detectInStaticImage(bitmap)
```

### 3. Background Processing
```kotlin
// Use for non-UI thread processing
val detector = YOLOv11nHumanDetector(assets, "yolov11n_human.tflite", numThreads = 1)
detector.use { // Auto-close after use
    it.detectHumans(image, rotation)
}
```

## Validation and Testing

### 1. Unit Testing
```kotlin
@Test
fun testYolov11nHumanDetection() {
    val detector = YOLOv11nHumanDetector(assets, "test_model.tflite")
    
    // Load test image with known human
    val testBitmap = BitmapFactory.decodeStream(
        assets.open("test_images/person_test.jpg")
    )
    
    val results = detector.detectInStaticImage(testBitmap)
    
    assert(results.size > 0)
    assert(results.first().confidence > 0.5f)
}
```

### 2. Performance Testing
```kotlin
@Test
fun testInferenceSpeed() {
    val detector = YOLOv11nHumanDetector(assets)
    val testBitmap = createTestBitmap(640, 640)
    
    val startTime = System.nanoTime()
    repeat(100) {
        detector.detectInStaticImage(testBitmap)
    }
    val endTime = System.nanoTime()
    
    val averageMs = (endTime - startTime) / 1_000_000.0 / 100
    assert(averageMs < 50) // Should be under 50ms per frame
}
```

## Best Practices

1. **Model Loading**: Load models asynchronously to avoid blocking the UI thread
2. **Error Handling**: Always handle model loading and inference errors gracefully
3. **Resource Management**: Use try-with-resources or `use` blocks for proper cleanup
4. **Memory Optimization**: Reuse buffers and bitmaps to reduce garbage collection
5. **Thread Safety**: Ensure thread-safe access to shared detector instances
6. **Configuration**: Make model parameters configurable (confidence thresholds, etc.)

## Next Steps

- See [Model Optimization Guide](model_optimization_guide.md) for advanced optimization techniques
- See [Performance Benchmarking Guide](performance_benchmarking.md) for testing instructions
- See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues and solutions

---

## References

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [TensorFlow Lite Android Guide](https://ai.google.dev/edge/litert/android)
- [Android Camera2 API](https://developer.android.com/reference/android/hardware/camera2/package-summary)
- [Kotlin Coroutines](https://kotlinlang.org/docs/coroutines-overview.html)