# YOLOv11n Android Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide covers common issues, diagnostic techniques, and solutions for YOLOv11n TFLite models on Android. It provides step-by-step debugging procedures, performance optimization tips, and solutions to integration challenges.

## Table of Contents

1. [Common Issues Overview](#common-issues-overview)
2. [Model Loading and Conversion Issues](#model-loading-and-conversion-issues)
3. [Runtime and Performance Issues](#runtime-and-performance-issues)
4. [Memory and Resource Issues](#memory-and-resource-issues)
5. [Accuracy and Detection Issues](#accuracy-and-detection-issues)
6. [Hardware Acceleration Issues](#hardware-acceleration-issues)
7. [Camera and Integration Issues](#camera-and-integration-issues)
8. [Build and Deployment Issues](#build-and-deployment-issues)
9. [Diagnostic Tools and Techniques](#diagnostic-tools-and-techniques)
10. [Advanced Debugging](#advanced-debugging)

## Common Issues Overview

### Most Frequent Issues

| Issue | Frequency | Impact | Difficulty |
|-------|-----------|---------|------------|
| Model loading failures | High | Critical | Easy |
| Poor accuracy | High | High | Medium |
| Slow inference | High | High | Medium |
| Memory errors (OOM) | Medium | High | Hard |
| GPU delegate issues | Medium | Medium | Hard |
| Camera integration problems | Medium | Medium | Medium |

### Issue Classification

- **Critical**: Prevents app from starting or core functionality
- **High**: Significant impact on user experience
- **Medium**: Noticeable issues but app remains functional
- **Low**: Minor cosmetic or performance issues

## Model Loading and Conversion Issues

### Issue 1: Model File Not Found or Corrupted

**Symptoms:**
```
java.io.FileNotFoundException: yolov11n_human.tflite
java.io.IOException: Error reading model data
```

**Diagnostic Steps:**
1. Verify file exists in assets directory
2. Check file size and integrity
3. Validate file permissions
4. Test file on desktop

**Solutions:**

**A. File Location Check:**
```kotlin
// Verify file path and existence
val assetManager = assets
val fileList = assetManager.list("models")
if (fileList == null || !fileList.contains("yolov11n_human.tflite")) {
    throw IllegalStateException("Model file not found in assets/models/")
}

// Alternative: Check specific path
try {
    val inputStream = assetManager.open("models/yolov11n_human.tflite")
    val available = inputStream.available()
    Log.d(TAG, "Model file size: $available bytes")
} catch (e: IOException) {
    Log.e(TAG, "Model file not accessible", e)
}
```

**B. File Integrity Check:**
```bash
# Check file size
adb shell ls -la /data/app/com.yourapp-*/cache/models/

# Verify file MD5
adb shell md5sum /data/app/com.yourapp-*/cache/models/yolov11n_human.tflite
```

**C. Alternative Loading Method:**
```kotlin
fun loadModelFromFile(context: Context, modelPath: String): Interpreter {
    try {
        // Method 1: Direct file
        val modelFile = File(modelPath)
        if (modelFile.exists()) {
            return Interpreter(modelFile)
        }
    } catch (e: Exception) {
        Log.w(TAG, "Direct file loading failed", e)
    }
    
    try {
        // Method 2: Assets
        val fileDescriptor = context.assets.openFd("models/$modelPath")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val mappedByteBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )
        return Interpreter(mappedByteBuffer)
    } catch (e: Exception) {
        Log.w(TAG, "Asset loading failed", e)
    }
    
    // Method 3: Memory-mapped file
    val file = File(context.cacheDir, modelPath)
    if (!file.exists()) {
        context.assets.open("models/$modelPath").use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
    }
    
    return Interpreter(file)
}
```

### Issue 2: TFLite Model Incompatible Operations

**Symptoms:**
```
java.lang.IllegalArgumentException: Internal error: Failed to apply graph to the lite runtime
 tensorflow/lite/kernels/conv.cc:262 Op is not supported on this platform
```

**Diagnostic Steps:**
1. Check TFLite operation compatibility
2. Verify delegate support
3. Test with different model variants

**Solutions:**

**A. Operation Compatibility Check:**
```python
import tensorflow as tf
import numpy as np

def check_model_operations(tflite_path):
    """Check which operations are supported by TFLite."""
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("=== Model Operation Analysis ===")
    print(f"Inputs: {len(input_details)}")
    for i, inp in enumerate(input_details):
        print(f"  Input {i}: {inp['name']} - {inp['shape']} - {inp['dtype']}")
    
    print(f"Outputs: {len(output_details)}")
    for i, out in enumerate(output_details):
        print(f"  Output {i}: {out['name']} - {out['shape']} - {out['dtype']}")
    
    # Test with dummy data
    dummy_input = np.random.random(input_details[0]['shape']).astype(input_details[0]['dtype'])
    try:
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        print("✓ Model can be executed successfully")
    except Exception as e:
        print(f"✗ Model execution failed: {e}")
    
    return True

# Usage
check_model_operations("yolov11n_human.tflite")
```

**B. Delegate Support Verification:**
```kotlin
class DelegateCompatibilityChecker {
    
    fun checkDelegateSupport(
        context: Context,
        modelPath: String
    ): Map<DelegateType, Boolean> {
        
        val results = mutableMapOf<DelegateType, Boolean>()
        
        // Test CPU (always supported)
        results[DelegateType.CPU] = testDelegate(DelegateType.CPU, modelPath)
        
        // Test GPU
        results[DelegateType.GPU] = testDelegate(DelegateType.GPU, modelPath)
        
        // Test NNAPI
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            results[DelegateType.NNAPI] = testDelegate(DelegateType.NNAPI, modelPath)
        }
        
        return results
    }
    
    private fun testDelegate(
        delegateType: DelegateType,
        modelPath: String
    ): Boolean {
        return try {
            val options = Interpreter.Options()
            
            when (delegateType) {
                DelegateType.GPU -> {
                    // Check GPU delegate availability
                    val gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                    gpuDelegate.close()
                }
                DelegateType.NNAPI -> {
                    val nnApiDelegate = NnApiDelegate()
                    options.addDelegate(nnApiDelegate)
                    nnApiDelegate.close()
                }
                else -> { /* CPU - no special setup */ }
            }
            
            val interpreter = Interpreter(loadModelBuffer(modelPath), options)
            interpreter.close()
            true
            
        } catch (e: Exception) {
            Log.w(TAG, "Delegate $delegateType not supported", e)
            false
        }
    }
}
```

**C. Fallback Strategy:**
```kotlin
class RobustModelLoader {
    
    fun loadModelWithFallback(
        context: Context,
        modelPath: String
    ): Interpreter {
        
        val candidates = listOf(
            ModelLoadingStrategy("GPU", this::tryLoadWithGPU),
            ModelLoadingStrategy("NNAPI", this::tryLoadWithNNAPI),
            ModelLoadingStrategy("CPU", this::tryLoadWithCPU)
        )
        
        for (strategy in candidates) {
            try {
                val interpreter = strategy.loadModel(context, modelPath)
                Log.i(TAG, "Model loaded successfully with ${strategy.name}")
                return interpreter
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load with ${strategy.name}", e)
            }
        }
        
        throw IllegalStateException("Failed to load model with any available strategy")
    }
    
    private fun tryLoadWithGPU(context: Context, modelPath: String): Interpreter {
        val options = Interpreter.Options().apply {
            addDelegate(GpuDelegate())
            setNumThreads(1) // GPU typically works best with 1 thread
        }
        return Interpreter(loadModelBuffer(context, modelPath), options)
    }
    
    private fun tryLoadWithNNAPI(context: Context, modelPath: String): Interpreter {
        val options = Interpreter.Options().apply {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                addDelegate(NnApiDelegate())
            }
        }
        return Interpreter(loadModelBuffer(context, modelPath), options)
    }
    
    private fun tryLoadWithCPU(context: Context, modelPath: String): Interpreter {
        val options = Interpreter.Options().apply {
            setNumThreads(Runtime.getRuntime().availableProcessors())
            setUseXNNPack(true)
        }
        return Interpreter(loadModelBuffer(context, modelPath), options)
    }
}
```

### Issue 3: Model Conversion Warnings and Errors

**Symptoms:**
```
WARNING: TensorFlow Lite: Operator 'Cast' is not supported on CPU
WARNING: TensorFlow Lite: Value for attr 'dtype' of RESHAPE is not present
ERROR: Failed to convert TensorFlow Lite model
```

**Solutions:**

**A. Fix Common Conversion Issues:**
```python
import tensorflow as tf
from ultralytics import YOLO

def convert_yolo_with_fixes(
    source_model_path: str,
    output_path: str,
    input_size: int = 640
) -> str:
    """Convert YOLO with common issue fixes."""
    
    # 1. First, try direct export
    try:
        model = YOLO(source_model_path)
        model.export(
            format='tflite',
            imgsz=input_size,
            half=True,  # FP16 for better compatibility
            nms=True,   # Bundle NMS to avoid post-processing issues
            batch=1,
            verbose=True
        )
        return output_path
    except Exception as e:
        print(f"Direct export failed: {e}")
    
    # 2. If direct export fails, try ONNX intermediate
    try:
        # Export to ONNX first
        onnx_path = source_model_path.replace('.pt', '.onnx')
        model.export(format='onnx', imgsz=input_size, simplify=True)
        
        # Convert ONNX to TFLite
        converter = tf.lite.TFLiteConverter.from_onnx(onnx_path)
        
        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Convert
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        return output_path
        
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
    
    # 3. Last resort: Manual conversion with additional fixes
    return manual_conversion_workaround(source_model_path, output_path)

def manual_conversion_workaround(source_path: str, output_path: str) -> str:
    """Manual conversion with extensive fixes for compatibility."""
    
    # Load model
    model = YOLO(source_path)
    
    # Get model's input and output specifications
    input_shape = (1, 3, 640, 640)  # NCHW format
    output_shapes = [(1, 3, 80, 80, 85), (1, 3, 40, 40, 85), (1, 3, 20, 20, 85)]
    
    # Create a simple representative model for conversion
    @tf.function
    def representative_function(image_input):
        # Simple preprocessing to ensure compatibility
        image_input = tf.cast(image_input, tf.float32)
        image_input = image_input / 255.0
        return image_input
    
    # Create converter with representative dataset
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = representative_dataset_generator(input_shape)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.convert()
    
    return output_path

def representative_dataset_generator(input_shape):
    """Generate representative dataset for conversion."""
    
    def representative_data():
        # Generate 100 random samples for calibration
        for _ in range(100):
            # Create random input
            yield [np.random.random(input_shape).astype(np.float32)]
    
    return representative_data
```

## Runtime and Performance Issues

### Issue 4: Slow Inference Performance

**Symptoms:**
- Low FPS (<10)
- High latency (>100ms)
- UI jank and stuttering

**Diagnostic Steps:**
1. Profile inference time
2. Check delegate performance
3. Analyze memory usage
4. Monitor thermal throttling

**Solutions:**

**A. Performance Profiling:**
```kotlin
class InferenceProfiler {
    
    data class ProfilingResult(
        val totalTime: Long,
        val preprocessTime: Long,
        val inferenceTime: Long,
        val postprocessTime: Long,
        val memoryUsage: Long,
        val threadCount: Int,
        val delegateType: String
    )
    
    fun profileInference(
        detector: YOLOv11nHumanDetector,
        testImage: Bitmap,
        iterations: Int = 100
    ): ProfilingResult {
        
        val results = mutableListOf<ProfilingResult>()
        
        repeat(iterations) { iteration ->
            val result = profileSingleInference(detector, testImage)
            results.add(result)
        }
        
        return analyzeResults(results)
    }
    
    private fun profileSingleInference(
        detector: YOLOv11nHumanDetector,
        testImage: Bitmap
    ): ProfilingResult {
        
        val startMemory = getMemoryUsage()
        
        // Preprocess
        val preprocessStart = System.nanoTime()
        val preprocessedImage = detector.preprocessImage(testImage)
        val preprocessEnd = System.nanoTime()
        
        // Inference
        val inferenceStart = System.nanoTime()
        val results = detector.detectHumans(preprocessedImage)
        val inferenceEnd = System.nanoTime()
        
        // Post-process
        val postprocessStart = System.nanoTime()
        val processedResults = detector.postprocessResults(results)
        val postprocessEnd = System.nanoTime()
        
        val endMemory = getMemoryUsage()
        
        return ProfilingResult(
            totalTime = postprocessEnd - preprocessStart,
            preprocessTime = preprocessEnd - preprocessStart,
            inferenceTime = inferenceEnd - inferenceStart,
            postprocessTime = postprocessEnd - postprocessStart,
            memoryUsage = endMemory - startMemory,
            threadCount = detector.threadCount,
            delegateType = detector.delegateType
        )
    }
    
    private fun getMemoryUsage(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }
    
    private fun analyzeResults(results: List<ProfilingResult>): ProfilingResult {
        val avg = ProfilingResult(
            totalTime = results.map { it.totalTime }.average().toLong(),
            preprocessTime = results.map { it.preprocessTime }.average().toLong(),
            inferenceTime = results.map { it.inferenceTime }.average().toLong(),
            postprocessTime = results.map { it.postprocessTime }.average().toLong(),
            memoryUsage = results.map { it.memoryUsage }.average().toLong(),
            threadCount = results.first().threadCount,
            delegateType = results.first().delegateType
        )
        
        Log.i(TAG, "Performance Profile:")
        Log.i(TAG, "  Total: ${avg.totalTime / 1_000_000}ms")
        Log.i(TAG, "  Preprocess: ${avg.preprocessTime / 1_000_000}ms")
        Log.i(TAG, "  Inference: ${avg.inferenceTime / 1_000_000}ms")
        Log.i(TAG, "  Postprocess: ${avg.postprocessTime / 1_000_000}ms")
        
        return avg
    }
}
```

**B. Performance Optimization:**
```kotlin
class PerformanceOptimizer {
    
    fun optimizeForPerformance(
        detector: YOLOv11nHumanDetector,
        deviceCapabilities: DeviceCapabilities
    ): YOLOv11nHumanDetector {
        
        // 1. Thread count optimization
        val optimalThreadCount = calculateOptimalThreadCount(deviceCapabilities)
        detector.setThreadCount(optimalThreadCount)
        
        // 2. Delegate optimization
        val optimalDelegate = selectOptimalDelegate(deviceCapabilities)
        detector.setDelegate(optimalDelegate)
        
        // 3. Memory optimization
        configureMemorySettings(detector, deviceCapabilities)
        
        // 4. Input size optimization
        val optimalInputSize = calculateOptimalInputSize(deviceCapabilities)
        detector.setInputSize(optimalInputSize)
        
        return detector
    }
    
    private fun calculateOptimalThreadCount(capabilities: DeviceCapabilities): Int {
        return when {
            capabilities.hasGpu -> 1 // GPU works best with single thread
            capabilities.cpuCores <= 4 -> capabilities.cpuCores / 2
            else -> 4 // General good default
        }
    }
    
    private fun selectOptimalDelegate(capabilities: DeviceCapabilities): DelegateType {
        return when {
            capabilities.hasGpu && capabilities.ramSize >= 4 * 1024 * 1024 * 1024L -> 
                DelegateType.GPU
            capabilities.hasNpu -> DelegateType.NNAPI
            else -> DelegateType.CPU
        }
    }
    
    private fun calculateOptimalInputSize(capabilities: DeviceCapabilities): Int {
        return when {
            capabilities.ramSize < 2 * 1024 * 1024 * 1024L -> 320  // < 2GB RAM
            capabilities.cpuCores <= 4 -> 416                      // <= 4 cores
            else -> 640                                          // >= 4 cores, >= 2GB RAM
        }
    }
}
```

**C. Real-time Performance Monitoring:**
```kotlin
class RealTimePerformanceMonitor {
    
    private val performanceMetrics = CircularBuffer<PerformanceMetric>(1000)
    
    data class PerformanceMetric(
        val timestamp: Long,
        val inferenceTime: Long,
        val fps: Float,
        val memoryUsage: Long,
        val cpuUsage: Float
    )
    
    fun startMonitoring() {
        CoroutineScope(Dispatchers.Default).launch {
            while (isMonitoring) {
                val metric = collectCurrentMetric()
                performanceMetrics.add(metric)
                
                // Check for performance degradation
                if (isPerformanceDegraded(metric)) {
                    handlePerformanceIssue(metric)
                }
                
                delay(100) // Sample every 100ms
            }
        }
    }
    
    private fun isPerformanceDegraded(metric: PerformanceMetric): Boolean {
        if (performanceMetrics.size < 10) return false
        
        val recentMetrics = performanceMetrics.toList().takeLast(10)
        val avgRecentFps = recentMetrics.map { it.fps }.average()
        val avgRecentInferenceTime = recentMetrics.map { it.inferenceTime }.average()
        
        // Consider degraded if FPS drops by 20% or inference time increases by 50%
        return avgRecentFps < metric.fps * 0.8 || 
               avgRecentInferenceTime > metric.inferenceTime * 1.5
    }
    
    private fun handlePerformanceIssue(metric: PerformanceMetric) {
        Log.w(TAG, "Performance degradation detected")
        
        // Automatically optimize settings
        optimizationManager?.onPerformanceIssue(metric)
    }
}

class OptimizationManager {
    private var currentConfig = PerformanceConfig()
    
    fun onPerformanceIssue(metric: RealTimePerformanceMonitor.PerformanceMetric) {
        when {
            metric.inferenceTime > 50000000L -> { // 50ms
                // Reduce input size
                currentConfig.inputSize = maxOf(320, currentConfig.inputSize - 32)
            }
            metric.cpuUsage > 80f -> {
                // Increase frame skipping
                currentConfig.frameSkipCount = currentConfig.frameSkipCount + 1
            }
            metric.memoryUsage > 100 * 1024 * 1024L -> { // 100MB
                // Clear caches and reduce batch size
                clearCaches()
            }
        }
    }
}
```

### Issue 5: High Memory Usage and OOM Errors

**Symptoms:**
```
OutOfMemoryError: Failed to allocate a X byte allocation with Y free bytes
GC activity causing frame drops
Application crashes with low memory warnings
```

**Solutions:**

**A. Memory Usage Monitoring:**
```kotlin
class MemoryMonitor {
    
    data class MemorySnapshot(
        val timestamp: Long,
        val heapUsed: Long,
        val heapTotal: Long,
        val nativeHeapUsed: Long,
        val gcCount: Int,
        val gcTime: Long
    )
    
    fun captureMemorySnapshot(): MemorySnapshot {
        val runtime = Runtime.getRuntime()
        val debug = Debug.getMemoryInfo(Debug.getGlobalMemoryInfo())
        
        return MemorySnapshot(
            timestamp = System.currentTimeMillis(),
            heapUsed = runtime.totalMemory() - runtime.freeMemory(),
            heapTotal = runtime.totalMemory(),
            nativeHeapUsed = debug.getTotalSwappablePss().toLong(),
            gcCount = Debug.getGlobalGcInvocations(),
            gcTime = 0 // Requires specific GC metrics
        )
    }
    
    fun analyzeMemoryLeaks(snapshots: List<MemorySnapshot>): MemoryLeakAnalysis {
        if (snapshots.size < 10) {
            return MemoryLeakAnalysis("Insufficient data for analysis")
        }
        
        val firstSnapshot = snapshots.first()
        val lastSnapshot = snapshots.last()
        
        val heapGrowth = lastSnapshot.heapUsed - firstSnapshot.heapUsed
        val timeSpan = lastSnapshot.timestamp - firstSnapshot.timestamp
        
        val growthRatePerHour = (heapGrowth.toFloat() / timeSpan) * 3600 * 1000
        
        return MemoryLeakAnalysis(
            hasLeak = growthRatePerHour > 10 * 1024 * 1024, // 10MB per hour
            growthRate = growthRatePerHour,
            suggestions = generateMemoryOptimizationSuggestions(snapshots)
        )
    }
    
    private fun generateMemoryOptimizationSuggestions(
        snapshots: List<MemorySnapshot>
    ): List<String> {
        val suggestions = mutableListOf<String>()
        
        val avgHeapUsage = snapshots.map { it.heapUsed }.average()
        val peakHeapUsage = snapshots.maxOf { it.heapUsed }
        
        if (peakHeapUsage > 100 * 1024 * 1024) { // 100MB
            suggestions.add("Consider reducing input image size")
            suggestions.add("Implement bitmap pooling")
        }
        
        val gcFrequency = calculateGCFrequency(snapshots)
        if (gcFrequency > 1) {
            suggestions.add("High GC frequency detected - reduce object allocations")
            suggestions.add("Use object pools for frequently created objects")
        }
        
        return suggestions
    }
}
```

**B. Memory Optimization Strategies:**
```kotlin
class MemoryOptimizedDetector {
    
    // Object pools to prevent allocations
    private val bitmapPool = BitmapPool(3) // Cache up to 3 bitmaps
    private val byteBufferPool = ByteBufferPool(5) // Cache 5 buffers
    private val resultListPool = ListPool<DetectionResult>(10)
    
    // Pre-allocated arrays
    private val preprocessedBuffer = ByteBuffer.allocateDirect(640 * 640 * 3 * 4)
    private val outputBuffer = Array(1) { Array(300) { FloatArray(6) } }
    
    fun detectWithOptimization(image: Image): List<DetectionResult> {
        // Reuse bitmap from pool
        val bitmap = bitmapPool.acquire()
        try {
            convertImageToBitmap(image, bitmap)
            
            // Reuse buffer
            val inputBuffer = byteBufferPool.acquire()
            try {
                preprocessedBuffer.clear()
                preprocessImageToBuffer(bitmap, preprocessedBuffer)
                
                // Use pre-allocated output
                model.run(preprocessedBuffer, outputBuffer)
                
                // Reuse result list
                val results = resultListPool.acquire()
                try {
                    processResults(outputBuffer[0], results)
                    return ArrayList(results) // Return copy, keep pool entry
                } finally {
                    results.clear()
                    resultListPool.release(results)
                }
            } finally {
                inputBuffer.clear()
                byteBufferPool.release(inputBuffer)
            }
        } finally {
            bitmap.eraseColor(Color.TRANSPARENT) // Clear for reuse
            bitmapPool.release(bitmap)
        }
    }
    
    // Implement pooling classes
    class BitmapPool(private val maxSize: Int) {
        private val available = ArrayDeque<Bitmap>()
        
        fun acquire(): Bitmap {
            return if (available.isNotEmpty()) {
                available.removeFirst()
            } else {
                Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
            }
        }
        
        fun release(bitmap: Bitmap) {
            if (available.size < maxSize) {
                available.addLast(bitmap)
            } else {
                bitmap.recycle()
            }
        }
    }
    
    class ByteBufferPool(private val maxSize: Int) {
        private val available = ArrayDeque<ByteBuffer>()
        
        fun acquire(): ByteBuffer {
            return if (available.isNotEmpty()) {
                available.removeFirst().apply { clear() }
            } else {
                ByteBuffer.allocateDirect(640 * 640 * 3 * 4).order(ByteOrder.nativeOrder())
            }
        }
        
        fun release(buffer: ByteBuffer) {
            if (available.size < maxSize) {
                available.addLast(buffer)
            }
        }
    }
}
```

## Memory and Resource Issues

### Issue 6: Garbage Collection and Frame Drops

**Symptoms:**
- Jank and frame stuttering
- Periodic slowdowns
- High GC activity in logcat

**Solutions:**

**A. GC Monitoring and Analysis:**
```kotlin
class GCAnalyzer {
    
    data class GCEvent(
        val timestamp: Long,
        val gcType: String,
        val duration: Long,
        val freedMemory: Long,
        val heapBefore: Long,
        val heapAfter: Long
    )
    
    private val gcEvents = mutableListOf<GCEvent>()
    private val gcListener = ComponentCallbacks2 { event ->
        when (event) {
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> {
                handleCriticalMemoryPressure()
            }
            ComponentCallbacks2.TRIM_MEMORY_MODERATE -> {
                handleModerateMemoryPressure()
            }
            ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> {
                handleCompleteMemoryPressure()
            }
        }
    }
    
    fun startGCAnalysis() {
        // Register component callbacks
        context.registerComponentCallbacks(gcListener)
        
        // Monitor GC through finalization
        startGCFinalizationMonitor()
    }
    
    private fun startGCFinalizationMonitor() {
        CoroutineScope(Dispatchers.Default).launch {
            val finalizationQueue = ReferenceQueue<Any>()
            val weakRefs = mutableListOf<WeakReference<Any>>()
            
            // Create weak references to monitor finalization
            repeat(100) { i ->
                val ref = WeakReference(Object(), finalizationQueue, i)
                weakRefs.add(ref)
            }
            
            while (true) {
                val ref = finalizationQueue.remove(1000) // Timeout after 1 second
                if (ref != null) {
                    handleObjectFinalization(ref)
                }
            }
        }
    }
    
    private fun handleObjectFinalization(ref: WeakReference<*>) {
        val timestamp = System.currentTimeMillis()
        val heapInfo = getCurrentHeapInfo()
        
        Log.d(TAG, "Object finalized at $timestamp, heap: ${heapInfo.usedHeap}MB")
        
        // Analyze for potential memory leaks
        if (isHighFinalizationRate()) {
            Log.w(TAG, "High finalization rate detected - check for memory leaks")
        }
    }
    
    fun analyzeGCPerformance(): String {
        if (gcEvents.isEmpty()) return "No GC events recorded"
        
        val totalGCs = gcEvents.size
        val totalGCDuration = gcEvents.sumOf { it.duration }
        val avgGCDuration = totalGCDuration / totalGCs
        
        val sb = StringBuilder()
        sb.append("GC Performance Analysis:\n")
        sb.append("Total GC events: $totalGCs\n")
        sb.append("Total GC time: ${totalGCDuration}ms\n")
        sb.append("Average GC time: ${avgGCDuration}ms\n")
        
        if (avgGCDuration > 10) {
            sb.append("WARNING: Average GC time too high (>10ms)\n")
        }
        
        return sb.toString()
    }
}
```

**B. Allocation Optimization:**
```kotlin
class LowAllocationProcessor {
    
    // Pre-allocated, reusable objects
    private val reusableRect = Rect()
    private val reusablePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val reusableTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        typeface = Typeface.DEFAULT
    }
    
    // Pre-allocated arrays to avoid object creation
    private val colorArray = IntArray(3)
    private val coordinateArray = FloatArray(4)
    
    fun processDetectionsWithMinimalAllocations(detections: List<DetectionResult>, canvas: Canvas) {
        detections.forEach { detection ->
            // Reuse pre-allocated objects
            reusableRect.set(
                detection.boundingBox.left.toInt(),
                detection.boundingBox.top.toInt(),
                detection.boundingBox.right.toInt(),
                detection.boundingBox.bottom.toInt()
            )
            
            // Draw without creating new objects
            canvas.drawRect(reusableRect, reusablePaint)
            
            // Draw text without string concatenation
            drawDetectionText(detection, canvas, reusableRect)
        }
    }
    
    private fun drawDetectionText(
        detection: DetectionResult,
        canvas: Canvas,
        rect: Rect
    ) {
        // Create text directly without temporary objects
        val textBuffer = StringBuilder(32)
        textBuffer.append("Human: ")
        textBuffer.append((detection.confidence * 100).toInt())
        textBuffer.append("%")
        
        val text = textBuffer.toString()
        canvas.drawText(text, rect.left + 8f, rect.top - 8f, reusableTextPaint)
    }
}
```

## Accuracy and Detection Issues

### Issue 7: Poor Detection Accuracy

**Symptoms:**
- No detections on clear human images
- Low confidence scores
- Incorrect bounding boxes
- High false positive/negative rates

**Diagnostic Steps:**

**A. Model Validation:**
```kotlin
class ModelAccuracyValidator {
    
    fun validateModelAccuracy(
        detector: YOLOv11nHumanDetector,
        testImages: List<TestImage>,
        groundTruth: List<GroundTruthBox>
    ): AccuracyReport {
        
        val results = mutableListOf<DetectionResult>()
        val metrics = mutableListOf<DetectionMetric>()
        
        for (testImage in testImages) {
            val detections = detector.detectHumans(testImage.bitmap, 0)
            val groundTruthBoxes = groundTruth.filter { 
                it.imageId == testImage.id 
            }
            
            val metric = calculateDetectionMetrics(detections, groundTruthBoxes)
            metrics.add(metric)
        }
        
        return AccuracyReport(
            averagePrecision = metrics.map { it.precision }.average(),
            averageRecall = metrics.map { it.recall }.average(),
            averageF1 = metrics.map { it.f1Score }.average(),
            totalImages = testImages.size,
            detailedMetrics = metrics
        )
    }
    
    private fun calculateDetectionMetrics(
        predictions: List<DetectionResult>,
        groundTruth: List<GroundTruthBox>,
        iouThreshold: Float = 0.5f
    ): DetectionMetric {
        
        val truePositives = countTruePositives(predictions, groundTruth, iouThreshold)
        val falsePositives = predictions.size - truePositives
        val falseNegatives = groundTruth.size - truePositives
        
        val precision = if (predictions.isNotEmpty()) {
            truePositives.toFloat() / (truePositives + falsePositives)
        } else 0f
        
        val recall = if (groundTruth.isNotEmpty()) {
            truePositives.toFloat() / (truePositives + falseNegatives)
        } else 0f
        
        val f1Score = if (precision + recall > 0) {
            2 * (precision * recall) / (precision + recall)
        } else 0f
        
        return DetectionMetric(
            precision = precision,
            recall = recall,
            f1Score = f1Score,
            truePositives = truePositives,
            falsePositives = falsePositives,
            falseNegatives = falseNegatives
        )
    }
}
```

**B. Preprocessing Validation:**
```kotlin
class PreprocessingValidator {
    
    fun validatePreprocessing(
        detector: YOLOv11nHumanDetector,
        originalImage: Bitmap
    ): PreprocessingReport {
        
        val report = PreprocessingReport()
        
        // Test different preprocessing approaches
        val method1Result = testPreprocessingMethod1(detector, originalImage)
        val method2Result = testPreprocessingMethod2(detector, originalImage)
        val method3Result = testPreprocessingMethod3(detector, originalImage)
        
        report.addMethod("Original", method1Result)
        report.addMethod("Normalized", method2Result)
        report.addMethod("Enhanced", method3Result)
        
        return report
    }
    
    private fun testPreprocessingMethod1(
        detector: YOLOv11nHumanDetector,
        image: Bitmap
    ): MethodResult {
        
        val startTime = System.nanoTime()
        
        // Method 1: Direct preprocessing
        val buffer = ByteBuffer.allocateDirect(640 * 640 * 3 * 4)
        for (y in 0 until 640) {
            for (x in 0 until 640) {
                val pixel = image.getPixel(x, y)
                buffer.putFloat(((pixel shr 16 and 0xFF) / 255.0f))
                buffer.putFloat(((pixel shr 8 and 0xFF) / 255.0f))
                buffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }
        buffer.rewind()
        
        val endTime = System.nanoTime()
        
        return MethodResult(
            methodName = "Direct",
            processingTime = endTime - startTime,
            result = "Success"
        )
    }
    
    // Implement other preprocessing methods...
}
```

**C. Model-Specific Debugging:**
```kotlin
class ModelDebuggingTools {
    
    fun debugModelPredictions(
        detector: YOLOv11nHumanDetector,
        testImage: Bitmap
    ): ModelDebugInfo {
        
        val debugInfo = ModelDebugInfo()
        
        // 1. Check input preprocessing
        val preprocessedInput = detector.preprocessImage(testImage)
        debugInfo.inputStats = analyzeInputStats(preprocessedInput)
        
        // 2. Get raw model outputs
        val rawOutputs = getRawModelOutputs(detector, preprocessedInput)
        debugInfo.rawOutputs = analyzeRawOutputs(rawOutputs)
        
        // 3. Step through post-processing
        val postProcessedResults = detector.postprocessResults(rawOutputs)
        debugInfo.postProcessedResults = analyzePostProcessedResults(postProcessedResults)
        
        // 4. Check thresholds
        debugInfo.thresholdAnalysis = analyzeThresholds(postProcessedResults)
        
        return debugInfo
    }
    
    private fun analyzeInputStats(inputBuffer: ByteBuffer): InputStats {
        val originalPosition = inputBuffer.position()
        inputBuffer.rewind()
        
        var minValue = Float.MAX_VALUE
        var maxValue = Float.MIN_VALUE
        var sum = 0.0
        var count = 0
        
        while (inputBuffer.hasRemaining()) {
            val value = inputBuffer.float
            minValue = minOf(minValue, value)
            maxValue = maxOf(maxValue, value)
            sum += value
            count++
        }
        
        inputBuffer.position(originalPosition) // Restore position
        
        return InputStats(
            count = count,
            minValue = minValue,
            maxValue = maxValue,
            meanValue = (sum / count).toFloat(),
            range = maxValue - minValue
        )
    }
    
    private fun analyzeRawOutputs(rawOutputs: Array<FloatArray>): RawOutputAnalysis {
        val analysis = RawOutputAnalysis()
        
        for (output in rawOutputs) {
            for (value in output) {
                if (value.isFinite()) {
                    analysis.totalValues++
                    if (value > 0.5f) analysis.highConfidenceCount++
                    if (value > analysis.maxValue) analysis.maxValue = value
                    if (value < analysis.minValue) analysis.minValue = value
                }
            }
        }
        
        return analysis
    }
}
```

### Issue 8: Confidence and Threshold Issues

**Solutions:**

**A. Dynamic Threshold Adjustment:**
```kotlin
class AdaptiveThresholdManager {
    
    private val thresholdHistory = mutableListOf<ThresholdStat>()
    private val optimalThreshold = MutableLiveData<Float>()
    
    data class ThresholdStat(
        val timestamp: Long,
        val confidenceThreshold: Float,
        val precision: Float,
        val recall: Float,
        val f1Score: Float
    )
    
    fun adjustThresholdsBasedOnPerformance(
        currentResults: List<DetectionResult>,
        groundTruth: List<GroundTruthBox>
    ) {
        val currentStats = calculateCurrentStats(currentResults, groundTruth)
        thresholdHistory.add(currentStats)
        
        // Keep only recent history
        if (thresholdHistory.size > 100) {
            thresholdHistory.removeFirst()
        }
        
        // Analyze threshold performance
        val optimalConfidence = findOptimalConfidenceThreshold()
        optimalThreshold.value = optimalConfidence
        
        Log.i(TAG, "Optimal confidence threshold: $optimalConfidence")
    }
    
    private fun findOptimalConfidenceThreshold(): Float {
        if (thresholdHistory.size < 10) return 0.5f // Default
        
        // Find threshold with best F1 score
        val bestThreshold = thresholdHistory.maxByOrNull { it.f1Score }
        return bestThreshold?.confidenceThreshold ?: 0.5f
    }
    
    fun getOptimalThresholds(): Pair<Float, Float> {
        val confidenceThreshold = optimalThreshold.value ?: 0.5f
        val iouThreshold = calculateOptimalIoUThreshold()
        return Pair(confidenceThreshold, iouThreshold)
    }
    
    private fun calculateOptimalIoUThreshold(): Float {
        // Analyze IoU distribution to find optimal threshold
        val recentResults = thresholdHistory.takeLast(20)
        if (recentResults.isEmpty()) return 0.5f
        
        // Return average threshold from recent high-performance results
        val highPerformanceResults = recentResults.filter { it.f1Score > 0.7f }
        return if (highPerformanceResults.isNotEmpty()) {
            0.45f // Conservative default
        } else {
            0.5f
        }
    }
}
```

## Hardware Acceleration Issues

### Issue 9: GPU Delegate Issues

**Symptoms:**
```
Failed to initialize GPU delegate: Failed to create shader
GPU delegate is not supported on this device
W/GPU: Op not supported, falling back to CPU
```

**Solutions:**

**A. GPU Delegate Compatibility Check:**
```kotlin
class GpuCompatibilityChecker {
    
    fun checkGpuCompatibility(context: Context): GpuCompatibilityReport {
        val report = GpuCompatibilityReport()
        
        // 1. Check basic GPU support
        report.hasGpuSupport = checkBasicGpuSupport()
        if (!report.hasGpuSupport) {
            report.reason = "GPU not supported"
            return report
        }
        
        // 2. Check OpenGL support
        report.openGlVersion = getOpenGlVersion()
        if (report.openGlVersion < 3.0f) {
            report.reason = "OpenGL version too old: ${report.openGlVersion}"
            return report
        }
        
        // 3. Check available GPU memory
        report.gpuMemorySize = getGpuMemorySize()
        if (report.gpuMemorySize < 512 * 1024 * 1024) { // 512MB
            report.reason = "Insufficient GPU memory: ${report.gpuMemorySize} bytes"
            return report
        }
        
        // 4. Test TFLite GPU delegate
        report.tfliteGpuSupported = testTfliteGpuSupport()
        if (!report.tfliteGpuSupported) {
            report.reason = "TFLite GPU delegate not supported"
            return report
        }
        
        // 5. Check shader compilation
        report.shadersSupported = testShaderCompilation()
        if (!report.shadersSupported) {
            report.reason = "Shader compilation failed"
            return report
        }
        
        report.isFullyCompatible = true
        return report
    }
    
    private fun checkBasicGpuSupport(): Boolean {
        return try {
            val glesVersion = activityInfo.getGlEsVersion()
            val versionCode = glesVersion.toInt()
            versionCode >= 0x20000 // OpenGL ES 2.0 minimum
        } catch (e: Exception) {
            false
        }
    }
    
    private fun testTfliteGpuSupport(): Boolean {
        return try {
            val gpuDelegateHelper = Class.forName("org.tensorflow.lite.gpu.GpuDelegateHelper")
            val method = gpuDelegateHelper.getMethod("isGpuDelegateSupported")
            method.invoke(null) as Boolean
        } catch (e: Exception) {
            Log.w(TAG, "GPU delegate availability check failed", e)
            false
        }
    }
}
```

**B. GPU Delegate Fallback Strategy:**
```kotlin
class GpuDelegateManager {
    
    enum class GpuMode {
        GPU_PREFERRED,      // Try GPU first
        CPU_PREFERRED,      // Try CPU first
        GPU_ONLY,          // GPU only
        CPU_ONLY           // CPU only
    }
    
    fun createOptimalInterpreter(
        context: Context,
        modelPath: String,
        preferredMode: GpuMode = GpuMode.GPU_PREFERRED
    ): Interpreter {
        
        val candidates = when (preferredMode) {
            GpuMode.GPU_PREFERRED -> listOf(
                DelegateStrategy("GPU", this::createGpuInterpreter),
                DelegateStrategy("CPU", this::createCpuInterpreter)
            )
            GpuMode.CPU_PREFERRED -> listOf(
                DelegateStrategy("CPU", this::createCpuInterpreter),
                DelegateStrategy("GPU", this::createGpuInterpreter)
            )
            GpuMode.GPU_ONLY -> listOf(
                DelegateStrategy("GPU", this::createGpuInterpreter)
            )
            GpuMode.CPU_ONLY -> listOf(
                DelegateStrategy("CPU", this::createCpuInterpreter)
            )
        }
        
        for (strategy in candidates) {
            try {
                val interpreter = strategy.create(context, modelPath)
                Log.i(TAG, "Successfully created ${strategy.name} interpreter")
                return interpreter
            } catch (e: Exception) {
                Log.w(TAG, "Failed to create ${strategy.name} interpreter", e)
            }
        }
        
        throw IllegalStateException("Failed to create interpreter with any available strategy")
    }
    
    private fun createGpuInterpreter(context: Context, modelPath: String): Interpreter {
        val options = Interpreter.Options().apply {
            // GPU-specific settings
            val gpuOptions = GpuDelegateOptions.Builder()
                .setPrecisionLossAllowed(true) // Allow FP16/FP32 mixing
                .setInferencePreference(GpuInferencePreference.FAST_SINGLE_ANSWER)
                .build()
            
            val gpuDelegate = GpuDelegate(gpuOptions)
            addDelegate(gpuDelegate)
            
            // CPU fallback settings
            setUseXNNPack(true)
            setNumThreads(1) // GPU typically works best with single thread
        }
        
        return Interpreter(loadModelBuffer(context, modelPath), options)
    }
    
    private fun createCpuInterpreter(context: Context, modelPath: String): Interpreter {
        val options = Interpreter.Options().apply {
            setNumThreads(Runtime.getRuntime().availableProcessors())
            setUseXNNPack(true)
        }
        
        return Interpreter(loadModelBuffer(context, modelPath), options)
    }
}
```

## Camera and Integration Issues

### Issue 10: Camera2 Integration Problems

**Symptoms:**
- Black preview screen
- High camera latency
- Corrupted image formats
- Camera permission issues

**Solutions:**

**A. Camera2 Debugging Tools:**
```kotlin
class CameraDebugger {
    
    fun debugCameraSession(cameraDevice: CameraDevice): CameraDebugInfo {
        val debugInfo = CameraDebugInfo()
        
        try {
            // 1. Check camera characteristics
            val characteristics = cameraManager.getCameraCharacteristics(cameraDevice.id)
            debugInfo.cameraCharacteristics = analyzeCameraCharacteristics(characteristics)
            
            // 2. Check supported formats
            val supportedFormats = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)?.outputFormats
            debugInfo.supportedFormats = supportedFormats?.toList() ?: emptyList()
            
            // 3. Check max resolutions
            val streamConfigMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val maxResolutions = getMaxResolutions(streamConfigMap)
            debugInfo.maxResolutions = maxResolutions
            
            // 4. Test image formats
            val formatTestResults = testSupportedFormats(cameraDevice, supportedFormats)
            debugInfo.formatTestResults = formatTestResults
            
        } catch (e: CameraAccessException) {
            debugInfo.error = e.message ?: "Camera access exception"
        }
        
        return debugInfo
    }
    
    private fun testSupportedFormats(
        cameraDevice: CameraDevice,
        supportedFormats: IntArray?
    ): Map<Int, String> {
        val results = mutableMapOf<Int, String>()
        
        if (supportedFormats == null) return results
        
        val formats = listOf(
            ImageFormat.YUV_420_888,
            ImageFormat.JPEG,
            ImageFormat.RGBA_8888,
            ImageFormat.RGB_565
        )
        
        for (format in formats) {
            if (supportedFormats.contains(format)) {
                val testResult = testImageFormat(cameraDevice, format)
                results[format] = testResult
            }
        }
        
        return results
    }
    
    private fun testImageFormat(cameraDevice: CameraDevice, format: Int): String {
        return try {
            val imageReader = ImageReader.newInstance(1920, 1080, format, 2)
            
            val captureSession = cameraDevice.createCaptureSession(
                listOf(imageReader.surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        // Session configured successfully
                    }
                    
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        // Configuration failed
                    }
                },
                null
            )
            
            // Test completed
            imageReader.close()
            "Supported"
            
        } catch (e: Exception) {
            "Failed: ${e.message}"
        }
    }
}
```

**B. Robust Camera Initialization:**
```kotlin
class RobustCameraInitializer {
    
    enum class CameraResolution(val width: Int, val height: Int) {
        LOW(640, 480),
        MEDIUM(1280, 720),
        HIGH(1920, 1080)
    }
    
    fun initializeCameraWithFallback(
        activity: FragmentActivity,
        callback: (CameraDevice, CaptureSession) -> Unit
    ) {
        val cameraManager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        
        // Try different camera IDs
        val cameraIds = try {
            cameraManager.cameraIdList.toList()
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to get camera list", e)
            return
        }
        
        for (cameraId in cameraIds) {
            try {
                val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                
                if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                    initializeBackCamera(activity, cameraManager, cameraId, callback)
                    return
                }
            } catch (e: CameraAccessException) {
                Log.w(TAG, "Failed to access camera $cameraId", e)
            }
        }
        
        Log.e(TAG, "No suitable camera found")
    }
    
    private fun initializeBackCamera(
        activity: FragmentActivity,
        cameraManager: CameraManager,
        cameraId: String,
        callback: (CameraDevice, CaptureSession) -> Unit
    ) {
        try {
            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    // Choose resolution with fallback
                    val resolution = chooseOptimalResolution(cameraManager, cameraId)
                    createCaptureSession(activity, camera, resolution, callback)
                }
                
                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                }
                
                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e(TAG, "Camera error: $error")
                    camera.close()
                }
            }, null)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to open camera", e)
        }
    }
    
    private fun chooseOptimalResolution(
        cameraManager: CameraManager,
        cameraId: String
    ): CameraResolution {
        return try {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val streamConfigMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            
            // Check if high resolution is supported
            val highResSizes = streamConfigMap?.getOutputSizes(ImageFormat.YUV_420_888)
                ?.filter { it.width >= 1920 && it.height >= 1080 }
            
            if (!highResSizes.isNullOrEmpty()) {
                CameraResolution.HIGH
            } else {
                CameraResolution.MEDIUM
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to determine optimal resolution, using LOW", e)
            CameraResolution.LOW
        }
    }
}
```

## Diagnostic Tools and Techniques

### Issue 11: Comprehensive System Diagnostics

**Solutions:**

**A. System Diagnostics Collector:**
```kotlin
class SystemDiagnosticsCollector {
    
    data class SystemInfo(
        val deviceModel: String,
        val androidVersion: String,
        val apiLevel: Int,
        val cpuArchitecture: String,
        val totalMemory: Long,
        val availableProcessors: Int,
        val openglVersion: String,
        val hasGpu: Boolean,
        val gpuVendor: String,
        val supportedFeatures: Set<String>
    )
    
    fun collectSystemInfo(context: Context): SystemInfo {
        val packageManager = context.packageManager
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        val supportedFeatures = mutableSetOf<String>()
        
        // Check for various features
        supportedFeatures.addAll(checkSupportedFeatures(context))
        
        return SystemInfo(
            deviceModel = Build.MODEL,
            androidVersion = Build.VERSION.RELEASE,
            apiLevel = Build.VERSION.SDK_INT,
            cpuArchitecture = Build.SUPPORTED_ABIS.joinToString(","),
            totalMemory = memoryInfo.totalMem,
            availableProcessors = Runtime.getRuntime().availableProcessors(),
            openglVersion = getOpenGlVersion(),
            hasGpu = checkGpuSupport(context),
            gpuVendor = getGpuVendor(),
            supportedFeatures = supportedFeatures
        )
    }
    
    private fun checkSupportedFeatures(context: Context): List<String> {
        val packageManager = context.packageManager
        val features = mutableListOf<String>()
        
        // Camera features
        if (packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA)) {
            features.add("Camera")
        }
        if (packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_AUTOFOCUS)) {
            features.add("Camera-AutoFocus")
        }
        
        // OpenGL features
        if (packageManager.hasSystemFeature(PackageManager.FEATURE_OPENGLES_VERSION_2_0)) {
            features.add("OpenGL-ES-2.0")
        }
        if (packageManager.hasSystemFeature(PackageManager.FEATURE_OPENGLES_VERSION_3_0)) {
            features.add("OpenGL-ES-3.0")
        }
        
        // Hardware features
        if (packageManager.hasSystemFeature("android.hardware.vulkan.version")) {
            features.add("Vulkan")
        }
        
        return features
    }
    
    private fun getGpuVendor(): String {
        return try {
            val renderer = GLES30.glGetString(GLES30.GL_RENDERER)
            val vendor = GLES30.glGetString(GLES30.GL_VENDOR)
            "$vendor $renderer"
        } catch (e: Exception) {
            "Unknown"
        }
    }
}
```

**B. Performance Regression Detector:**
```kotlin
class PerformanceRegressionDetector {
    
    private val performanceHistory = mutableListOf<PerformanceRecord>()
    private val regressionThresholds = RegressionThresholds()
    
    data class PerformanceRecord(
        val timestamp: Long,
        val modelVersion: String,
        val deviceId: String,
        val avgLatency: Double,
        val avgFps: Double,
        val memoryUsage: Long
    )
    
    data class RegressionThresholds(
        val latencyIncrease: Double = 0.2,    // 20% increase
        val fpsDecrease: Double = 0.15,       // 15% decrease
        val memoryIncrease: Double = 0.25     // 25% increase
    )
    
    fun checkForRegressions(
        currentRecord: PerformanceRecord
    ): List<PerformanceRegression> {
        val regressions = mutableListOf<PerformanceRegression>()
        
        // Get recent baseline (last 10 records for same device/model)
        val baseline = performanceHistory
            .filter { it.deviceId == currentRecord.deviceId && it.modelVersion == currentRecord.modelVersion }
            .takeLast(10)
            .ifEmpty { return regressions }
        
        val baselineLatency = baseline.map { it.avgLatency }.average()
        val baselineFps = baseline.map { it.avgFps }.average()
        val baselineMemory = baseline.map { it.memoryUsage }.average()
        
        // Check for latency regression
        if (currentRecord.avgLatency > baselineLatency * (1 + regressionThresholds.latencyIncrease)) {
            regressions.add(
                PerformanceRegression(
                    type = RegressionType.LATENCY,
                    severity = calculateSeverity(
                        currentRecord.avgLatency,
                        baselineLatency,
                        regressionThresholds.latencyIncrease
                    ),
                    currentValue = currentRecord.avgLatency,
                    baselineValue = baselineLatency,
                    changePercent = ((currentRecord.avgLatency - baselineLatency) / baselineLatency * 100)
                )
            )
        }
        
        // Check for FPS regression
        if (currentRecord.avgFps < baselineFps * (1 - regressionThresholds.fpsDecrease)) {
            regressions.add(
                PerformanceRegression(
                    type = RegressionType.FPS,
                    severity = calculateSeverity(
                        baselineFps,
                        currentRecord.avgFps,
                        regressionThresholds.fpsDecrease
                    ),
                    currentValue = currentRecord.avgFps,
                    baselineValue = baselineFps,
                    changePercent = ((baselineFps - currentRecord.avgFps) / baselineFps * 100)
                )
            )
        }
        
        return regressions
    }
    
    private fun calculateSeverity(
        current: Double,
        baseline: Double,
        threshold: Double
    ): RegressionSeverity {
        val changePercent = Math.abs((current - baseline) / baseline)
        
        return when {
            changePercent > threshold * 2 -> RegressionSeverity.HIGH
            changePercent > threshold * 1.5 -> RegressionSeverity.MEDIUM
            else -> RegressionSeverity.LOW
        }
    }
}
```

## Advanced Debugging

### Issue 12: Model Execution Tracing

**Solutions:**

**A. Step-by-step Execution Tracer:**
```kotlin
class ModelExecutionTracer {
    
    data class ExecutionTrace(
        val stepName: String,
        val startTime: Long,
        val endTime: Long,
        val memoryBefore: Long,
        val memoryAfter: Long,
        val parameters: Map<String, Any>
    )
    
    private val traceEvents = mutableListOf<ExecutionTrace>()
    
    fun traceModelExecution(
        detector: YOLOv11nHumanDetector,
        testImage: Bitmap
    ): List<ExecutionTrace> {
        traceEvents.clear()
        
        // Preprocessing
        val preprocessStart = getMemoryUsage()
        traceEvent("preprocess_start", preprocessStart, emptyMap())
        
        val preprocessedImage = detector.preprocessImage(testImage)
        val preprocessEnd = getMemoryUsage()
        traceEvent("preprocess_end", preprocessEnd, mapOf(
            "image_size" to "${testImage.width}x${testImage.height}",
            "preprocessed_size" to "${preprocessedImage.remaining()}"
        ))
        
        // Inference
        val inferenceStart = getMemoryUsage()
        traceEvent("inference_start", inferenceStart, emptyMap())
        
        val results = detector.detectHumans(preprocessedImage)
        val inferenceEnd = getMemoryUsage()
        traceEvent("inference_end", inferenceEnd, mapOf(
            "results_count" to results.size
        ))
        
        // Post-processing
        val postprocessStart = getMemoryUsage()
        traceEvent("postprocess_start", postprocessStart, emptyMap())
        
        val processedResults = detector.postprocessResults(results)
        val postprocessEnd = getMemoryUsage()
        traceEvent("postprocess_end", postprocessEnd, mapOf(
            "processed_count" to processedResults.size
        ))
        
        return traceEvents.toList()
    }
    
    private fun traceEvent(
        stepName: String,
        memoryUsage: Long,
        parameters: Map<String, Any>
    ) {
        val timestamp = System.nanoTime()
        val lastMemory = traceEvents.lastOrNull()?.memoryAfter ?: 0L
        
        val event = ExecutionTrace(
            stepName = stepName,
            startTime = timestamp,
            endTime = timestamp, // Will be updated
            memoryBefore = lastMemory,
            memoryAfter = memoryUsage,
            parameters = parameters
        )
        
        traceEvents.add(event)
    }
    
    fun generateTraceReport(): String {
        if (traceEvents.isEmpty()) return "No trace data available"
        
        val sb = StringBuilder()
        sb.append("Model Execution Trace Report\n")
        sb.append("============================\n\n")
        
        for (i in traceEvents.indices) {
            val event = traceEvents[i]
            val duration = if (i < traceEvents.size - 1) {
                traceEvents[i + 1].startTime - event.startTime
            } else 0
            
            sb.append("${i + 1}. ${event.stepName}\n")
            sb.append("   Duration: ${duration / 1_000}μs\n")
            sb.append("   Memory before: ${event.memoryBefore / 1024}KB\n")
            sb.append("   Memory after: ${event.memoryAfter / 1024}KB\n")
            sb.append("   Memory delta: ${(event.memoryAfter - event.memoryBefore) / 1024}KB\n")
            
            if (event.parameters.isNotEmpty()) {
                sb.append("   Parameters:\n")
                for ((key, value) in event.parameters) {
                    sb.append("     $key: $value\n")
                }
            }
            sb.append("\n")
        }
        
        return sb.toString()
    }
}
```

## Best Practices Summary

### Quick Reference Checklist

**Model Loading:**
- [ ] Verify model file exists and is accessible
- [ ] Check file size and integrity
- [ ] Test with different delegates
- [ ] Validate TFLite operation compatibility

**Performance Issues:**
- [ ] Profile inference time vs expected
- [ ] Check memory usage and GC activity
- [ ] Monitor thermal behavior
- [ ] Verify delegate selection

**Accuracy Issues:**
- [ ] Validate preprocessing pipeline
- [ ] Check confidence thresholds
- [ ] Test with known good images
- [ ] Compare with reference implementation

**Integration Issues:**
- [ ] Verify camera permissions
- [ ] Check image format compatibility
- [ ] Test on multiple devices
- [ ] Validate thread safety

### Emergency Debugging Commands

```bash
# Check TFLite logs
adb logcat | grep -i tensorflow

# Monitor memory usage
adb shell dumpsys meminfo com.yourapp

# Check CPU usage
adb shell top | grep com.yourapp

# Monitor GPU activity
adb shell cat /sys/class/kgsl/kgsl-3d0/gpubusy

# Check thermal state
adb shell cat /sys/class/thermal/thermal_zone0/temp

# Test model loading
adb shell am instrument -w com.yourapp.test/androidx.test.runner.AndroidJUnitRunner

# Monitor camera
adb shell dumpsys media.camera

# Check NNAPI logs
adb logcat | grep -i nnapi
```

### When to Seek Help

- **After trying all troubleshooting steps** in this guide
- **When issues are device-specific** and not reproducible elsewhere
- **For performance regressions** that can't be explained
- **When model accuracy drops significantly** after updates
- **For memory leaks** that resist standard optimization techniques

### Support Resources

- **TFLite Issues**: https://github.com/tensorflow/tensorflow/labels/tf-lite
- **Ultralytics Support**: https://github.com/ultralytics/ultralytics/discussions
- **Android Camera2**: https://developer.android.com/reference/android/hardware/camera2/package-summary
- **Stack Overflow**: Tag questions with `android`, `tensorflow-lite`, `yolo`

---

## Conclusion

This troubleshooting guide covers the most common issues encountered when integrating YOLOv11n models on Android. Most issues can be resolved by following the diagnostic steps and solutions provided. For persistent problems, the advanced debugging tools and systematic approaches will help identify root causes.

Remember to:
1. Always test on multiple devices
2. Use proper logging and monitoring
3. Follow the systematic debugging approach
4. Document solutions for future reference
5. Keep your dependencies up to date

For additional help, refer to the other guides in this collection or consult the official documentation for TensorFlow Lite and Ultralytics YOLO.