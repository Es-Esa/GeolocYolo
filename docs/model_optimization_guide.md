# YOLOv11n Model Optimization Guide for Android Mobile

## Overview

This guide covers advanced optimization techniques for YOLOv11n TFLite models on Android, focusing on quantization, pruning, knowledge distillation, and performance tuning. The goal is to achieve the best balance between model size, inference speed, and detection accuracy for human detection on mobile devices.

## Table of Contents

1. [Model Quantization](#model-quantization)
   - [FP16 Quantization](#fp16-quantization)
   - [INT8 Quantization](#int8-quantization)
   - [Dynamic Range Quantization](#dynamic-range-quantization)
2. [Pruning Techniques](#pruning-techniques)
3. [Knowledge Distillation](#knowledge-distillation)
4. [Mobile-Specific Optimizations](#mobile-specific-optimizations)
5. [Delegate-Specific Optimizations](#delegate-specific-optimizations)
6. [Performance Profiling](#performance-profiling)
7. [Advanced Techniques](#advanced-techniques)

## Model Quantization

### FP16 Quantization

FP16 (Half Precision) is the most straightforward quantization method, reducing model size by approximately 50% while maintaining excellent accuracy.

#### Benefits
- ~50% reduction in model size
- Minimal accuracy loss (<1% mAP)
- Broad hardware support (CPU, GPU, NNAPI)
- No calibration data required

#### Implementation

**Python Conversion:**
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.export(
    format='tflite',
    imgsz=640,
    half=True,           # Enable FP16
    nms=True,
    batch=1
)
```

**Manual TFLite Conversion:**
```python
import tensorflow as tf

# Load and convert
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open("yolov11n_fp16.tflite", "wb") as f:
    f.write(tflite_model)
```

**Android Integration:**
```kotlin
class OptimizedYoloDetector(
    modelPath: String,
    delegateType: DelegateType = DelegateType.CPU,
    numThreads: Int = 4
) {
    
    private val options = Interpreter.Options().apply {
        setNumThreads(numThreads)
        
        when (delegateType) {
            DelegateType.GPU -> addDelegate(GpuDelegate())
            DelegateType.NNAPI -> {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                    addDelegate(NnApiDelegate())
                }
            }
            else -> { /* CPU default */ }
        }
    }
    
    enum class DelegateType { CPU, GPU, NNAPI }
}
```

### INT8 Quantization

INT8 quantization provides the most aggressive compression and speed improvements, but requires careful calibration to maintain accuracy.

#### Benefits
- ~75% reduction in model size
- Significant speed improvements on supported hardware
- Lower power consumption
- Best for resource-constrained devices

#### Calibration Dataset Requirements

For human detection INT8 quantization, your calibration dataset should include:

- **Distance variations**: Close, medium, and far humans
- **Pose variations**: Standing, walking, running, sitting
- **Lighting conditions**: Day, night, indoor, outdoor, mixed lighting
- **Crowding scenarios**: Single person, small groups, large crowds
- **Occlusions**: Partial body visibility, overlapping people
- **Background variety**: Urban, natural, indoor environments

#### Implementation

**1. Prepare Calibration Dataset:**
```python
import os
import numpy as np
from PIL import Image

def prepare_calibration_dataset(image_dir, output_dir, num_samples=500, input_size=640):
    """Prepare representative calibration dataset."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    # Sample representative images
    np.random.seed(42)
    selected_files = np.random.choice(image_files, 
                                    min(num_samples, len(image_files)), 
                                    replace=False)
    
    calibration_samples = []
    
    for i, img_path in enumerate(selected_files):
        # Load and preprocess
        image = Image.open(img_path)
        
        # Resize and normalize (YOLO preprocessing)
        resized = image.resize((input_size, input_size), Image.Resampling.LANCZOS)
        np_image = np.array(resized).astype(np.float32) / 255.0
        
        calibration_samples.append(np_image)
        
        # Save sample
        output_path = os.path.join(output_dir, f"calib_{i:04d}.jpg")
        Image.fromarray((np_image * 255).astype(np.uint8)).save(output_path)
    
    return calibration_samples

# Usage
calibration_dir = "calibration_samples"
prepare_calibration_dataset("raw_images", calibration_dir, num_samples=500)
```

**2. Convert with INT8:**
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.export(
    format='tflite',
    imgsz=640,
    int8=True,           # Enable INT8
    nms=True,
    batch=1,
    data=calibration_dir  # Path to calibration data
)
```

**3. Advanced INT8 with Per-Channel Quantization:**
```python
import tensorflow as tf

def convert_to_int8_per_channel(saved_model_path, calibration_data, output_path):
    """Convert with per-channel quantization for better accuracy."""
    
    def representative_dataset_gen():
        for image in calibration_data[:200]:  # Use 200 samples
            # Add batch dimension
            yield [np.expand_dims(image, axis=0)]
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset
    converter.representative_dataset = representative_dataset_gen
    
    # Enable full-int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Set input/output types
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Quantize weights per channel
    converter._experimental_calibrate_only = False
    converter._experimental_new_converter = True
    
    # Convert
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    return output_path
```

### Dynamic Range Quantization

For scenarios where full INT8 is too aggressive, dynamic range quantization provides a middle ground.

#### Benefits
- ~50% model size reduction
- No calibration data required
- Faster inference than FP32
- Good accuracy retention

#### Implementation
```python
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16, tf.int8]
tflite_model = converter.convert()
```

## Pruning Techniques

Network pruning removes less important weights or entire channels to reduce model complexity and improve inference speed.

### Magnitude-Based Pruning

Remove weights with the smallest absolute values:

```python
import tensorflow_model_optimization as tfmot

def apply_magnitude_pruning(model, pruning_percentage=0.2):
    """Apply magnitude-based pruning to reduce model size."""
    
    # Define pruning schedule
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=pruning_percentage,
            begin_step=0,
            end_step=1000
        )
    }
    
    # Apply pruning to model layers
    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer
    
    # Prune the model
    pruned_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_dense,
        input_tensors=None
    )
    
    return pruned_model

# Train with pruning
pruned_model = apply_magnitude_pruning(yolo_model, pruning_percentage=0.3)
pruned_model.compile(optimizer='adam', loss='mse')
pruned_model.fit(calibration_data, epochs=50)

# Strip pruning wrappers and convert
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```

### Structured Pruning

Remove entire channels for better hardware efficiency:

```python
def apply_structured_pruning(model, pruning_percentage=0.2):
    """Apply structured pruning to remove entire channels."""
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=pruning_percentage,
            begin_step=0,
            end_step=2000
        )
    }
    
    # For each Conv2D layer, add pruning wrapper
    pruning_wrapper = tfmot.sparsity.keras.PruneLowMagnitude
    
    def add_pruning_to_conv2d(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return pruning_wrapper(layer, **pruning_params)
        return layer
    
    pruned_model = tf.keras.models.clone_model(
        model,
        clone_function=add_pruning_to_conv2d
    )
    
    return pruned_model
```

## Knowledge Distillation

Train a smaller student model to mimic a larger teacher model's behavior:

```python
class DistillationTrainer:
    """Train a smaller YOLOv11n model using knowledge distillation."""
    
    def __init__(self, teacher_model_path, student_model_path):
        self.teacher = YOLO(teacher_model_path)
        self.student = YOLO(student_model_path)
        
    def distillation_loss(self, student_outputs, teacher_outputs, true_labels, 
                         alpha=0.5, temperature=4.0):
        """
        Combined distillation and task loss.
        
        Args:
            student_outputs: Student model predictions
            teacher_outputs: Teacher model predictions  
            true_labels: Ground truth labels
            alpha: Weight of distillation loss vs task loss
            temperature: Temperature for soft targets
        """
        # Convert logits to probabilities with temperature
        student_probs = tf.nn.softmax(student_outputs / temperature)
        teacher_probs = tf.nn.softmax(teacher_outputs / temperature)
        
        # KL divergence loss (distillation loss)
        kl_loss = tf.keras.losses.KLDivergence()(
            tf.stop_gradient(teacher_probs), student_probs
        ) * (temperature ** 2)
        
        # Task loss (mse for bbox, cross-entropy for classification)
        task_loss = self.task_loss(student_outputs, true_labels)
        
        # Combined loss
        total_loss = alpha * kl_loss + (1 - alpha) * task_loss
        return total_loss
    
    def train_student(self, dataset, epochs=100, batch_size=16):
        """Train student model with distillation."""
        
        for epoch in range(epochs):
            for batch in dataset.take(1000):  # Limit for efficiency
                images, labels = batch
                
                # Teacher predictions (no gradients)
                with tf.GradientTape(persistent=True) as teacher_tape:
                    teacher_outputs = self.teacher(images, training=True)
                    student_outputs = self.student(images, training=True)
                    
                    loss = self.distillation_loss(
                        student_outputs, teacher_outputs, labels
                    )
                
                # Update student
                gradients = teacher_tape.gradient(
                    loss, self.student.trainable_variables
                )
                self.student.optimizer.apply_gradients(
                    zip(gradients, self.student.trainable_variables)
                )
```

## Mobile-Specific Optimizations

### Input Size Optimization

Optimize input resolution based on your use case:

```python
# Input size recommendations by use case
INPUT_SIZE_CONFIG = {
    "ultra_fast": {
        "size": 320,
        "description": "Real-time detection on low-end devices",
        "expected_fps": "30+",
        "accuracy_impact": "-3% to -5%"
    },
    "balanced": {
        "size": 416,
        "description": "Good balance of speed and accuracy",
        "expected_fps": "20-30", 
        "accuracy_impact": "-1% to -2%"
    },
    "high_accuracy": {
        "size": 640,
        "description": "Best accuracy, slower inference",
        "expected_fps": "10-20",
        "accuracy_impact": "baseline"
    },
    "custom": {
        "size": 512,
        "description": "Custom size based on testing",
        "expected_fps": "variable",
        "accuracy_impact": "measured"
    }
}

def select_optimal_input_size(device_specs: Dict, use_case: str) -> int:
    """Select optimal input size based on device capabilities."""
    
    if use_case == "custom":
        return device_specs.get("custom_size", 416)
    
    config = INPUT_SIZE_CONFIG.get(use_case, INPUT_SIZE_CONFIG["balanced"])
    
    # Adjust based on device specs
    if device_specs.get("ram_gb", 4) < 4:  # Low RAM
        return min(config["size"], 416)
    
    if device_specs.get("cpu_cores", 4) <= 4:  # Limited CPU
        return min(config["size"], 416)
    
    if device_specs.get("has_nnapi", False):  # Has accelerator
        return config["size"]
    
    return config["size"]
```

### Batch Processing Optimization

For applications that can process multiple frames:

```python
class BatchProcessor:
    """Optimize inference with intelligent batching."""
    
    def __init__(self, model, max_batch_size=4, max_latency_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        
    def process_batch(self, frames: List[Image], timeout_ms: int = 20) -> List[DetectionResult]:
        """Process frames in optimal batch size based on latency constraints."""
        
        batch_size = self._calculate_optimal_batch_size(frames, timeout_ms)
        results = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = self._process_batch_inference(batch)
            results.extend(batch_results)
            
        return results
    
    def _calculate_optimal_batch_size(self, frames: List[Image], timeout_ms: int) -> int:
        """Calculate batch size that meets latency requirements."""
        
        # Start with single frame
        start_time = time.time()
        self.model.run(np.expand_dims(frames[0], axis=0))
        single_frame_time = (time.time() - start_time) * 1000
        
        # Calculate maximum batch size based on latency budget
        max_batch_by_latency = int(timeout_ms / single_frame_time)
        max_batch_by_memory = self._estimate_max_batch_size(frames[0].shape)
        
        return min(self.max_batch_size, max_batch_by_latency, max_batch_by_memory)
```

### Memory Optimization

```kotlin
class MemoryOptimizedDetector {
    
    // Object pools to avoid allocations
    private val bitmapPool = Pool<Bitmap> { createBitmap() }
    private val byteBufferPool = Pool<ByteBuffer> { 
        ByteBuffer.allocateDirect(1 * 640 * 640 * 3 * 4).order(ByteOrder.nativeOrder())
    }
    
    // Pre-allocated arrays
    private val preprocessedArray = FloatArray(640 * 640 * 3)
    private val outputArray = Array(1) { Array(300) { FloatArray(6) } }
    
    fun detectWithMemoryOptimization(image: Image): List<DetectionResult> {
        // Reuse bitmap from pool
        val bitmap = bitmapPool.acquire()
        try {
            convertYuvToBitmap(image, bitmap)
            
            // Reuse byte buffer
            val inputBuffer = byteBufferPool.acquire()
            try {
                preprocessIntoBuffer(bitmap, inputBuffer, preprocessedArray)
                
                // Use pre-allocated output array
                model.run(inputBuffer, outputArray)
                
                return postprocessResults(outputArray[0])
            } finally {
                byteBufferPool.release(inputBuffer)
            }
        } finally {
            bitmapPool.release(bitmap)
        }
    }
    
    // Object pool implementation
    private class Pool<T>(private val factory: () -> T) {
        private val available = ArrayDeque<T>()
        private val inUse = mutableSetOf<T>()
        
        fun acquire(): T {
            return if (available.isNotEmpty()) {
                available.removeFirst().also { inUse.add(it) }
            } else {
                factory().also { inUse.add(it) }
            }
        }
        
        fun release(item: T) {
            if (inUse.remove(item)) {
                available.addLast(item)
            }
        }
    }
}
```

## Delegate-Specific Optimizations

### GPU Delegate Optimization

```kotlin
class GpuOptimizedDetector : YOLOv11nHumanDetector() {
    
    init {
        setupGpuDelegate()
    }
    
    private fun setupGpuDelegate() {
        val options = Interpreter.Options().apply {
            // Use 4-channel input for GPU efficiency
            setUseNNAPI(false) // Disable NNAPI to force GPU
            
            // GPU-specific optimizations
            addDelegate(GpuDelegate().apply {
                // Enable precision options
                val delegateOptions = GpuDelegateOptions.Builder()
                    .setPrecisionLossAllowed(true) // Allow FP16/FP32 mixing
                    .setInferencePreference(GpuInferencePreference.FAST_SINGLE_ANSWER)
                    .build()
                
                // Set memory fraction
                delegateOptions.setMemoryFraction(0.6f)
            })
            
            // CPU fallback for unsupported ops
            setUseXNNPack(true)
        }
        
        this.options = options
    }
    
    // Optimize preprocessing for GPU
    override fun preprocessImage(image: Image, targetWidth: Int, targetHeight: Int): ByteBuffer {
        // Use RGBA format for GPU efficiency
        val rgbaBitmap = createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        convertYuvToRgba(image, rgbaBitmap)
        
        // Resize while maintaining aspect ratio
        val scaledBitmap = Bitmap.createScaledBitmap(rgbaBitmap, targetWidth, targetHeight, true)
        
        // Convert to 4-channel buffer (GPU-optimized)
        val buffer = ByteBuffer.allocateDirect(4 * targetWidth * targetHeight)
        buffer.order(ByteOrder.nativeOrder())
        
        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val pixel = scaledBitmap.getPixel(x, y)
                buffer.putFloat(((pixel shr 16 and 0xFF) / 255.0f)) // R
                buffer.putFloat(((pixel shr 8 and 0xFF) / 255.0f))  // G
                buffer.putFloat((pixel and 0xFF) / 255.0f)         // B
                buffer.putFloat(1.0f)                              // A (alpha)
            }
        }
        buffer.rewind()
        return buffer
    }
}
```

### NNAPI Optimization

```kotlin
class NnApiOptimizedDetector : YOLOv11nHumanDetector() {
    
    init {
        setupNnApiDelegate()
    }
    
    private fun setupNnApiDelegate() {
        val options = Interpreter.Options().apply {
            // NNAPI-specific settings
            val nnApiOptions = NnApiDelegate.Options()
            
            // Execution preference for camera streams
            nnApiOptions.setExecutionPreference(
                NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED
            )
            
            // Enable NNAPI memory optimization
            nnApiOptions.setUseNnApiCpu(true) // Allow CPU fallback
            nnApiOptions.setPriority(NnApiDelegate.Options.FAST_SINGLE_ANSWER)
            
            addDelegate(NnApiDelegate(nnApiOptions))
            
            // Thread configuration for NNAPI
            setNumThreads(1) // Let NNAPI handle threading
        }
    }
    
    // NNAPI-optimized preprocessing
    override fun preprocessImage(image: Image, targetWidth: Int, targetHeight: Int): ByteBuffer {
        // Convert to the format preferred by NNAPI
        val yuvBytes = extractYuvBytes(image)
        val nv21Buffer = ByteBuffer.allocateDirect(yuvBytes.size)
        nv21Buffer.put(yuvBytes)
        nv21Buffer.rewind()
        
        return nv21Buffer
    }
}
```

## Performance Profiling

### Comprehensive Performance Measurement

```python
class PerformanceProfiler:
    """Comprehensive performance profiling for YOLOv11n models."""
    
    def __init__(self, model_path: str, test_images_dir: str):
        self.model_path = model_path
        self.test_images_dir = test_images_dir
        self.results = {}
        
    def profile_quantization_methods(self, test_images: List[str]) -> Dict[str, Dict]:
        """Profile different quantization methods."""
        
        quantization_methods = {
            'fp32': {'half': False, 'int8': False},
            'fp16': {'half': True, 'int8': False}, 
            'int8_basic': {'half': False, 'int8': True},
            'int8_per_channel': {'half': False, 'int8': True}
        }
        
        results = {}
        
        for method_name, quant_args in quantization_methods.items():
            logger.info(f"Testing {method_name} quantization...")
            
            # Convert model with this quantization
            model_path = self.convert_with_quantization(**quant_args)
            
            # Profile performance
            performance = self.profile_model(model_path, test_images)
            results[method_name] = performance
            
            # Clean up
            os.remove(model_path)
        
        return results
    
    def profile_model(self, model_path: str, test_images: List[str]) -> Dict:
        """Profile a single model configuration."""
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()
        
        # Warm up
        for _ in range(10):
            dummy_input = np.random.random(input_details['shape']).astype(input_details['dtype'])
            interpreter.set_tensor(input_details['index'], dummy_input)
            interpreter.invoke()
        
        # Measure inference times
        inference_times = []
        memory_usage = []
        
        for image_path in test_images:
            # Load and preprocess
            image = self.preprocess_image(image_path, input_details['shape'])
            
            # Measure inference time
            start_time = time.perf_counter()
            interpreter.set_tensor(input_details['index'], image)
            interpreter.invoke()
            end_time = time.perf_counter()
            
            inference_times.append((end_time - start_time) * 1000)  # ms
            
            # Get outputs
            outputs = []
            for output_detail in output_details:
                output = interpreter.get_tensor(output_detail['index'])
                outputs.append(output)
        
        # Calculate statistics
        stats = {
            'mean_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'p50_inference_time_ms': np.percentile(inference_times, 50),
            'p95_inference_time_ms': np.percentile(inference_times, 95),
            'p99_inference_time_ms': np.percentile(inference_times, 99),
            'fps': 1000.0 / np.mean(inference_times),
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
        
        return stats
    
    def generate_performance_report(self, results: Dict) -> str:
        """Generate a detailed performance comparison report."""
        
        report = """
# YOLOv11n Performance Analysis Report

## Model Performance Comparison

| Quantization | Avg Latency (ms) | P95 Latency (ms) | FPS | Model Size (MB) |
|-------------|-----------------|-----------------|-----|----------------|
"""
        
        for method, stats in results.items():
            report += f"| {method} | {stats['mean_inference_time_ms']:.1f} | "
            report += f"{stats['p95_inference_time_ms']:.1f} | {stats['fps']:.1f} | "
            report += f"{stats['model_size_mb']:.1f} |\n"
        
        report += """
## Recommendations

Based on the performance analysis:

"""
        
        # Add recommendations
        fastest = min(results.items(), key=lambda x: x[1]['mean_inference_time_ms'])
        smallest = min(results.items(), key=lambda x: x[1]['model_size_mb'])
        best_fps = max(results.items(), key=lambda x: x[1]['fps'])
        
        report += f"- **Fastest**: {fastest[0]} ({fastest[1]['mean_inference_time_ms']:.1f}ms)\n"
        report += f"- **Smallest**: {smallest[0]} ({smallest[1]['model_size_mb']:.1f}MB)\n"
        report += f"- **Highest FPS**: {best_fps[0]} ({best_fps[1]['fps']:.1f} FPS)\n\n"
        
        return report
```

### Android Performance Testing

```kotlin
class PerformanceBenchmark {
    
    data class BenchmarkResult(
        val method: String,
        val avgLatencyMs: Double,
        val p95LatencyMs: Double,
        val fps: Double,
        val memoryUsageMb: Float,
        val modelSizeMb: Float
    )
    
    suspend fun benchmarkQuantizationMethods(context: Context): List<BenchmarkResult> {
        val methods = listOf("fp32", "fp16", "int8")
        val results = mutableListOf<BenchmarkResult>()
        
        for (method in methods) {
            val result = benchmarkSingleMethod(context, method)
            results.add(result)
        }
        
        return results
    }
    
    private suspend fun benchmarkSingleMethod(context: Context, method: String): BenchmarkResult {
        // Load model for this method
        val modelName = "yolov11n_$method.tflite"
        val detector = createDetectorForMethod(context, modelName)
        
        val latencyMeasurements = mutableListOf<Long>()
        var memoryBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
        
        // Run inference on test images
        val testImages = loadTestImages(context)
        
        repeat(100) { iteration ->
            val startTime = System.nanoTime()
            
            for (image in testImages) {
                val detections = detector.detectHumans(image, 0)
                // Process detections (but don't store results to avoid memory buildup)
            }
            
            val endTime = System.nanoTime()
            latencyMeasurements.add(endTime - startTime)
            
            // Give system time to stabilize
            delay(10)
        }
        
        var memoryAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
        val memoryUsageMb = ((memoryAfter - memoryBefore) / 1024 / 1024).toFloat()
        
        // Calculate statistics
        val latencyMs = latencyMeasurements.map { it / 1_000_000.0 }
        val avgLatency = latencyMs.average()
        val p95Index = (latencyMs.size * 0.95).toInt()
        val sortedLatencies = latencyMs.sorted()
        val p95Latency = sortedLatencies[p95Index]
        val fps = 1000.0 / avgLatency
        
        detector.close()
        
        return BenchmarkResult(
            method = method,
            avgLatencyMs = avgLatency,
            p95LatencyMs = p95Latency,
            fps = fps,
            memoryUsageMb = memoryUsageMb,
            modelSizeMb = getModelSizeMb(context, "yolov11n_$method.tflite")
        )
    }
}
```

## Advanced Techniques

### Adaptive Model Loading

```kotlin
class AdaptiveModelLoader(private val context: Context) {
    
    data class DeviceCapabilities(
        val hasGpu: Boolean,
        val hasNnapi: Boolean,
        val cpuCores: Int,
        val ramGb: Int,
        val androidVersion: Int
    )
    
    fun getOptimalModel(): String {
        val capabilities = detectDeviceCapabilities()
        
        return when {
            capabilities.hasGpu && capabilities.ramGb >= 4 -> "yolov11n_fp16.tflite"
            capabilities.hasNnapi && capabilities.cpuCores >= 6 -> "yolov11n_int8.tflite"
            capabilities.ramGb >= 3 -> "yolov11n_fp16.tflite"
            else -> "yolov11n_int8.tflite"
        }
    }
    
    private fun detectDeviceCapabilities(): DeviceCapabilities {
        val packageManager = context.packageManager
        val hasGpu = GpuDelegateHelper.isGpuDelegateSupported()
        val hasNnapi = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1
        val cpuCores = Runtime.getRuntime().availableProcessors()
        
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val ramGb = (memoryInfo.totalMem / 1024 / 1024 / 1024).toInt()
        
        return DeviceCapabilities(
            hasGpu = hasGpu,
            hasNnapi = hasNnapi,
            cpuCores = cpuCores,
            ramGb = ramGb,
            androidVersion = Build.VERSION.SDK_INT
        )
    }
}
```

### Dynamic Inference Scaling

```kotlin
class DynamicInferenceScaler {
    
    private var currentPerformanceLevel = PerformanceLevel.BALANCED
    
    enum class PerformanceLevel(val targetFps: Int, val confidenceThreshold: Float) {
        POWER_SAVER(targetFps = 10, confidenceThreshold = 0.7f),
        BALANCED(targetFps = 20, confidenceThreshold = 0.5f),
        HIGH_PERFORMANCE(targetFps = 30, confidenceThreshold = 0.3f)
    }
    
    fun adaptToPerformance(fps: Float, thermalState: ThermalState) {
        when {
            fps < currentPerformanceLevel.targetFps * 0.8 || 
            thermalState == ThermalState.THROTTLING -> {
                // Reduce performance requirements
                currentPerformanceLevel = when (currentPerformanceLevel) {
                    PerformanceLevel.HIGH_PERFORMANCE -> PerformanceLevel.BALANCED
                    PerformanceLevel.BALANCED -> PerformanceLevel.POWER_SAVER
                    else -> PerformanceLevel.POWER_SAVER
                }
            }
            fps > currentPerformanceLevel.targetFps * 1.2 -> {
                // Increase performance requirements
                currentPerformanceLevel = when (currentPerformanceLevel) {
                    PerformanceLevel.POWER_SAVER -> PerformanceLevel.BALANCED
                    PerformanceLevel.BALANCED -> PerformanceLevel.HIGH_PERFORMANCE
                    else -> PerformanceLevel.HIGH_PERFORMANCE
                }
            }
        }
    }
    
    fun shouldProcessFrame(frame: FrameInfo): Boolean {
        return when (currentPerformanceLevel) {
            PerformanceLevel.POWER_SAVER -> frame.sequenceNumber % 3 == 0  // Every 3rd frame
            PerformanceLevel.BALANCED -> frame.sequenceNumber % 2 == 0     // Every 2nd frame  
            PerformanceLevel.HIGH_PERFORMANCE -> true                    // All frames
        }
    }
}
```

## Best Practices Summary

1. **Start with FP16**: It provides the best balance of performance and accuracy with minimal setup
2. **Use INT8 for resource-constrained devices**: Requires careful calibration but delivers the best compression
3. **Test on target devices**: Performance varies significantly across different Android devices
4. **Monitor thermal behavior**: High performance modes can trigger throttling on mobile devices
5. **Profile memory usage**: Object detection models can consume significant memory
6. **Use appropriate delegates**: GPU for compute-heavy models, NNAPI for broad compatibility
7. **Implement adaptive performance**: Dynamically adjust based on device state and requirements

---

## Next Steps

- See [Performance Benchmarking Guide](performance_benchmarking.md) for testing strategies
- See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues
- Review the [Model Integration Guide](MODEL_INTEGRATION.md) for implementation details