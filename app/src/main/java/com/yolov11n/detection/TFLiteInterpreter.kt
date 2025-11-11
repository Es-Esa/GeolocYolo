package com.yolov11n.detection

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import kotlinx.coroutines.*
import org.tensorflow.lite.*
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.model.ModelOptions
import org.tensorflow.lite.support.model.Preference
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Core TFLite interpreter wrapper with GPU/NNAPI acceleration support.
 * Implements hardware acceleration fallback, memory management, and async processing.
 * Based on mobile optimization techniques for real-time YOLO human detection.
 */
class TFLiteInterpreter(
    private val context: Context,
    private val modelConfig: ModelConfig
) {
    private val TAG = "TFLiteInterpreter"
    
    // Core TFLite components
    private var interpreter: Interpreter? = null
    private var imageProcessor: ImageProcessor? = null
    
    // Performance optimization
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: Delegate? = null
    
    // Memory management
    private var inputImageBuffer: TensorImage? = null
    private val bufferPool = mutableListOf<ByteBuffer>()
    private val maxBufferPoolSize = 3
    
    // Threading and async processing
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val isInitialized = AtomicBoolean(false)
    private val currentDelegate = AtomicReference<DelegateType>(DelegateType.CPU)
    
    // Input/output processing
    private var inputTensorWidth = 0
    private var inputTensorHeight = 0
    private var inputTensorChannels = 0
    private var outputTensorSize = 0
    
    // Processing queue for frame skipping
    private val frameQueue = mutableListOf<Bitmap>()
    private val maxQueueSize = 2
    
    enum class DelegateType(val description: String) {
        CPU("CPU (XNNPACK)"),
        GPU("GPU Delegate"),
        NNAPI("NNAPI"),
        HEXAGON("Hexagon DSP")
    }
    
    enum class ProcessResult {
        SUCCESS,
        MODEL_NOT_INITIALIZED,
        INTERPRETER_ERROR,
        MEMORY_ERROR,
        FRAME_SKIPPED
    }
    
    /**
     * Initialize TFLite interpreter with optimal delegate selection
     */
    suspend fun initialize(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Initializing TFLite interpreter with delegate preference: ${modelConfig.preferredDelegate}")
            
            // Load model from assets
            val modelBuffer = loadModelBuffer()
                ?: return@withContext Result.failure(Exception("Failed to load model from assets"))
            
            // Create interpreter options with performance optimizations
            val options = createInterpreterOptions()
            
            // Initialize with best available delegate
            val delegate = selectBestDelegate(options)
            if (delegate == null) {
                Log.w(TAG, "Failed to select delegate, falling back to CPU")
                currentDelegate.set(DelegateType.CPU)
            } else {
                Log.i(TAG, "Selected delegate: ${delegate.description}")
            }
            
            // Initialize interpreter
            interpreter = Interpreter(modelBuffer, options)
            
            // Set thread count for CPU execution
            interpreter?.setNumThreads(modelConfig.threadCount)
            
            // Initialize image processor
            initializeImageProcessor()
            
            // Analyze model input/output shapes
            analyzeModelShapes()
            
            // Initialize memory pool
            initializeBufferPool()
            
            isInitialized.set(true)
            
            Log.i(TAG, "TFLite interpreter initialized successfully. " +
                    "Input tensor: ${inputTensorWidth}x${inputTensorHeight}x${inputTensorChannels}")
            
            Result.success(Unit)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize TFLite interpreter", e)
            Result.failure(e)
        }
    }
    
    /**
     * Process frame with frame skipping for performance optimization
     */
    suspend fun processFrame(bitmap: Bitmap): Result<DetectionResult> = withContext(Dispatchers.Default) {
        if (!isInitialized.get()) {
            return@withContext Result.failure(Exception("Interpreter not initialized"))
        }
        
        // Frame skipping for performance
        if (frameQueue.size >= maxQueueSize) {
            Log.d(TAG, "Frame skipped - queue full")
            return@withContext Result.failure(Exception("Frame skipped"))
        }
        
        try {
            frameQueue.add(bitmap)
            
            val result = tryProcessLatestFrame()
            frameQueue.clear()
            
            result
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame", e)
            Result.failure(e)
        }
    }
    
    /**
     * Process the latest frame in queue
     */
    private suspend fun tryProcessLatestFrame(): Result<DetectionResult> = withContext(Dispatchers.Default) {
        if (frameQueue.isEmpty()) {
            return@withContext Result.failure(Exception("No frames to process"))
        }
        
        val bitmap = frameQueue.last()
        
        try {
            // Preprocess image
            val tensorImage = preprocessImage(bitmap)
            if (tensorImage == null) {
                return@withContext Result.failure(Exception("Failed to preprocess image"))
            }
            
            // Run inference
            val inferenceStart = System.nanoTime()
            val outputs = runInference(tensorImage)
            val inferenceEnd = System.nanoTime()
            val inferenceTimeMs = (inferenceEnd - inferenceStart) / 1_000_000.0
            
            if (outputs == null) {
                return@withContext Result.failure(Exception("Inference failed"))
            }
            
            // Post-process results
            val detections = postProcessResults(outputs)
            
            val result = DetectionResult(
                detections = detections,
                inferenceTimeMs = inferenceTimeMs,
                delegateUsed = currentDelegate.get(),
                inputWidth = inputTensorWidth,
                inputHeight = inputTensorHeight
            )
            
            Result.success(result)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in frame processing pipeline", e)
            Result.failure(e)
        }
    }
    
    /**
     * Run inference on preprocessed image
     */
    private fun runInference(tensorImage: TensorImage): Array<ByteArray>? {
        val interpreter = this.interpreter ?: return null
        
        val inputs = Array(1) { tensorImage.buffer }
        val outputs = arrayOf<ByteArray>(ByteArray(outputTensorSize))
        
        try {
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
            return outputs
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            return null
        }
    }
    
    /**
     * Preprocess image according to model requirements
     */
    private fun preprocessImage(bitmap: Bitmap): TensorImage? {
        try {
            if (inputImageBuffer == null) {
                inputImageBuffer = TensorImage()
            }
            
            inputImageBuffer!!.load(bitmap)
            
            // Resize and pad to match model input size
            val processor = imageProcessor ?: return null
            val processedImage = processor.process(inputImageBuffer)
            
            return processedImage
        } catch (e: Exception) {
            Log.e(TAG, "Image preprocessing failed", e)
            return null
        }
    }
    
    /**
     * Post-process inference results for YOLO output format
     * Expected output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
     */
    private fun postProcessResults(outputs: Array<ByteArray>): List<HumanDetection> {
        val detections = mutableListOf<HumanDetection>()
        
        try {
            val outputData = outputs[0]
            val numDetections = outputData.size / 6
            
            for (i in 0 until numDetections) {
                val offset = i * 6
                
                if (offset + 5 >= outputData.size) break
                
                val confidence = getFloatValue(outputData, offset + 4) / 255.0f
                val classId = getFloatValue(outputData, offset + 5).toInt()
                
                // Filter for human detection only (assuming class 0 is person)
                if (classId == modelConfig.humanClassId && confidence >= modelConfig.confidenceThreshold) {
                    val x1 = getFloatValue(outputData, offset + 0) / 255.0f
                    val y1 = getFloatValue(outputData, offset + 1) / 255.0f
                    val x2 = getFloatValue(outputData, offset + 2) / 255.0f
                    val y2 = getFloatValue(outputData, offset + 3) / 255.0f
                    
                    detections.add(
                        HumanDetection(
                            boundingBox = BoundingBox(
                                x = x1,
                                y = y1,
                                width = x2 - x1,
                                height = y2 - y1
                            ),
                            confidence = confidence,
                            classId = classId,
                            className = "person"
                        )
                    )
                }
            }
            
            // Apply Non-Maximum Suppression (NMS) if needed
            return if (modelConfig.applyNMS) {
                applyNMS(detections, modelConfig.nmsThreshold)
            } else {
                detections
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error post-processing results", e)
            return emptyList()
        }
    }
    
    /**
     * Simple Non-Maximum Suppression implementation
     */
    private fun applyNMS(detections: List<HumanDetection>, iouThreshold: Float): List<HumanDetection> {
        if (detections.isEmpty()) return emptyList()
        
        // Sort by confidence (descending)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val result = mutableListOf<HumanDetection>()
        
        for (detection in sortedDetections) {
            var shouldKeep = true
            
            for (keptDetection in result) {
                val iou = calculateIoU(detection.boundingBox, keptDetection.boundingBox)
                if (iou > iouThreshold) {
                    shouldKeep = false
                    break
                }
            }
            
            if (shouldKeep) {
                result.add(detection)
            }
        }
        
        return result
    }
    
    /**
     * Calculate Intersection over Union (IoU) between two bounding boxes
     */
    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x, box2.x)
        val y1 = maxOf(box1.y, box2.y)
        val x2 = minOf(box1.x + box1.width, box2.x + box2.width)
        val y2 = minOf(box1.y + box1.height, box2.y + box2.height)
        
        if (x2 <= x1 || y2 <= y1) return 0.0f
        
        val intersection = (x2 - x1) * (y2 - y1)
        val union = box1.width * box1.height + box2.width * box2.height - intersection
        
        return if (union > 0) intersection / union else 0.0f
    }
    
    /**
     * Helper function to extract float value from byte array
     */
    private fun getFloatValue(data: ByteArray, offset: Int): Float {
        return ByteBuffer.wrap(data, offset, 4).order(ByteOrder.LITTLE_ENDIAN).float
    }
    
    /**
     * Select the best available delegate based on device capabilities and model config
     */
    private fun selectBestDelegate(options: Interpreter.Options): Delegate? {
        return when (modelConfig.preferredDelegate) {
            DelegateType.GPU -> selectGPUDelegate(options)
            DelegateType.NNAPI -> selectNNAPIDelegate(options)
            DelegateType.HEXAGON -> selectHexagonDelegate(options)
            DelegateType.CPU -> null
        }
    }
    
    /**
     * Select GPU delegate with fallback to CPU
     */
    private fun selectGPUDelegate(options: Interpreter.Options): Delegate? {
        return try {
            val gpuOptions = GpuDelegateOptions.Builder()
                .setPrecisionLossAllowed(false)  // Maintain accuracy
                .setInferencePreference(GpuDelegateOptions.GpuInferencePreference.FAST_SINGLE_ANSWER)
                .build()
            
            GpuDelegate(gpuOptions).also {
                gpuDelegate = it
                currentDelegate.set(DelegateType.GPU)
                Log.i(TAG, "GPU delegate selected successfully")
            }
        } catch (e: Exception) {
            Log.w(TAG, "GPU delegate not available, falling back to CPU", e)
            currentDelegate.set(DelegateType.CPU)
            null
        }
    }
    
    /**
     * Select NNAPI delegate with Android version compatibility checks
     */
    private fun selectNNAPIDelegate(options: Interpreter.Options): Delegate? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            try {
                val nnApiDelegate =NnApiDelegate()
                options.addDelegate(nnApiDelegate)
                currentDelegate.set(DelegateType.NNAPI)
                Log.i(TAG, "NNAPI delegate selected successfully")
                nnApiDelegate
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI delegate not available, falling back to CPU", e)
                currentDelegate.set(DelegateType.CPU)
                null
            }
        } else {
            Log.w(TAG, "NNAPI not available on Android < 8.1, using CPU")
            currentDelegate.set(DelegateType.CPU)
            null
        }
    }
    
    /**
     * Select Hexagon DSP delegate (if available)
     */
    private fun selectHexagonDelegate(options: Interpreter.Options): Delegate? {
        // Hexagon delegate requires separate library
        try {
            // Note: This is a placeholder as actual implementation would require
            // TensorFlow Lite Hexagon delegate library
            Log.i(TAG, "Hexagon delegate not implemented in this version, using CPU")
            currentDelegate.set(DelegateType.CPU)
            return null
        } catch (e: Exception) {
            Log.w(TAG, "Hexagon delegate not available, falling back to CPU", e)
            currentDelegate.set(DelegateType.CPU)
            return null
        }
    }
    
    /**
     * Create interpreter options with performance optimizations
     */
    private fun createInterpreterOptions(): Interpreter.Options {
        return Interpreter.Options().apply {
            // Enable XNNPACK for CPU execution
            setUseXNNPACK(true)
            
            // Set thread count
            setNumThreads(modelConfig.threadCount)
            
            // Add memory optimizations
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                setBufferHandleOutputDirectly(true)
            }
        }
    }
    
    /**
     * Initialize image processor for preprocessing
     */
    private fun initializeImageProcessor() {
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(inputTensorHeight, inputTensorWidth))
            .add(ResizeOp(inputTensorHeight, inputTensorWidth, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(TransformToGrayscaleOp())  // YOLO expects grayscale input
            .build()
    }
    
    /**
     * Analyze model input and output tensor shapes
     */
    private fun analyzeModelShapes() {
        val interpreter = this.interpreter ?: return
        
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        
        // Analyze input tensor
        inputTensor.shape.let { shape ->
            inputTensorHeight = shape[1]
            inputTensorWidth = shape[2]
            inputTensorChannels = shape[3]
            Log.d(TAG, "Input tensor shape: ${shape.contentToString()}")
        }
        
        // Analyze output tensor
        outputTensor.shape.let { shape ->
            outputTensorSize = shape.fold(1) { acc, dim -> acc * dim }
            Log.d(TAG, "Output tensor shape: ${shape.contentToString()}, size: $outputTensorSize")
        }
    }
    
    /**
     * Initialize buffer pool for memory efficiency
     */
    private fun initializeBufferPool() {
        bufferPool.clear()
        repeat(maxBufferPoolSize) {
            val buffer = ByteBuffer.allocateDirect(outputTensorSize)
            buffer.order(ByteOrder.nativeOrder())
            bufferPool.add(buffer)
        }
        Log.d(TAG, "Buffer pool initialized with $maxBufferPoolSize buffers")
    }
    
    /**
     * Load model from assets
     */
    private fun loadModelBuffer(): ByteBuffer? {
        return try {
            val modelBytes = FileUtil.loadMappedFile(context, modelConfig.modelFileName)
            val buffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
            buffer.put(modelBytes)
            buffer.rewind()
            buffer
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model from assets", e)
            null
        }
    }
    
    /**
     * Get current performance metrics
     */
    fun getPerformanceMetrics(): PerformanceMetrics {
        return PerformanceMetrics(
            currentDelegate = currentDelegate.get(),
            isInitialized = isInitialized.get(),
            inputTensorSize = "$inputTensorWidth x $inputTensorHeight x $inputTensorChannels",
            outputTensorSize = outputTensorSize,
            frameQueueSize = frameQueue.size,
            bufferPoolSize = bufferPool.size
        )
    }
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        scope.cancel()
        
        gpuDelegate?.close()
        nnApiDelegate?.close()
        interpreter?.close()
        
        bufferPool.clear()
        frameQueue.clear()
        
        isInitialized.set(false)
        Log.i(TAG, "TFLite interpreter resources cleaned up")
    }
}