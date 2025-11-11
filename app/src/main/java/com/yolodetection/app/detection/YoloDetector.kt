package com.yolodetection.app.detection

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.*
import android.os.SystemClock
import com.yolodetection.app.detection.models.BoundingBox
import com.yolodetection.app.detection.models.Detection
import com.yolodetection.app.detection.models.ModelInfo
import com.yolodetection.app.utils.Constants
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import timber.log.Timber
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

/**
 * YOLOv11n Human Detection Model Handler
 * 
 * Handles TensorFlow Lite model loading, inference, and output post-processing
 * Optimized for real-time human detection on Android devices
 */
class YoloDetector(private val context: Context) {
    
    // TFLite components
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null
    
    // Model configuration
    private var modelInfo: ModelInfo? = null
    private val inputSize = Constants.MODEL_INPUT_SIZE
    private val confidenceThreshold = Constants.CONFIDENCE_THRESHOLD
    private val iouThreshold = Constants.IOU_THRESHOLD
    
    // Preprocessing utilities
    private var inputImageBuffer: ByteBuffer? = null
    private var outputBuffer: Array<ByteBuffer>? = null
    private val imageProcessor = ImageProcessor.Builder().build()
    
    // Performance tracking
    private var inferenceCount = 0
    private var totalInferenceTime = 0L
    private var lastInferenceTime = 0L
    
    /**
     * Initialize the YOLO detector with model loading and delegate setup
     */
    suspend fun initialize() = withContext(Dispatchers.Default) {
        try {
            Timber.i("Initializing YOLOv11n Human Detection Model")
            
            // Load model from assets
            val modelBuffer = loadModelFile()
            
            // Setup TFLite options
            val options = Interpreter.Options().apply {
                // Performance optimization
                setNumThreads(4)
                setUseNNAPI(BuildConfig.USE_NNAPI)
                setUseXNNPack(true)
                
                // Enable GPU delegate if available
                if (BuildConfig.USE_GPU) {
                    gpuDelegate = GpuDelegate().apply {
                        setOptions(GpuDelegate.Options().apply {
                            setPrecisionLossAllowed(true)
                            setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                        })
                        addDelegate(this)
                    }
                }
                
                // Enable NNAPI
                if (BuildConfig.USE_NNAPI) {
                    nnapiDelegate = NnApiDelegate()
                    addDelegate(nnapiDelegate)
                }
                
                // Model quantization and optimization
                setUseXNNPack(true)
                setUseBufferHandle(false)
            }
            
            // Create interpreter
            interpreter = Interpreter(modelBuffer, options)
            
            // Initialize model info and buffers
            initializeModelInfo()
            initializeBuffers()
            
            Timber.i("YOLOv11n detector initialized successfully with ${getDelegateInfo()}")
            Timber.i("Model input size: $inputSize, threads: 4")
            
        } catch (e: Exception) {
            Timber.e(e, "Failed to initialize YOLO detector")
            throw IllegalStateException("YOLO detector initialization failed: ${e.message}", e)
        }
    }
    
    /**
     * Process an image and return human detections
     */
    suspend fun detect(image: Bitmap, rotation: Float = 0f): List<Detection> = withContext(Dispatchers.Default) {
        val startTime = SystemClock.elapsedRealtime()
        
        try {
            // Preprocess image
            val preprocessedImage = preprocessImage(image, rotation)
            
            // Run inference
            val inferenceStart = SystemClock.elapsedRealtime()
            interpreter?.run(preprocessedImage, outputBuffer)
            val inferenceEnd = SystemClock.elapsedRealtime()
            
            // Post-process results
            val detections = postProcessResults()
            
            // Update performance metrics
            val endTime = SystemClock.elapsedRealtime()
            updatePerformanceMetrics(endTime - startTime, inferenceEnd - inferenceStart)
            
            Timber.d("Detection completed: ${detections.size} objects found in ${endTime - startTime}ms")
            return@withContext detections
            
        } catch (e: Exception) {
            Timber.e(e, "Error during detection")
            return@withContext emptyList()
        }
    }
    
    /**
     * Get model information
     */
    fun getModelInfo(): ModelInfo? = modelInfo
    
    /**
     * Get performance statistics
     */
    fun getPerformanceStats(): PerformanceStats {
        return PerformanceStats(
            totalInferences = inferenceCount,
            averageInferenceTime = if (inferenceCount > 0) totalInferenceTime / inferenceCount else 0,
            lastInferenceTime = lastInferenceTime,
            framesPerSecond = if (inferenceCount > 0) (1000 * inferenceCount / totalInferenceTime).toInt() else 0
        )
    }
    
    /**
     * Set confidence threshold
     */
    fun setConfidenceThreshold(threshold: Float) {
        // Update confidence threshold for new detections
        // This would be used in post-processing
    }
    
    /**
     * Enable or disable specific classes (for human-only detection)
     */
    fun setClassFilter(enabledClasses: Set<Int>) {
        // Implement class filtering for human detection
    }
    
    private fun loadModelFile(): MappedByteBuffer {
        val modelFileName = getModelFileName()
        return FileUtil.loadMappedFile(context, modelFileName) as MappedByteBuffer
    }
    
    private fun getModelFileName(): String {
        // Return appropriate model file based on quantization
        return if (isNNAPISupported()) {
            "yolov11n_int8.tflite"  // Use INT8 model for NNAPI
        } else {
            "yolov11n_fp16.tflite"  // Use FP16 model for GPU/CPU
        }
    }
    
    private fun isNNAPISupported(): Boolean {
        return BuildConfig.USE_NNAPI && NnApiDelegate.isAvailable()
    }
    
    private fun getDelegateInfo(): String {
        val delegates = mutableListOf<String>()
        if (interpreter?.usedModules?.contains("XNNPACK") == true) delegates.add("XNNPACK")
        if (gpuDelegate != null) delegates.add("GPU")
        if (nnapiDelegate != null) delegates.add("NNAPI")
        return delegates.joinToString("+")
    }
    
    private fun initializeModelInfo() {
        modelInfo = ModelInfo(
            name = "YOLOv11n",
            version = "11",
            inputSize = inputSize,
            numClasses = 80, // COCO dataset classes
            hasNMS = true, // Based on research, models with embedded NMS
            confidenceThreshold = confidenceThreshold,
            iouThreshold = iouThreshold
        )
    }
    
    private fun initializeBuffers() {
        // Initialize input buffer
        inputImageBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        inputImageBuffer?.order(ByteOrder.nativeOrder())
        
        // Initialize output buffer based on model type
        // This depends on whether NMS is embedded in the model
        if (modelInfo?.hasNMS == true) {
            // Model returns direct detections
            outputBuffer = arrayOf(ByteBuffer.allocateDirect(1 * 300 * 6)) // [1, 300, 6] = [batch, max_detections, (x1,y1,x2,y2,score,class)]
        } else {
            // Model returns raw YOLO outputs
            outputBuffer = arrayOf(
                ByteBuffer.allocateDirect(1 * 25200 * 85), // Raw outputs
                ByteBuffer.allocateDirect(1 * 6300 * 85),  // Raw outputs
                ByteBuffer.allocateDirect(1 * 1575 * 85)   // Raw outputs
            )
        }
    }
    
    private fun preprocessImage(image: Bitmap, rotation: Float): ByteBuffer {
        // Convert bitmap to tensor input
        val targetBitmap = Bitmap.createScaledBitmap(image, inputSize, inputSize, true)
        
        // Create tensor image for processing
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(targetBitmap)
        
        // Apply preprocessing operations
        imageProcessor.process(tensorImage).also { processedImage ->
            // Get the processed buffer
            return processedImage.buffer
        }
    }
    
    private fun postProcessResults(): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        if (modelInfo?.hasNMS == true) {
            // Parse direct detections from output
            val detectionBuffer = outputBuffer?.get(0) ?: return emptyList()
            detectionBuffer.rewind()
            
            val maxDetections = 300
            val valuesPerDetection = 6 // x1, y1, x2, y2, score, class
            
            for (i in 0 until maxDetections) {
                if (detectionBuffer.remaining() >= valuesPerDetection * 4) {
                    val x1 = detectionBuffer.float
                    val y1 = detectionBuffer.float
                    val x2 = detectionBuffer.float
                    val y2 = detectionBuffer.float
                    val score = detectionBuffer.float
                    val classId = detectionBuffer.int
                    
                    if (score > confidenceThreshold && classId == 0) { // Person class
                        val boundingBox = BoundingBox(
                            x1 = x1,
                            y1 = y1,
                            x2 = x2,
                            y2 = y2
                        )
                        
                        detections.add(Detection(
                            boundingBox = boundingBox,
                            confidence = score,
                            classId = classId,
                            className = "person"
                        ))
                    }
                }
            }
        } else {
            // Implement custom NMS if model doesn't have embedded NMS
            // This would require parsing raw YOLO outputs
        }
        
        return detections
    }
    
    private fun updatePerformanceMetrics(totalTime: Long, inferenceTime: Long) {
        inferenceCount++
        totalInferenceTime += inferenceTime
        lastInferenceTime = inferenceTime
        
        // Log performance every 100 inferences
        if (inferenceCount % 100 == 0) {
            val avgInferenceTime = totalInferenceTime / inferenceCount
            Timber.i("Performance: avg=${avgInferenceTime}ms, fps=${1000 * inferenceCount / totalInferenceTime}")
        }
    }
    
    fun cleanup() {
        try {
            interpreter?.close()
            gpuDelegate?.close()
            nnapiDelegate?.close()
            
            Timber.i("YOLO detector cleaned up successfully")
        } catch (e: Exception) {
            Timber.e(e, "Error during cleanup")
        }
    }
    
    /**
     * Performance statistics data class
     */
    data class PerformanceStats(
        val totalInferences: Int,
        val averageInferenceTime: Long,
        val lastInferenceTime: Long,
        val framesPerSecond: Int
    )
}