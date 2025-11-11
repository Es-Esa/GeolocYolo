package com.yolov11n.detection

import android.graphics.*
import android.util.Log
import kotlinx.coroutines.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * Human Detection Processor - handles preprocessing and postprocessing pipeline
 * Implements frame skipping, memory management, and performance optimization techniques
 * based on mobile optimization research findings.
 */
class HumanDetectionProcessor(
    private val modelConfig: ModelConfig
) {
    private val TAG = "HumanDetectionProcessor"
    
    // Memory management - reuse buffers to prevent GC pressure
    private val reusableByteBuffer = ThreadLocal<ByteBuffer?> { null }
    private val reusableBitmap = ThreadLocal<Bitmap?> { null }
    
    // Frame skipping and performance monitoring
    private val frameSkipCount = 0
    private var lastProcessTime = 0L
    private val targetFrameInterval = 1000L / modelConfig.targetFPS
    
    // Preprocessing configuration
    private val inputSize = modelConfig.inputImageSize
    private val keepAspectRatio = modelConfig.keepAspectRatio
    
    // Postprocessing thresholds
    private val confidenceThreshold = modelConfig.confidenceThreshold
    private val nmsThreshold = modelConfig.nmsThreshold
    private val humanClassId = modelConfig.humanClassId
    
    /**
     * Preprocess image for YOLO input requirements
     * Implements efficient memory management and aspect ratio preservation
     */
    fun preprocessImage(bitmap: Bitmap): Bitmap? {
        return try {
            val startTime = System.nanoTime()
            
            // Get or create reusable bitmap
            val targetBitmap = getReusableBitmap(inputSize, inputSize)
            
            // Apply letterboxing and resizing
            val processedBitmap = applyLetterboxing(bitmap, targetBitmap)
            
            if (processedBitmap == null) {
                Log.e(TAG, "Failed to apply letterboxing to image")
                return null
            }
            
            // Convert to grayscale if required by model
            val finalBitmap = if (modelConfig.convertToGrayscale) {
                convertToGrayscale(processedBitmap)
            } else {
                processedBitmap
            }
            
            val endTime = System.nanoTime()
            Log.d(TAG, "Preprocessing completed in ${(endTime - startTime) / 1_000_000}ms")
            
            finalBitmap
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during image preprocessing", e)
            null
        }
    }
    
    /**
     * Apply letterboxing to preserve aspect ratio while fitting to model input size
     */
    private fun applyLetterboxing(sourceBitmap: Bitmap, targetBitmap: Bitmap): Bitmap? {
        return try {
            val canvas = Canvas(targetBitmap)
            canvas.drawColor(Color.BLACK)  // Black padding
            
            val sourceWidth = sourceBitmap.width
            val sourceHeight = sourceBitmap.height
            val targetWidth = targetBitmap.width
            val targetHeight = targetBitmap.height
            
            // Calculate scaling factor to preserve aspect ratio
            val scaleX = targetWidth.toFloat() / sourceWidth
            val scaleY = targetHeight.toFloat() / sourceHeight
            val scale = if (keepAspectRatio) min(scaleX, scaleY) else max(scaleX, scaleY)
            
            // Calculate centered position
            val scaledWidth = (sourceWidth * scale).toInt()
            val scaledHeight = (sourceHeight * scale).toInt()
            val left = (targetWidth - scaledWidth) / 2
            val top = (targetHeight - scaledHeight) / 2
            
            // Create scaling matrix
            val matrix = Matrix()
            matrix.postScale(scale, scale)
            matrix.postTranslate(left.toFloat(), top.toFloat())
            
            // Draw the scaled and centered image
            canvas.drawBitmap(sourceBitmap, matrix, null)
            
            targetBitmap
            
        } catch (e: Exception) {
            Log.e(TAG, "Error applying letterboxing", e)
            null
        }
    }
    
    /**
     * Convert bitmap to grayscale for model input
     */
    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val resultBitmap = if (reusableBitmap.get() == null) {
            Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        } else {
            reusableBitmap.get()!!
        }
        
        val canvas = Canvas(resultBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)  // Convert to grayscale
        val colorFilter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = colorFilter
        
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return resultBitmap
    }
    
    /**
     * Get reusable bitmap to reduce memory allocations
     */
    private fun getReusableBitmap(width: Int, height: Int): Bitmap {
        var bitmap = reusableBitmap.get()
        if (bitmap == null || bitmap.width != width || bitmap.height != height) {
            bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            reusableBitmap.set(bitmap)
        }
        return bitmap
    }
    
    /**
     * Post-process YOLO output for human detection
     * Handles confidence thresholding, class filtering, and NMS
     */
    fun postProcessResults(
        outputBuffer: ByteBuffer,
        originalWidth: Int,
        originalHeight: Int
    ): List<HumanDetection> {
        return try {
            val startTime = System.nanoTime()
            
            // Reset buffer position
            outputBuffer.rewind()
            
            // Parse YOLO output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
            val detections = parseYOLOOutput(outputBuffer, originalWidth, originalHeight)
            
            // Apply confidence threshold
            val filteredDetections = detections.filter { it.confidence >= confidenceThreshold }
            
            // Apply class filtering (human only)
            val humanDetections = filteredDetections.filter { it.classId == humanClassId }
            
            // Apply Non-Maximum Suppression
            val finalDetections = if (modelConfig.applyNMS) {
                applyNMS(humanDetections, nmsThreshold)
            } else {
                humanDetections
            }
            
            val endTime = System.nanoTime()
            Log.d(TAG, "Postprocessing completed: ${finalDetections.size} detections in ${(endTime - startTime) / 1_000_000}ms")
            
            finalDetections
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during post-processing", e)
            emptyList()
        }
    }
    
    /**
     * Parse YOLO output tensor into detection objects
     */
    private fun parseYOLOOutput(
        outputBuffer: ByteBuffer,
        originalWidth: Int,
        originalHeight: Int
    ): List<HumanDetection> {
        val detections = mutableListOf<HumanDetection>()
        val numDetections = outputBuffer.capacity() / 6  // Assuming 6 values per detection
        
        for (i in 0 until numDetections) {
            val offset = i * 6
            
            try {
                // Read detection data
                val x1 = outputBuffer.getFloat(offset + 0)
                val y1 = outputBuffer.getFloat(offset + 1)
                val x2 = outputBuffer.getFloat(offset + 2)
                val y2 = outputBuffer.getFloat(offset + 3)
                val confidence = outputBuffer.getFloat(offset + 4)
                val classId = outputBuffer.getInt(offset + 5)
                
                // Normalize coordinates if needed (0-255 range)
                val normalizedX1 = if (x1 > 1.0f) x1 / 255.0f else x1
                val normalizedY1 = if (y1 > 1.0f) y1 / 255.0f else y1
                val normalizedX2 = if (x2 > 1.0f) x2 / 255.0f else x2
                val normalizedY2 = if (y2 > 1.0f) y2 / 255.0f else y2
                
                // Create bounding box
                val width = normalizedX2 - normalizedX1
                val height = normalizedY2 - normalizedY1
                
                val boundingBox = BoundingBox(
                    x = normalizedX1,
                    y = normalizedY1,
                    width = width,
                    height = height
                )
                
                detections.add(
                    HumanDetection(
                        boundingBox = boundingBox,
                        confidence = confidence,
                        classId = classId,
                        className = getClassName(classId)
                    )
                )
                
            } catch (e: Exception) {
                Log.w(TAG, "Error parsing detection $i", e)
                continue
            }
        }
        
        return detections
    }
    
    /**
     * Apply Non-Maximum Suppression to remove duplicate detections
     */
    private fun applyNMS(detections: List<HumanDetection>, iouThreshold: Float): List<HumanDetection> {
        if (detections.isEmpty()) return emptyList()
        
        // Sort by confidence score (descending)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val result = mutableListOf<HumanDetection>()
        
        for (currentDetection in sortedDetections) {
            var shouldKeep = true
            
            for (keptDetection in result) {
                val iou = calculateIoU(currentDetection.boundingBox, keptDetection.boundingBox)
                if (iou > iouThreshold) {
                    shouldKeep = false
                    break
                }
            }
            
            if (shouldKeep) {
                result.add(currentDetection)
            }
        }
        
        Log.d(TAG, "NMS: ${detections.size} -> ${result.size} detections")
        return result
    }
    
    /**
     * Calculate Intersection over Union (IoU) between two bounding boxes
     */
    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = max(box1.x, box2.x)
        val y1 = max(box1.y, box2.y)
        val x2 = min(box1.x + box1.width, box2.x + box2.width)
        val y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if (x2 <= x1 || y2 <= y1) return 0.0f
        
        val intersection = (x2 - x1) * (y2 - y1)
        val union = box1.width * box1.height + box2.width * box2.height - intersection
        
        return if (union > 0) intersection / union else 0.0f
    }
    
    /**
     * Get class name from class ID
     */
    private fun getClassName(classId: Int): String {
        return when (classId) {
            0 -> "person"
            else -> "class_$classId"
        }
    }
    
    /**
     * Frame skipping logic based on performance monitoring
     */
    fun shouldSkipFrame(): Boolean {
        val currentTime = System.currentTimeMillis()
        val timeSinceLastProcess = currentTime - lastProcessTime
        
        return if (timeSinceLastProcess < targetFrameInterval) {
            true
        } else {
            lastProcessTime = currentTime
            false
        }
    }
    
    /**
     * Adaptive frame skipping based on detection confidence and scene stability
     */
    fun shouldSkipFrameAdaptive(
        previousDetections: List<HumanDetection>,
        currentDetections: List<HumanDetection>
    ): Boolean {
        if (currentTime - lastProcessTime < targetFrameInterval) {
            return true
        }
        
        // Skip frames if previous detections are stable and confident
        if (previousDetections.isNotEmpty()) {
            val avgConfidence = previousDetections.sumOf { it.confidence.toDouble() } / previousDetections.size
            val stabilityScore = calculateStabilityScore(previousDetections, currentDetections)
            
            // Skip frame if detections are stable and confident
            if (avgConfidence > 0.8 && stabilityScore > 0.9) {
                return true
            }
        }
        
        lastProcessTime = System.currentTimeMillis()
        return false
    }
    
    /**
     * Calculate scene stability score between two sets of detections
     */
    private fun calculateStabilityScore(
        detections1: List<HumanDetection>,
        detections2: List<HumanDetection>
    ): Double {
        if (detections1.isEmpty() || detections2.isEmpty()) return 0.0
        
        var totalIoU = 0.0
        var comparisonCount = min(detections1.size, detections2.size)
        
        for (i in 0 until comparisonCount) {
            val iou = calculateIoU(detections1[i].boundingBox, detections2[i].boundingBox)
            totalIoU += iou
        }
        
        return if (comparisonCount > 0) totalIoU / comparisonCount else 0.0
    }
    
    /**
     * Dynamic image resizing based on device performance
     */
    fun getOptimalInputSize(devicePerformance: DevicePerformance): Int {
        return when (devicePerformance) {
            DevicePerformance.HIGH -> 640
            DevicePerformance.MEDIUM -> 480
            DevicePerformance.LOW -> 320
        }
    }
    
    /**
     * Process image with async support for better performance
     */
    suspend fun processImageAsync(bitmap: Bitmap): Result<Bitmap> = withContext(Dispatchers.Default) {
        try {
            val processedBitmap = preprocessImage(bitmap)
            if (processedBitmap != null) {
                Result.success(processedBitmap)
            } else {
                Result.failure(Exception("Image processing failed"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Post-process with async support
     */
    suspend fun postProcessAsync(
        outputBuffer: ByteBuffer,
        originalWidth: Int,
        originalHeight: Int
    ): Result<List<HumanDetection>> = withContext(Dispatchers.Default) {
        try {
            val detections = postProcessResults(outputBuffer, originalWidth, originalHeight)
            Result.success(detections)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Clear reusable resources to prevent memory leaks
     */
    fun clearReusableResources() {
        reusableByteBuffer.set(null)
        reusableBitmap.set(null)
    }
    
    /**
     * Get processing statistics
     */
    fun getProcessingStats(): ProcessingStats {
        return ProcessingStats(
            targetFPS = modelConfig.targetFPS,
            inputSize = inputSize,
            confidenceThreshold = confidenceThreshold,
            nmsThreshold = nmsThreshold,
            humanClassId = humanClassId,
            applyNMS = modelConfig.applyNMS,
            keepAspectRatio = keepAspectRatio
        )
    }
}

/**
 * Supporting data classes
 */
data class BoundingBox(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float
)

data class HumanDetection(
    val boundingBox: BoundingBox,
    val confidence: Float,
    val classId: Int,
    val className: String
)

data class ProcessingStats(
    val targetFPS: Int,
    val inputSize: Int,
    val confidenceThreshold: Float,
    val nmsThreshold: Float,
    val humanClassId: Int,
    val applyNMS: Boolean,
    val keepAspectRatio: Boolean
)

enum class DevicePerformance {
    HIGH,    // Flagship devices with strong GPU/NNAPI
    MEDIUM,  // Mid-range devices
    LOW      // Budget devices or older hardware
}