package com.yolodetection.app.utils

import android.graphics.*
import android.os.SystemClock
import android.util.Log
import com.yolodetection.app.detection.YoloDetector
import com.yolodetection.app.detection.models.Detection
import com.yolodetection.app.overlay.OverlayView
import kotlinx.coroutines.*
import timber.log.Timber
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Semaphore
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Frame processor for real-time camera frames
 * 
 * Handles the complete pipeline from camera frame to detection display
 * with backpressure handling and performance optimization
 */
class FrameProcessor(
    private val yoloDetector: YoloDetector,
    private val overlayView: OverlayView,
    private val onDetectionUpdate: (List<Detection>) -> Unit
) {
    
    companion object {
        private const val TAG = "FrameProcessor"
        private const val TARGET_FPS = 30
        private const val PROCESSING_DELAY = (1000.0 / TARGET_FPS).toLong()
    }
    
    // Threading
    private val processingScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val processingSemaphore = Semaphore(1)
    
    // Frame queue and backpressure
    private val frameQueue = ConcurrentLinkedQueue<FrameData>()
    private val maxQueueSize = 3
    private val isProcessing = AtomicBoolean(false)
    
    // Configuration
    private var isEnabled = true
    private var confidenceThreshold = 0.5f
    private var personOnly = true // For human detection focus
    
    // Performance tracking
    private var frameCount = 0
    private var detectionCount = 0
    private var lastFpsUpdate = SystemClock.elapsedRealtime()
    
    /**
     * Process a camera frame for detection
     */
    suspend fun processFrame(bitmap: Bitmap, rotation: Float) {
        if (!isEnabled) return
        
        // Check queue size for backpressure
        if (frameQueue.size >= maxQueueSize) {
            // Remove oldest frame
            frameQueue.poll()?.release()
            Timber.w("Dropping frame due to backpressure")
        }
        
        // Add new frame
        val frameData = FrameData(bitmap, rotation)
        frameQueue.offer(frameData)
        
        // Start processing if not already processing
        if (isProcessing.compareAndSet(false, true)) {
            processingScope.launch {
                try {
                    processFrameQueue()
                } finally {
                    isProcessing.set(false)
                }
            }
        }
    }
    
    /**
     * Process frames from the queue
     */
    private suspend fun processFrameQueue() {
        while (frameQueue.isNotEmpty()) {
            val frameData = frameQueue.poll() ?: break
            
            try {
                // Acquire processing semaphore
                processingSemaphore.acquire()
                
                // Process the frame
                processFrameData(frameData)
                
            } catch (e: Exception) {
                Timber.e(e, "Error processing frame")
            } finally {
                frameData.release()
                processingSemaphore.release()
                
                // Add delay to maintain target FPS
                delay(PROCESSING_DELAY)
            }
        }
    }
    
    /**
     * Process individual frame data
     */
    private suspend fun processFrameData(frameData: FrameData) {
        val startTime = SystemClock.elapsedRealtime()
        
        try {
            // Run YOLO detection
            val detections = yoloDetector.detect(frameData.bitmap, frameData.rotation)
            
            // Filter detections for person class if enabled
            val filteredDetections = if (personOnly) {
                detections.filter { it.className == "person" }
            } else {
                detections
            }
            
            // Apply confidence threshold
            val confidentDetections = filteredDetections.filter {
                it.confidence >= confidenceThreshold
            }
            
            // Update overlay
            withContext(Dispatchers.Main) {
                overlayView.setDetections(confidentDetections)
                onDetectionUpdate(confidentDetections)
            }
            
            // Update statistics
            updateStatistics(startTime, confidentDetections.size)
            
        } catch (e: Exception) {
            Timber.e(e, "Error in frame data processing")
        }
    }
    
    /**
     * Update performance statistics
     */
    private fun updateStatistics(startTime: Long, detectionCount: Int) {
        val currentTime = SystemClock.elapsedRealtime()
        val processingTime = currentTime - startTime
        
        frameCount++
        this.detectionCount += detectionCount
        
        // Update FPS every second
        if (currentTime - lastFpsUpdate >= 1000) {
            val elapsed = (currentTime - lastFpsUpdate) / 1000.0
            val fps = (frameCount / elapsed).toInt()
            val avgDetections = detectionCount / frameCount
            
            Timber.d("Performance: $fps FPS, ${String.format("%.1f", processingTime.toFloat())}ms processing, ${avgDetections} avg detections")
            
            frameCount = 0
            this.detectionCount = 0
            lastFpsUpdate = currentTime
        }
    }
    
    /**
     * Enable or disable processing
     */
    fun setEnabled(enabled: Boolean) {
        isEnabled = enabled
        if (!enabled) {
            frameQueue.clear()
            withContext(Dispatchers.Main) {
                overlayView.clearDetections()
            }
        }
        Timber.i("Frame processing ${if (enabled) "enabled" else "disabled"}")
    }
    
    /**
     * Set confidence threshold
     */
    fun setConfidenceThreshold(threshold: Float) {
        confidenceThreshold = threshold.coerceIn(0.1f, 0.9f)
        Timber.i("Confidence threshold set to $confidenceThreshold")
    }
    
    /**
     * Enable person-only detection
     */
    fun setPersonOnly(enabled: Boolean) {
        personOnly = enabled
        Timber.i("Person-only detection ${if (enabled) "enabled" else "disabled"}")
    }
    
    /**
     * Set maximum processing FPS
     */
    fun setMaxFPS(maxFps: Int) {
        // This would be used to adjust processing delay
        Timber.i("Max FPS set to $maxFps")
    }
    
    /**
     * Get current performance statistics
     */
    fun getPerformanceStats(): PerformanceStats {
        val detectorStats = yoloDetector.getPerformanceStats()
        return PerformanceStats(
            averageProcessingTime = detectorStats.averageInferenceTime,
            framesPerSecond = detectorStats.framesPerSecond,
            totalDetections = detectorStats.totalInferences,
            personDetections = detectionCount
        )
    }
    
    /**
     * Clear all pending frames
     */
    fun clearQueue() {
        while (frameQueue.isNotEmpty()) {
            frameQueue.poll()?.release()
        }
        Timber.i("Frame queue cleared")
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        processingScope.cancel("Frame processor cleanup")
        clearQueue()
        Timber.i("Frame processor cleaned up")
    }
    
    /**
     * Data class for frame information
     */
    private data class FrameData(
        val bitmap: Bitmap,
        val rotation: Float
    ) {
        private val reference = java.lang.ref.WeakReference(bitmap)
        
        fun release() {
            // Bitmap will be garbage collected naturally
            // This method exists for future use if needed
        }
    }
    
    /**
     * Performance statistics
     */
    data class PerformanceStats(
        val averageProcessingTime: Long,
        val framesPerSecond: Int,
        val totalDetections: Int,
        val personDetections: Int
    ) {
        fun getSummary(): String {
            return "$framesPerSecond FPS | $totalDetections total | $personDetections person"
        }
    }
}