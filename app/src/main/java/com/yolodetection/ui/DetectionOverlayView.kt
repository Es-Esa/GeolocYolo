package com.yolodetection.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import kotlin.math.max
import kotlin.math.min

/**
 * Custom view for rendering detection bounding boxes and annotations
 * Optimized for real-time human detection display
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    companion object {
        private const val TAG = "DetectionOverlayView"
        
        // Drawing constants
        private const val CORNER_RADIUS = 8f
        private const val STROKE_WIDTH = 3f
        private const val PADDING = 8f
        private const val TEXT_SIZE = 14f
        private const val CONFIDENCE_BAR_HEIGHT = 4f
        private const val ANIMATION_DURATION = 200L
        
        // Color constants (using Material Design colors)
        private val DETECTION_COLOR = Color.parseColor("#4CAF50") // Green
        private val CONFIDENCE_BAR_COLOR = Color.parseColor("#2196F3") // Blue
        private val TEXT_COLOR = Color.WHITE
        private val BACKGROUND_COLOR = Color.argb(180, 0, 0, 0) // Semi-transparent black
    }
    
    // Detection data
    private var detections: List<Detection> = emptyList()
    
    // Paint objects for optimized drawing
    private val detectionPaint = Paint().apply {
        color = DETECTION_COLOR
        style = Paint.Style.STROKE
        strokeWidth = STROKE_WIDTH
        isAntiAlias = true
    }
    
    private val backgroundPaint = Paint().apply {
        color = BACKGROUND_COLOR
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = TEXT_COLOR
        textSize = TEXT_SIZE * resources.displayMetrics.scaledDensity
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }
    
    private val confidenceBarPaint = Paint().apply {
        color = CONFIDENCE_BAR_COLOR
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Animation state
    private var animationStartTime = 0L
    private var animatedDetections: List<Detection> = emptyList()
    private var animator: ValueAnimator? = null
    
    // Viewport scaling
    private var viewWidth = 0
    private var viewHeight = 0
    
    // Detection data class
    data class Detection(
        val boundingBox: RectF,
        val confidence: Float,
        val className: String,
        val classIndex: Int,
        val isHuman: Boolean = true,
        val trackingId: Int? = null
    ) {
        fun getConfidenceText(): String = "%.1f%%".format(confidence * 100)
        fun getDisplayName(): String = if (isHuman) "Human" else className
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        viewWidth = w
        viewHeight = h
        invalidate()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        if (animatedDetections.isEmpty()) {
            return
        }
        
        // Apply viewport scaling
        canvas.save()
        
        // Get the current animation progress (0.0 to 1.0)
        val animationProgress = if (animator != null && animator!!.isRunning) {
            (animator!!.animatedValue as Float)
        } else {
            1.0f
        }
        
        // Draw each detection with animation
        animatedDetections.forEach { detection ->
            drawDetection(canvas, detection, animationProgress)
        }
        
        canvas.restore()
    }
    
    /**
     * Draw a single detection with smooth animation
     */
    private fun drawDetection(canvas: Canvas, detection: Detection, progress: Float) {
        val rect = detection.boundingBox
        
        // Apply animation to make bounding boxes appear smoothly
        val animatedRect = if (progress < 1.0f) {
            // Start with a small rectangle that expands to full size
            val centerX = rect.centerX()
            val centerY = rect.centerY()
            val halfWidth = rect.width() * progress / 2
            val halfHeight = rect.height() * progress / 2
            RectF(centerX - halfWidth, centerY - halfHeight, centerX + halfWidth, centerY + halfHeight)
        } else {
            rect
        }
        
        // Draw rounded rectangle background for better visibility
        drawDetectionBackground(canvas, animatedRect)
        
        // Draw bounding box
        canvas.drawRoundRect(animatedRect, CORNER_RADIUS, CORNER_RADIUS, detectionPaint)
        
        // Draw confidence bar
        drawConfidenceBar(canvas, animatedRect, detection.confidence)
        
        // Draw labels
        drawLabels(canvas, animatedRect, detection)
        
        // Draw center point for debugging (optional)
        if (isDebugMode()) {
            drawDebugInfo(canvas, animatedRect, detection)
        }
    }
    
    /**
     * Draw detection background for better visibility
     */
    private fun drawDetectionBackground(canvas: Canvas, rect: RectF) {
        val backgroundRect = RectF(
            rect.left - PADDING,
            rect.top - PADDING,
            rect.right + PADDING,
            rect.bottom + PADDING
        )
        canvas.drawRoundRect(backgroundRect, CORNER_RADIUS, CORNER_RADIUS, backgroundPaint)
    }
    
    /**
     * Draw confidence level bar
     */
    private fun drawConfidenceBar(canvas: Canvas, rect: RectF, confidence: Float) {
        val barTop = rect.top - 12f
        val barLeft = rect.left
        val barRight = rect.left + (rect.width() * confidence)
        val barBottom = barTop - CONFIDENCE_BAR_HEIGHT
        
        if (barRight > barLeft) {
            canvas.drawRoundRect(
                barLeft, barBottom, barRight, barTop,
                CONFIDENCE_BAR_HEIGHT / 2, CONFIDENCE_BAR_HEIGHT / 2,
                confidenceBarPaint
            )
        }
        
        // Draw bar background
        val barBackground = Paint().apply {
            color = Color.argb(100, 255, 255, 255)
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        canvas.drawRoundRect(
            barLeft, barBottom, rect.right, barTop,
            CONFIDENCE_BAR_HEIGHT / 2, CONFIDENCE_BAR_HEIGHT / 2,
            barBackground
        )
    }
    
    /**
     * Draw detection labels (class name and confidence)
     */
    private fun drawLabels(canvas: Canvas, rect: RectF, detection: Detection) {
        val textY = rect.top - 20f
        val className = detection.getDisplayName()
        val confidenceText = detection.getConfidenceText()
        
        // Create background for text
        val textBackground = "$className $confidenceText"
        val textWidth = textPaint.measureText(textBackground)
        val backgroundPadding = 8f
        val textBackgroundRect = RectF(
            rect.left,
            textY - textPaint.textSize,
            rect.left + textWidth + backgroundPadding * 2,
            textY + backgroundPadding
        )
        
        // Draw text background
        canvas.drawRoundRect(textBackgroundRect, 4f, 4f, backgroundPaint)
        
        // Draw text
        canvas.drawText(
            textBackground,
            rect.left + backgroundPadding,
            textY,
            textPaint
        )
        
        // Draw tracking ID if available
        detection.trackingId?.let { trackingId ->
            val trackingText = "ID: $trackingId"
            canvas.drawText(
                trackingText,
                rect.left + backgroundPadding,
                textY + textPaint.textSize + 2f,
                textPaint
            )
        }
    }
    
    /**
     * Draw debug information
     */
    private fun drawDebugInfo(canvas: Canvas, rect: RectF, detection: Detection) {
        val debugPaint = Paint().apply {
            color = Color.YELLOW
            textSize = 10f
            isAntiAlias = true
        }
        
        val centerX = rect.centerX()
        val centerY = rect.centerY()
        
        // Draw center point
        canvas.drawCircle(centerX, centerY, 3f, debugPaint)
        
        // Draw dimensions
        val dimensionsText = "W:${rect.width().toInt()} H:${rect.height().toInt()}"
        canvas.drawText(dimensionsText, rect.left, rect.bottom + 15f, debugPaint)
    }
    
    /**
     * Update detection data with smooth animation
     */
    fun setDetections(newDetections: List<Detection>) {
        if (newDetections.isEmpty()) {
            animatedDetections = emptyList()
            invalidate()
            return
        }
        
        // Start animation from current state to new state
        animationStartTime = System.currentTimeMillis()
        animatedDetections = if (detections.isEmpty()) {
            newDetections
        } else {
            // This would be more sophisticated in a real implementation,
            // tracking detections between frames for smooth transitions
            newDetections
        }
        
        // Animate the transition
        startAnimation()
        
        // Update stored detections
        detections = newDetections
    }
    
    /**
     * Start smooth animation for detection updates
     */
    private fun startAnimation() {
        animator?.cancel()
        
        animator = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = ANIMATION_DURATION
            interpolator = android.view.animation.DecelerateInterpolator()
            addUpdateListener {
                invalidate()
            }
            start()
        }
    }
    
    /**
     * Normalize detection coordinates to view size
     */
    fun updateDetectionsWithViewSize(normedDetections: List<Detection>) {
        if (viewWidth == 0 || viewHeight == 0) {
            // Store for later when view size is available
            detections = normedDetections
            return
        }
        
        val scaledDetections = normedDetections.map { detection ->
            detection.copy(
                boundingBox = RectF(
                    detection.boundingBox.left * viewWidth,
                    detection.boundingBox.top * viewHeight,
                    detection.boundingBox.right * viewWidth,
                    detection.boundingBox.bottom * viewHeight
                )
            )
        }
        
        setDetections(scaledDetections)
    }
    
    /**
     * Clear all detections
     */
    fun clearDetections() {
        detections = emptyList()
        animatedDetections = emptyList()
        animator?.cancel()
        invalidate()
    }
    
    /**
     * Get current detection count
     */
    fun getDetectionCount(): Int = detections.size
    
    /**
     * Get human detection count
     */
    fun getHumanDetectionCount(): Int = detections.count { it.isHuman }
    
    /**
     * Check if debug mode is enabled
     */
    private fun isDebugMode(): Boolean {
        return false // Can be made configurable
    }
    
    /**
     * Set custom colors for detection visualization
     */
    fun setCustomColors(
        detectionColor: Int = DETECTION_COLOR,
        textColor: Int = TEXT_COLOR,
        backgroundColor: Int = BACKGROUND_COLOR
    ) {
        detectionPaint.color = detectionColor
        textPaint.color = textColor
        backgroundPaint.color = backgroundColor
        invalidate()
    }
    
    /**
     * Update detection confidence threshold
     */
    fun setConfidenceThreshold(threshold: Float) {
        val filteredDetections = detections.filter { it.confidence >= threshold }
        setDetections(filteredDetections)
    }
    
    /**
     * Get current detection statistics
     */
    fun getDetectionStats(): DetectionStats {
        val humanCount = detections.count { it.isHuman }
        val avgConfidence = if (detections.isNotEmpty()) {
            detections.map { it.confidence }.average().toFloat()
        } else {
            0f
        }
        
        return DetectionStats(
            totalDetections = detections.size,
            humanDetections = humanCount,
            averageConfidence = avgConfidence
        )
    }
    
    // Statistics data class
    data class DetectionStats(
        val totalDetections: Int,
        val humanDetections: Int,
        val averageConfidence: Float
    )
    
    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        animator?.cancel()
    }
}