package com.yolodetection.app.overlay

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.yolodetection.app.detection.models.Detection

/**
 * Custom overlay view for drawing detection bounding boxes and labels
 * 
 * Optimized for real-time rendering with minimal performance impact
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    // Detection data
    private var detections: List<Detection> = emptyList()
    
    // Drawing resources
    private val personPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
        alpha = 220
    }
    
    private val labelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
        alpha = 255
    }
    
    private val backgroundPaint = Paint().apply {
        color = Color.argb(200, 0, 0, 0) // Semi-transparent black
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val shadowPaint = Paint().apply {
        color = Color.argb(100, 0, 0, 0) // Shadow for better visibility
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Visual configuration
    private val cornerRadius = 8f
    private val labelPadding = 8f
    private val labelTextSize = 32f
    private val minLabelWidth = 120f
    
    // Animation state
    private var animationProgress = 1.0f
    
    /**
     * Set detections to be displayed
     */
    fun setDetections(detections: List<Detection>) {
        this.detections = detections
        invalidate() // Request redraw
    }
    
    /**
     * Clear all detections
     */
    fun clearDetections() {
        detections = emptyList()
        invalidate()
    }
    
    /**
     * Enable animation for detection appearance
     */
    fun setAnimationProgress(progress: Float) {
        animationProgress = progress.coerceIn(0f, 1f)
        invalidate()
    }
    
    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)
        canvas ?: return
        
        // Draw each detection
        for (detection in detections) {
            drawDetection(canvas, detection)
        }
    }
    
    /**
     * Draw a single detection with bounding box and label
     */
    private fun drawDetection(canvas: Canvas, detection: Detection) {
        val bounds = detection.boundingBox.toRectF()
        
        // Scale bounds based on animation progress
        val scaledBounds = if (animationProgress < 1.0f) {
            val centerX = bounds.centerX()
            val centerY = bounds.centerY()
            val scaledWidth = bounds.width() * animationProgress
            val scaledHeight = bounds.height() * animationProgress
            
            RectF(
                centerX - scaledWidth / 2f,
                centerY - scaledHeight / 2f,
                centerX + scaledWidth / 2f,
                centerY + scaledHeight / 2f
            )
        } else {
            bounds
        }
        
        // Draw shadow for better visibility
        drawShadow(canvas, scaledBounds)
        
        // Draw bounding box
        drawBoundingBox(canvas, scaledBounds, detection)
        
        // Draw label
        drawLabel(canvas, scaledBounds, detection)
        
        // Draw center point for debugging
        if (isInDebugMode()) {
            drawCenterPoint(canvas, scaledBounds, detection)
        }
    }
    
    /**
     * Draw shadow for the bounding box
     */
    private fun drawShadow(canvas: Canvas, bounds: RectF) {
        canvas.save()
        val shadowRect = RectF(
            bounds.left + 2f,
            bounds.top + 2f,
            bounds.right + 2f,
            bounds.bottom + 2f
        )
        canvas.drawRoundRect(shadowRect, cornerRadius, cornerRadius, shadowPaint)
        canvas.restore()
    }
    
    /**
     * Draw the bounding box
     */
    private fun drawBoundingBox(canvas: Canvas, bounds: RectF, detection: Detection) {
        canvas.save()
        
        // Choose color based on detection class
        val paint = getPaintForClass(detection.className)
        
        // Draw rounded rectangle for the bounding box
        canvas.drawRoundRect(bounds, cornerRadius, cornerRadius, paint)
        
        // Add corner indicators for better visibility
        drawCorners(canvas, bounds, paint)
        
        canvas.restore()
    }
    
    /**
     * Draw corner indicators on bounding box
     */
    private fun drawCorners(canvas: Canvas, bounds: RectF, paint: Paint) {
        val cornerLength = 20f
        val cornerWidth = 3f
        
        canvas.save()
        paint.strokeWidth = cornerWidth
        
        // Top-left corner
        canvas.drawLine(bounds.left, bounds.top, bounds.left + cornerLength, bounds.top, paint)
        canvas.drawLine(bounds.left, bounds.top, bounds.left, bounds.top + cornerLength, paint)
        
        // Top-right corner
        canvas.drawLine(bounds.right - cornerLength, bounds.top, bounds.right, bounds.top, paint)
        canvas.drawLine(bounds.right, bounds.top, bounds.right, bounds.top + cornerLength, paint)
        
        // Bottom-left corner
        canvas.drawLine(bounds.left, bounds.bottom, bounds.left + cornerLength, bounds.bottom, paint)
        canvas.drawLine(bounds.left, bounds.bottom - cornerLength, bounds.left, bounds.bottom, paint)
        
        // Bottom-right corner
        canvas.drawLine(bounds.right - cornerLength, bounds.bottom, bounds.right, bounds.bottom, paint)
        canvas.drawLine(bounds.right, bounds.bottom - cornerLength, bounds.right, bounds.bottom, paint)
        
        canvas.restore()
    }
    
    /**
     * Draw detection label with background
     */
    private fun drawLabel(canvas: Canvas, bounds: RectF, detection: Detection) {
        val label = "${detection.className} ${String.format("%.1f", detection.confidence * 100)}%"
        val textMetrics = labelPaint.measureText(label)
        
        // Calculate label position
        val labelX = bounds.left
        val labelY = bounds.top - 10f
        
        // Calculate label dimensions
        val labelWidth = maxOf(textMetrics, minLabelWidth) + labelPadding * 2
        val labelHeight = labelTextSize + labelPadding * 2
        
        // Create label background rect
        val labelRect = RectF(
            labelX,
            maxOf(labelY - labelHeight, 0f),
            labelX + labelWidth,
            maxOf(labelY, labelHeight)
        )
        
        // Draw label background
        canvas.save()
        canvas.drawRoundRect(labelRect, cornerRadius / 2, cornerRadius / 2, backgroundPaint)
        canvas.restore()
        
        // Draw label text
        canvas.save()
        labelPaint.textSize = labelTextSize
        labelPaint.color = getTextColorForClass(detection.className)
        
        val textX = labelRect.left + labelPadding
        val textY = labelRect.bottom - labelPadding - (labelPaint.fontMetrics?.bottom ?: 0f)
        
        canvas.drawText(label, textX, textY, labelPaint)
        canvas.restore()
    }
    
    /**
     * Draw center point for debugging
     */
    private fun drawCenterPoint(canvas: Canvas, bounds: RectF, detection: Detection) {
        canvas.save()
        val centerX = bounds.centerX()
        val centerY = bounds.centerY()
        
        val centerPaint = Paint().apply {
            color = Color.YELLOW
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        
        canvas.drawCircle(centerX, centerY, 4f, centerPaint)
        canvas.restore()
    }
    
    /**
     * Get paint for specific class
     */
    private fun getPaintForClass(className: String): Paint {
        return when (className.lowercase()) {
            "person" -> personPaint
            else -> Paint().apply {
                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 4f
                isAntiAlias = true
                alpha = 220
            }
        }
    }
    
    /**
     * Get text color for specific class
     */
    private fun getTextColorForClass(className: String): Int {
        return when (className.lowercase()) {
            "person" -> Color.WHITE
            else -> Color.WHITE
        }
    }
    
    /**
     * Check if debug mode is enabled
     */
    private fun isInDebugMode(): Boolean {
        return false // Set to true for debugging
    }
    
    /**
     * Update visual configuration
     */
    fun updateVisualConfig(
        showPersonOnly: Boolean = true,
        showConfidence: Boolean = true,
        showCornerIndicators: Boolean = true
    ) {
        // This could be used to customize the overlay appearance
        invalidate()
    }
    
    /**
     * Get the number of current detections
     */
    fun getDetectionCount(): Int = detections.size
    
    /**
     * Get person detections only
     */
    fun getPersonDetections(): List<Detection> = detections.filter { it.className == "person" }
}