package com.yolov11n.detection

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.os.Build
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.max
import kotlin.math.min

/**
 * Bounding Box Renderer - Real-time bounding box drawing
 * Optimized for performance with object pooling, efficient drawing, and
 * configurable styling based on mobile optimization research findings
 */
class BoundingBoxRenderer @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    private val TAG = "BoundingBoxRenderer"
    
    // Drawing configuration
    private var boxPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var textPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var fillPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    
    // Rendering settings
    private var showLabels: Boolean = true
    private var showConfidence: Boolean = true
    private var boxStrokeWidth: Float = 2.0f
    private var textSize: Float = 14.0f
    private var cornerRadius: Float = 4.0f
    private var padding: Float = 8.0f
    
    // Color schemes
    private var humanBoxColor: Int = Color.GREEN
    private var humanTextColor: Int = Color.WHITE
    private var humanFillColor: Int = Color.argb(50, 0, 255, 0) // Semi-transparent green
    private var backgroundBoxColor: Int = Color.argb(128, 0, 0, 0) // Semi-transparent black
    
    // Performance optimization
    private val objectPool = mutableListOf<DetectionDrawable>()
    private val maxPoolSize = 50
    private var reuseObjects = true
    private var cacheTextMeasurements = true
    
    // Text measurement cache
    private val textMeasurementCache = ConcurrentHashMap<String, Rect>()
    
    // Animation support
    private var enableAnimations: Boolean = true
    private val animationDuration = 200L // milliseconds
    private var currentDetections: List<HumanDetection> = emptyList()
    private var targetDetections: List<HumanDetection> = emptyList()
    private var animationStartTime: Long = 0L
    
    // Thread safety
    private val renderScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private val isRendering = AtomicBoolean(false)
    
    init {
        initializePaints()
        setLayerType(LAYER_TYPE_HARDWARE, null) // Enable hardware acceleration
    }
    
    /**
     * Initialize paint objects for optimal performance
     */
    private fun initializePaints() {
        // Box outline paint
        boxPaint.apply {
            style = Paint.Style.STROKE
            strokeWidth = boxStrokeWidth
            color = humanBoxColor
            isAntiAlias = true
        }
        
        // Text paint
        textPaint.apply {
            textSize = this@BoundingBoxRenderer.textSize
            color = humanTextColor
            isAntiAlias = true
            typeface = Typeface.DEFAULT_BOLD
        }
        
        // Fill paint for bounding boxes
        fillPaint.apply {
            style = Paint.Style.FILL
            color = humanFillColor
            isAntiAlias = true
        }
        
        // Background paint for text labels
        backgroundPaint.apply {
            style = Paint.Style.FILL
            color = backgroundBoxColor
            isAntiAlias = true
        }
    }
    
    /**
     * Update rendering configuration
     */
    fun updateConfiguration(config: RenderingConfig) {
        showLabels = config.showLabels
        showConfidence = config.showConfidence
        boxStrokeWidth = config.boxStrokeWidth
        textSize = config.textSize
        cornerRadius = config.cornerRadius
        padding = config.padding
        humanBoxColor = config.humanBoxColor
        humanTextColor = config.humanTextColor
        humanFillColor = config.humanFillColor
        enableAnimations = config.enableAnimations
        reuseObjects = config.reuseObjects
        cacheTextMeasurements = config.cacheTextMeasurements
        
        // Reinitialize paints with new settings
        initializePaints()
        
        // Clear cached measurements if disabled
        if (!cacheTextMeasurements) {
            textMeasurementCache.clear()
        }
        
        invalidate() // Redraw with new configuration
    }
    
    /**
     * Set detections to render with animation support
     */
    fun setDetections(detections: List<HumanDetection>) {
        renderScope.launch {
            if (enableAnimations && currentDetections.isNotEmpty()) {
                // Start animation to new detections
                targetDetections = detections
                animationStartTime = System.currentTimeMillis()
                startAnimation()
            } else {
                // Direct update without animation
                currentDetections = detections
                invalidate()
            }
        }
    }
    
    /**
     * Start smooth animation between detection states
     */
    private suspend fun startAnimation() {
        val animationStart = animationStartTime
        val duration = animationDuration
        
        while (isActive) {
            val elapsed = System.currentTimeMillis() - animationStart
            val progress = min(1.0f, elapsed / duration.toFloat())
            
            // Interpolate between current and target detections
            currentDetections = interpolateDetections(currentDetections, targetDetections, progress)
            
            invalidate()
            
            if (progress >= 1.0f) {
                currentDetections = targetDetections
                break
            }
            
            delay(16) // ~60 FPS
        }
    }
    
    /**
     * Interpolate between two detection lists for smooth animation
     */
    private fun interpolateDetections(
        from: List<HumanDetection>,
        to: List<HumanDetection>,
        progress: Float
    ): List<HumanDetection> {
        return to.mapIndexed { index, targetDetection ->
            val sourceDetection = from.getOrNull(index)
            if (sourceDetection != null) {
                sourceDetection.copy(
                    boundingBox = BoundingBox(
                        x = lerp(sourceDetection.boundingBox.x, targetDetection.boundingBox.x, progress),
                        y = lerp(sourceDetection.boundingBox.y, targetDetection.boundingBox.y, progress),
                        width = lerp(sourceDetection.boundingBox.width, targetDetection.boundingBox.width, progress),
                        height = lerp(sourceDetection.boundingBox.height, targetDetection.boundingBox.height, progress)
                    ),
                    confidence = lerp(sourceDetection.confidence, targetDetection.confidence, progress)
                )
            } else {
                targetDetection
            }
        }
    }
    
    /**
     * Linear interpolation helper
     */
    private fun lerp(start: Float, end: Float, progress: Float): Float {
        return start + (end - start) * progress
    }
    
    /**
     * Main drawing function - optimized for performance
     */
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        if (currentDetections.isEmpty()) return
        
        val startTime = System.nanoTime()
        
        // Clear and reuse object pool
        if (reuseObjects && objectPool.isEmpty()) {
            fillObjectPool()
        }
        
        // Draw each detection
        for (detection in currentDetections) {
            val drawable = if (reuseObjects && objectPool.isNotEmpty()) {
                objectPool.removeAt(objectPool.lastIndex)
            } else {
                DetectionDrawable()
            }
            
            drawDetection(canvas, detection, drawable)
            
            // Return to pool for reuse
            if (reuseObjects) {
                objectPool.add(drawable)
            }
        }
        
        // Trim pool size
        if (reuseObjects && objectPool.size > maxPoolSize) {
            objectPool.subList(maxPoolSize, objectPool.size).clear()
        }
        
        val endTime = System.nanoTime()
        val drawTime = (endTime - startTime) / 1_000_000.0
        
        // Log performance if needed
        if (Build.DEBUG) {
            if (drawTime > 16.0) { // > 16ms indicates potential performance issues
                Log.w(TAG, "Bounding box rendering took ${drawTime}ms for ${currentDetections.size} detections")
            }
        }
    }
    
    /**
     * Draw single detection with optimal performance
     */
    private fun drawDetection(canvas: Canvas, detection: HumanDetection, drawable: DetectionDrawable) {
        val bounds = calculateScaledBounds(detection)
        
        // Create rounded rectangle path
        drawable.path.reset()
        drawable.path.addRoundRect(
            bounds.left, bounds.top, bounds.right, bounds.bottom,
            cornerRadius, cornerRadius, Path.Direction.CW
        )
        
        // Draw filled background with transparency
        canvas.drawPath(drawable.path, fillPaint)
        
        // Draw box outline
        canvas.drawPath(drawable.path, boxPaint)
        
        // Draw label if enabled
        if (showLabels) {
            drawLabel(canvas, detection, bounds, drawable)
        }
    }
    
    /**
     * Calculate scaled bounds for detection
     */
    private fun calculateScaledBounds(detection: HumanDetection): RectF {
        val width = width.toFloat()
        val height = height.toFloat()
        
        return RectF(
            detection.boundingBox.x * width,
            detection.boundingBox.y * height,
            (detection.boundingBox.x + detection.boundingBox.width) * width,
            (detection.boundingBox.y + detection.boundingBox.height) * height
        )
    }
    
    /**
     * Draw label with background for better readability
     */
    private fun drawLabel(canvas: Canvas, detection: HumanDetection, bounds: RectF, drawable: DetectionDrawable) {
        val label = buildLabelText(detection)
        val textBounds = getTextBounds(label)
        
        // Position label above the bounding box
        val textLeft = bounds.left
        val textTop = bounds.top - textBounds.height() - padding
        
        // Ensure label stays within view bounds
        val adjustedLeft = max(padding, textLeft)
        val adjustedTop = max(textBounds.height() + padding, textTop)
        
        val labelRect = RectF(
            adjustedLeft - padding,
            adjustedTop - textBounds.height() - padding,
            adjustedLeft + textBounds.width() + padding * 2,
            adjustedTop + padding
        )
        
        // Draw label background
        drawable.path.reset()
        drawable.path.addRoundRect(labelRect, cornerRadius, cornerRadius, Path.Direction.CW)
        canvas.drawPath(drawable.path, backgroundPaint)
        
        // Draw label text
        canvas.drawText(label, adjustedLeft, adjustedTop, textPaint)
    }
    
    /**
     * Build label text with confidence
     */
    private fun buildLabelText(detection: HumanDetection): String {
        return if (showConfidence) {
            "${detection.className} ${(detection.confidence * 100).toInt()}%"
        } else {
            detection.className
        }
    }
    
    /**
     * Get or create cached text bounds
     */
    private fun getTextBounds(text: String): Rect {
        return if (cacheTextMeasurements) {
            textMeasurementCache.getOrPut(text) {
                val rect = Rect()
                textPaint.getTextBounds(text, 0, text.length, rect)
                rect
            }
        } else {
            val rect = Rect()
            textPaint.getTextBounds(text, 0, text.length, rect)
            rect
        }
    }
    
    /**
     * Fill object pool for performance optimization
     */
    private fun fillObjectPool() {
        objectPool.clear()
        repeat(maxPoolSize) {
            objectPool.add(DetectionDrawable())
        }
    }
    
    /**
     * Handle configuration changes
     */
    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        
        // Clear text cache on size change as text measurements may no longer be valid
        if (cacheTextMeasurements) {
            textMeasurementCache.clear()
        }
    }
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        renderScope.cancel()
        objectPool.clear()
        textMeasurementCache.clear()
        currentDetections = emptyList()
    }
    
    /**
     * Get current rendering statistics
     */
    fun getRenderingStats(): RenderingStats {
        return RenderingStats(
            currentDetections = currentDetections.size,
            objectPoolSize = objectPool.size,
            cachedTextMeasurements = textMeasurementCache.size,
            isAnimating = enableAnimations && targetDetections.isNotEmpty(),
            showLabels = showLabels,
            showConfidence = showConfidence
        )
    }
}

/**
 * Reusable drawable object for performance optimization
 */
private class DetectionDrawable {
    val path = Path()
    val rect = RectF()
}

/**
 * Rendering configuration data class
 */
data class RenderingConfig(
    val showLabels: Boolean = true,
    val showConfidence: Boolean = true,
    val boxStrokeWidth: Float = 2.0f,
    val textSize: Float = 14.0f,
    val cornerRadius: Float = 4.0f,
    val padding: Float = 8.0f,
    val humanBoxColor: Int = Color.GREEN,
    val humanTextColor: Int = Color.WHITE,
    val humanFillColor: Int = Color.argb(50, 0, 255, 0),
    val enableAnimations: Boolean = true,
    val reuseObjects: Boolean = true,
    val cacheTextMeasurements: Boolean = true
)

/**
 * Rendering statistics
 */
data class RenderingStats(
    val currentDetections: Int,
    val objectPoolSize: Int,
    val cachedTextMeasurements: Int,
    val isAnimating: Boolean,
    val showLabels: Boolean,
    val showConfidence: Boolean
)