package com.yolodetection.ui

import android.content.Context
import android.graphics.*
import android.os.Debug
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import kotlin.math.max
import kotlin.math.min

/**
 * Custom view for displaying real-time performance statistics
 * Shows FPS, memory usage, and detection performance metrics
 */
class PerformanceStatsView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    companion object {
        private const val TAG = "PerformanceStatsView"
        
        // Layout constants
        private const val CORNER_RADIUS = 8f
        private const val PADDING = 8f
        private const val TEXT_SIZE = 12f
        private const val TITLE_SIZE = 14f
        private const val STAT_SPACING = 4f
        private const val CHART_HEIGHT = 40f
        
        // Update intervals
        private const val FPS_UPDATE_INTERVAL = 500L // ms
        private const val MEMORY_UPDATE_INTERVAL = 1000L // ms
        private const val MAX_HISTORY_SIZE = 60 // Number of points to keep in history
        
        // Memory thresholds (in MB)
        private const val MEMORY_WARNING_THRESHOLD = 500
        private const val MEMORY_CRITICAL_THRESHOLD = 1000
        
        // FPS thresholds
        private const val FPS_GOOD_THRESHOLD = 25f
        private const val FPS_EXCELLENT_THRESHOLD = 40f
    }
    
    // Performance data
    private var currentFps = 0f
    private var targetFps = 30f
    private var detectionCount = 0
    private var humanCount = 0
    private var processingTime = 0f // ms
    private var inferenceTime = 0f // ms
    
    // Memory data
    private var usedMemoryMB = 0L
    private var totalMemoryMB = 0L
    private var availableMemoryMB = 0L
    
    // History data for charts
    private val fpsHistory = mutableListOf<Float>()
    private val memoryHistory = mutableListOf<Long>()
    private val processingTimeHistory = mutableListOf<Float>()
    
    // Update timing
    private var lastFpsUpdate = 0L
    private var lastMemoryUpdate = 0L
    private var frameCount = 0
    
    // Paint objects for drawing
    private val backgroundPaint = Paint().apply {
        color = ContextCompat.getColor(context, android.R.color.background_dark)
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val borderPaint = Paint().apply {
        color = ContextCompat.getColor(context, android.R.color.darker_gray)
        style = Paint.Style.STROKE
        strokeWidth = 1f
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = ContextCompat.getColor(context, android.R.color.primary_text_dark)
        textSize = TEXT_SIZE * resources.displayMetrics.scaledDensity
        typeface = Typeface.MONOSPACE
        isAntiAlias = true
    }
    
    private val titlePaint = Paint().apply {
        color = ContextCompat.getColor(context, android.R.color.primary_text_dark)
        textSize = TITLE_SIZE * resources.displayMetrics.scaledDensity
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }
    
    private val goodColor = Color.parseColor("#4CAF50") // Green
    private val warningColor = Color.parseColor("#FF9800") // Orange
    private val errorColor = Color.parseColor("#F44336") // Red
    private val infoColor = Color.parseColor("#2196F3") // Blue
    
    // Chart paints
    private val fpsChartPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        isAntiAlias = true
    }
    
    private val memoryChartPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        isAntiAlias = true
    }
    
    private val processingTimeChartPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        isAntiAlias = true
    }
    
    // Animation state
    private val animator = android.animation.ValueAnimator.ofFloat(0f, 1f)
    private var animationProgress = 0f
    
    init {
        animator.duration = 300
        animator.addUpdateListener {
            animationProgress = it.animatedValue as Float
            invalidate()
        }
        start()
    }
    
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val desiredWidth = 200 * resources.displayMetrics.densityDpi / 160
        val desiredHeight = 120 * resources.displayMetrics.densityDpi / 160
        
        val widthMode = MeasureSpec.getMode(widthMeasureSpec)
        val widthSize = MeasureSpec.getSize(widthMeasureSpec)
        val heightMode = MeasureSpec.getMode(heightMeasureSpec)
        val heightSize = MeasureSpec.getSize(heightMeasureSpec)
        
        val width = when (widthMode) {
            MeasureSpec.EXACTLY -> widthSize
            MeasureSpec.AT_MOST -> min(desiredWidth.toInt(), widthSize)
            else -> desiredWidth.toInt()
        }
        
        val height = when (heightMode) {
            MeasureSpec.EXACTLY -> heightSize
            MeasureSpec.AT_MOST -> min(desiredHeight.toInt(), heightSize)
            else -> desiredHeight.toInt()
        }
        
        setMeasuredDimension(width, height)
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val width = width.toFloat()
        val height = height.toFloat()
        
        // Draw background with rounded corners
        drawRoundedBackground(canvas, width, height)
        
        // Apply animation progress
        val alpha = (animationProgress * 255).toInt()
        val translatedY = height * (1 - animationProgress)
        canvas.save()
        canvas.translate(0f, translatedY)
        canvas.alpha = alpha
        
        // Draw performance statistics
        drawStats(canvas, width, height)
        
        // Draw performance charts
        drawCharts(canvas, width, height)
        
        canvas.restore()
    }
    
    /**
     * Draw rounded background
     */
    private fun drawRoundedBackground(canvas: Canvas, width: Float, height: Float) {
        val path = Path().apply {
            addRoundRect(
                0f, 0f, width, height,
                CORNER_RADIUS, CORNER_RADIUS,
                Path.Direction.CW
            )
        }
        canvas.drawPath(path, backgroundPaint)
        canvas.drawPath(path, borderPaint)
    }
    
    /**
     * Draw performance statistics
     */
    private fun drawStats(canvas: Canvas, width: Float, height: Float) {
        var y = 20f
        val padding = PADDING
        
        // Title
        canvas.drawText("Performance", padding, y, titlePaint)
        y += 18f
        
        // FPS
        val fpsColor = getColorForFps(currentFps)
        textPaint.color = fpsColor
        canvas.drawText("FPS: ${"%.1f".format(currentFps)} (${"%.0f".format(targetFps)})", padding, y, textPaint)
        y += 14f
        
        // Detection count
        textPaint.color = infoColor
        canvas.drawText("Detections: $detectionCount ($humanCount human)", padding, y, textPaint)
        y += 14f
        
        // Processing time
        val processingColor = if (processingTime < 50) goodColor else if (processingTime < 100) warningColor else errorColor
        textPaint.color = processingColor
        canvas.drawText("Processing: ${"%.1f".format(processingTime)}ms", padding, y, textPaint)
        y += 14f
        
        // Inference time
        val inferenceColor = if (inferenceTime < 30) goodColor else if (inferenceTime < 60) warningColor else errorColor
        textPaint.color = inferenceColor
        canvas.drawText("Inference: ${"%.1f".format(inferenceTime)}ms", padding, y, textPaint)
        y += 14f
        
        // Memory usage
        val memoryColor = getColorForMemory(usedMemoryMB)
        textPaint.color = memoryColor
        canvas.drawText("Memory: ${usedMemoryMB}MB / ${totalMemoryMB}MB", padding, y, textPaint)
        y += 14f
        
        // Available memory
        textPaint.color = ContextCompat.getColor(context, android.R.color.secondary_text_dark)
        canvas.drawText("Available: ${availableMemoryMB}MB", padding, y, textPaint)
    }
    
    /**
     * Draw performance charts
     */
    private fun drawCharts(canvas: Canvas, width: Float, height: Float) {
        val chartY = height - CHART_HEIGHT - PADDING
        val chartLeft = PADDING
        val chartRight = width - PADDING
        val chartWidth = chartRight - chartLeft
        val chartHeight = CHART_HEIGHT
        
        // Draw FPS chart
        if (fpsHistory.isNotEmpty()) {
            fpsChartPaint.color = getColorForFps(currentFps)
            drawLineChart(canvas, fpsHistory, chartLeft, chartY, chartWidth, chartHeight, fpsChartPaint) { fps ->
                (fps / targetFps).coerceIn(0f, 1.2f)
            }
        }
        
        // Draw memory chart
        if (memoryHistory.isNotEmpty()) {
            memoryChartPaint.color = getColorForMemory(usedMemoryMB)
            drawLineChart(canvas, memoryHistory.map { it.toFloat() }, chartLeft, chartY, chartWidth, chartHeight, memoryChartPaint) { mem ->
                (mem / (totalMemoryMB * 0.8f)).coerceIn(0f, 1f)
            }
        }
        
        // Draw processing time chart
        if (processingTimeHistory.isNotEmpty()) {
            processingTimeChartPaint.color = if (processingTime < 50) goodColor else if (processingTime < 100) warningColor else errorColor
            drawLineChart(canvas, processingTimeHistory, chartLeft, chartY, chartWidth, chartHeight, processingTimeChartPaint) { time ->
                (time / 100f).coerceIn(0f, 1f)
            }
        }
    }
    
    /**
     * Draw a line chart from data points
     */
    private fun <T> drawLineChart(
        canvas: Canvas,
        data: List<T>,
        left: Float,
        top: Float,
        width: Float,
        height: Float,
        paint: Paint,
        normalizer: (T) -> Float
    ) {
        if (data.size < 2) return
        
        val path = Path()
        val stepX = width / (data.size - 1).toFloat()
        
        data.forEachIndexed { index, value ->
            val normalized = normalizer(value)
            val x = left + index * stepX
            val y = top + height - (normalized * height)
            
            if (index == 0) {
                path.moveTo(x, y)
            } else {
                path.lineTo(x, y)
            }
        }
        
        canvas.drawPath(path, paint)
    }
    
    /**
     * Get color for FPS value
     */
    private fun getColorForFps(fps: Float): Int {
        return when {
            fps >= FPS_EXCELLENT_THRESHOLD -> goodColor
            fps >= FPS_GOOD_THRESHOLD -> infoColor
            fps >= 15f -> warningColor
            else -> errorColor
        }
    }
    
    /**
     * Get color for memory usage
     */
    private fun getColorForMemory(memoryMB: Long): Int {
        return when {
            memoryMB < MEMORY_WARNING_THRESHOLD -> goodColor
            memoryMB < MEMORY_CRITICAL_THRESHOLD -> warningColor
            else -> errorColor
        }
    }
    
    /**
     * Update performance statistics
     */
    fun updateFps(fps: Float) {
        currentFps = fps
        frameCount++
        
        val now = System.currentTimeMillis()
        if (now - lastFpsUpdate >= FPS_UPDATE_INTERVAL) {
            addFpsToHistory(fps)
            lastFpsUpdate = now
            frameCount = 0
        }
        
        animate()
    }
    
    /**
     * Update detection statistics
     */
    fun updateDetections(totalCount: Int, humanCount: Int, processingTimeMs: Float, inferenceTimeMs: Float) {
        detectionCount = totalCount
        this.humanCount = humanCount
        processingTime = processingTimeMs
        inferenceTime = inferenceTimeMs
        
        addProcessingTimeToHistory(processingTimeMs)
        animate()
    }
    
    /**
     * Update memory statistics
     */
    fun updateMemory() {
        val runtime = Runtime.getRuntime()
        usedMemoryMB = runtime.totalMemory() - runtime.freeMemory() / (1024 * 1024)
        totalMemoryMB = runtime.maxMemory() / (1024 * 1024)
        availableMemoryMB = runtime.freeMemory() / (1024 * 1024)
        
        addMemoryToHistory(usedMemoryMB)
        
        val now = System.currentTimeMillis()
        if (now - lastMemoryUpdate >= MEMORY_UPDATE_INTERVAL) {
            lastMemoryUpdate = now
        }
        
        animate()
    }
    
    /**
     * Add FPS value to history
     */
    private fun addFpsToHistory(fps: Float) {
        fpsHistory.add(fps)
        if (fpsHistory.size > MAX_HISTORY_SIZE) {
            fpsHistory.removeAt(0)
        }
    }
    
    /**
     * Add memory value to history
     */
    private fun addMemoryToHistory(memory: Long) {
        memoryHistory.add(memory)
        if (memoryHistory.size > MAX_HISTORY_SIZE) {
            memoryHistory.removeAt(0)
        }
    }
    
    /**
     * Add processing time to history
     */
    private fun addProcessingTimeToHistory(time: Float) {
        processingTimeHistory.add(time)
        if (processingTimeHistory.size > MAX_HISTORY_SIZE) {
            processingTimeHistory.removeAt(0)
        }
    }
    
    /**
     * Set target FPS for comparison
     */
    fun setTargetFps(target: Float) {
        targetFps = target
        animate()
    }
    
    /**
     * Get current performance summary
     */
    fun getPerformanceSummary(): PerformanceSummary {
        return PerformanceSummary(
            currentFps = currentFps,
            targetFps = targetFps,
            detectionCount = detectionCount,
            humanCount = humanCount,
            processingTime = processingTime,
            inferenceTime = inferenceTime,
            memoryUsage = usedMemoryMB,
            memoryUsagePercent = (usedMemoryMB * 100f / max(totalMemoryMB, 1)).toInt(),
            performanceLevel = getPerformanceLevel()
        )
    }
    
    /**
     * Get performance level classification
     */
    private fun getPerformanceLevel(): String {
        return when {
            currentFps >= FPS_EXCELLENT_THRESHOLD && processingTime < 30 -> "Excellent"
            currentFps >= FPS_GOOD_THRESHOLD && processingTime < 50 -> "Good"
            currentFps >= 20f && processingTime < 100 -> "Fair"
            else -> "Poor"
        }
    }
    
    /**
     * Get performance score (0-100)
     */
    fun getPerformanceScore(): Int {
        val fpsScore = min((currentFps / targetFps * 100).toInt(), 100)
        val memoryScore = when {
            usedMemoryMB < MEMORY_WARNING_THRESHOLD -> 100
            usedMemoryMB < MEMORY_CRITICAL_THRESHOLD -> 70
            else -> 40
        }
        val processingScore = when {
            processingTime < 30 -> 100
            processingTime < 60 -> 80
            processingTime < 100 -> 60
            else -> 40
        }
        
        return (fpsScore * 0.4 + memoryScore * 0.3 + processingScore * 0.3).toInt()
    }
    
    /**
     * Animate view updates
     */
    private fun animate() {
        animator.start()
    }
    
    /**
     * Clear all history data
     */
    fun clearHistory() {
        fpsHistory.clear()
        memoryHistory.clear()
        processingTimeHistory.clear()
        invalidate()
    }
    
    /**
     * Start automatic memory monitoring
     */
    fun start() {
        post(memoryUpdateRunnable)
    }
    
    /**
     * Stop automatic memory monitoring
     */
    fun stop() {
        removeCallbacks(memoryUpdateRunnable)
    }
    
    /**
     * Memory update runnable
     */
    private val memoryUpdateRunnable = object : Runnable {
        override fun run() {
            updateMemory()
            postDelayed(this, MEMORY_UPDATE_INTERVAL)
        }
    }
    
    /**
     * Performance summary data class
     */
    data class PerformanceSummary(
        val currentFps: Float,
        val targetFps: Float,
        val detectionCount: Int,
        val humanCount: Int,
        val processingTime: Float,
        val inferenceTime: Float,
        val memoryUsage: Long,
        val memoryUsagePercent: Int,
        val performanceLevel: String
    )
    
    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        stop()
    }
}