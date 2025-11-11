package com.yolov11n.detection

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.Process
import android.util.Log
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Performance Monitor - FPS and Memory Monitoring
 * Implements comprehensive performance tracking based on mobile optimization research
 * Tracks latency, memory usage, FPS, energy, and thermal behavior for YOLOv11n detection
 */
class PerformanceMonitor(
    private val context: Context,
    private val monitoringInterval: Long = 1000L // 1 second
) {
    private val TAG = "PerformanceMonitor"
    
    // FPS tracking
    private val frameCount = AtomicLong(0)
    private val lastFPSUpdateTime = AtomicLong(0)
    private val currentFPS = AtomicInteger(0)
    private val fpsHistory = ConcurrentLinkedQueue<Int>()
    private val maxFPSHistorySize = 60 // Keep 1 minute of FPS data at 1 FPS
    
    // Latency tracking
    private val inferenceLatencies = ConcurrentLinkedQueue<Long>()
    private val preprocessingLatencies = ConcurrentLinkedQueue<Long>()
    private val postprocessingLatencies = ConcurrentLinkedQueue<Long>()
    private val endToEndLatencies = ConcurrentLinkedQueue<Long>()
    private val maxLatencyHistorySize = 100
    
    // Memory monitoring
    private val memorySnapshots = ConcurrentLinkedQueue<MemorySnapshot>()
    private val maxMemoryHistorySize = 60
    private val activityManager by lazy { context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager }
    
    // Performance thresholds
    private val performanceThresholds = PerformanceThresholds()
    private val isPerformanceDegraded = AtomicBoolean(false)
    
    // Threading
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var monitoringJob: Job? = null
    private val isMonitoring = AtomicBoolean(false)
    
    // Energy and thermal monitoring
    private val batteryManager by lazy { context.getSystemService(Context.BATTERY_SERVICE) as? android.os.BatteryManager }
    private val thermalManager by lazy { context.getSystemService(Context.THERMAL_SERVICE) as? android.os.IThermalService }
    private val energySnapshots = ConcurrentLinkedQueue<EnergySnapshot>()
    
    /**
     * Start performance monitoring
     */
    fun startMonitoring() {
        if (isMonitoring.getAndSet(true)) {
            Log.w(TAG, "Performance monitoring is already running")
            return
        }
        
        monitoringJob = scope.launch {
            Log.i(TAG, "Starting performance monitoring")
            while (isMonitoring.get()) {
                try {
                    capturePerformanceSnapshot()
                    delay(monitoringInterval)
                } catch (e: Exception) {
                    Log.e(TAG, "Error during performance monitoring", e)
                }
            }
        }
    }
    
    /**
     * Stop performance monitoring
     */
    fun stopMonitoring() {
        if (isMonitoring.getAndSet(false)) {
            monitoringJob?.cancel()
            Log.i(TAG, "Performance monitoring stopped")
        }
    }
    
    /**
     * Record inference timing
     */
    fun recordInferenceLatency(nanoseconds: Long) {
        inferenceLatencies.offer(nanoseconds)
        if (inferenceLatencies.size > maxLatencyHistorySize) {
            inferenceLatencies.poll()
        }
    }
    
    /**
     * Record preprocessing timing
     */
    fun recordPreprocessingLatency(nanoseconds: Long) {
        preprocessingLatencies.offer(nanoseconds)
        if (preprocessingLatencies.size > maxLatencyHistorySize) {
            preprocessingLatencies.poll()
        }
    }
    
    /**
     * Record postprocessing timing
     */
    fun recordPostprocessingLatency(nanoseconds: Long) {
        postprocessingLatencies.offer(nanoseconds)
        if (postprocessingLatencies.size > maxLatencyHistorySize) {
            postprocessingLatencies.poll()
        }
    }
    
    /**
     * Record end-to-end timing
     */
    fun recordEndToEndLatency(nanoseconds: Long) {
        endToEndLatencies.offer(nanoseconds)
        if (endToEndLatencies.size > maxLatencyHistorySize) {
            endToEndLatencies.poll()
        }
        
        // Update FPS calculation
        val currentTime = System.currentTimeMillis()
        val lastTime = lastFPSUpdateTime.get()
        
        if (currentTime - lastTime >= 1000) {
            val fps = (frameCount.get() * 1000.0 / (currentTime - lastTime)).toInt()
            currentFPS.set(fps)
            frameCount.set(0)
            lastFPSUpdateTime.set(currentTime)
            
            // Add to FPS history
            fpsHistory.offer(fps)
            if (fpsHistory.size > maxFPSHistorySize) {
                fpsHistory.poll()
            }
        }
        
        frameCount.incrementAndGet()
    }
    
    /**
     * Capture comprehensive performance snapshot
     */
    private suspend fun capturePerformanceSnapshot() {
        val timestamp = System.currentTimeMillis()
        
        // Memory snapshot
        val memorySnapshot = captureMemorySnapshot()
        memorySnapshots.offer(memorySnapshot)
        if (memorySnapshots.size > maxMemoryHistorySize) {
            memorySnapshots.poll()
        }
        
        // Energy snapshot
        val energySnapshot = captureEnergySnapshot()
        energySnapshots.offer(energySnapshot)
        
        // Performance analysis
        analyzePerformance()
        
        // Log performance summary periodically
        if (timestamp % 30000 < monitoringInterval) { // Every 30 seconds
            logPerformanceSummary()
        }
    }
    
    /**
     * Capture memory usage snapshot
     */
    private fun captureMemorySnapshot(): MemorySnapshot {
        val runtime = Runtime.getRuntime()
        val debugMemoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(debugMemoryInfo)
        
        // Get app memory info
        val memInfo = activityManager.processMemoryInfo(intArrayOf(Process.myPid()))
        val appMemoryInfo = memInfo.get(0)
        
        return MemorySnapshot(
            timestamp = System.currentTimeMillis(),
            totalMemoryMB = runtime.totalMemory() / (1024 * 1024),
            usedMemoryMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024),
            maxMemoryMB = runtime.maxMemory() / (1024 * 1024),
            nativeHeapSizeMB = debugMemoryInfo.nativePss / 1024,
            dalvikHeapSizeMB = debugMemoryInfo.dalvikPss / 1024,
            totalPssMB = appMemoryInfo.totalPss / 1024,
            totalPrivateCleanMB = appMemoryInfo.totalPrivateClean / 1024,
            totalPrivateDirtyMB = appMemoryInfo.totalPrivateDirty / 1024
        )
    }
    
    /**
     * Capture energy and thermal snapshot
     */
    private fun captureEnergySnapshot(): EnergySnapshot {
        val batteryManager = this.batteryManager
        
        return EnergySnapshot(
            timestamp = System.currentTimeMillis(),
            batteryLevel = batteryManager?.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1,
            batteryTemperature = batteryManager?.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_TEMPERATURE) ?: -1,
            batteryCurrentNow = batteryManager?.getLongProperty(android.os.BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) ?: -1,
            batteryCurrentAverage = batteryManager?.getLongProperty(android.os.BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE) ?: -1,
            chargingStatus = batteryManager?.isCharging ?: false
        )
    }
    
    /**
     * Analyze current performance and detect degradation
     */
    private fun analyzePerformance() {
        val currentMemoryUsage = getCurrentMemoryUsage()
        val currentFPS = this.currentFPS.get()
        val avgInferenceLatency = getAverageInferenceLatencyMs()
        
        val isDegraded = when {
            currentMemoryUsage > performanceThresholds.maxMemoryUsagePercent -> true
            currentFPS < performanceThresholds.minimumFPS -> true
            avgInferenceLatency > performanceThresholds.maxInferenceLatencyMs -> true
            else -> false
        }
        
        val previousDegraded = isPerformanceDegraded.getAndSet(isDegraded)
        
        if (isDegraded && !previousDegraded) {
            Log.w(TAG, "Performance degradation detected")
            Log.w(TAG, "Memory: ${currentMemoryUsage}%, FPS: $currentFPS, Inference: ${avgInferenceLatency}ms")
        } else if (!isDegraded && previousDegraded) {
            Log.i(TAG, "Performance recovered to normal levels")
        }
    }
    
    /**
     * Get current performance metrics
     */
    fun getCurrentMetrics(): PerformanceMetrics {
        return PerformanceMetrics(
            currentFPS = currentFPS.get(),
            averageFPS = getAverageFPS(),
            minFPS = getMinFPS(),
            maxFPS = getMaxFPS(),
            averageInferenceLatencyMs = getAverageInferenceLatencyMs(),
            percentile95InferenceLatencyMs = getPercentileInferenceLatency(95),
            averagePreprocessingLatencyMs = getAveragePreprocessingLatencyMs(),
            averagePostprocessingLatencyMs = getAveragePostprocessingLatencyMs(),
            averageEndToEndLatencyMs = getAverageEndToEndLatencyMs(),
            currentMemoryUsageMB = getCurrentMemoryUsage(),
            peakMemoryUsageMB = getPeakMemoryUsage(),
            currentMemoryUsagePercent = getCurrentMemoryUsagePercent(),
            batteryLevel = getCurrentBatteryLevel(),
            isPerformanceDegraded = isPerformanceDegraded.get(),
            thermalState = getCurrentThermalState()
        )
    }
    
    /**
     * Get performance statistics for reporting
     */
    fun getPerformanceStatistics(): PerformanceStatistics {
        return PerformanceStatistics(
            fps = FPSStatistics(
                current = currentFPS.get(),
                average = getAverageFPS(),
                min = getMinFPS(),
                max = getMaxFPS(),
                stdDev = getFPSStandardDeviation()
            ),
            latency = LatencyStatistics(
                inference = getLatencyStatistics(inferenceLatencies),
                preprocessing = getLatencyStatistics(preprocessingLatencies),
                postprocessing = getLatencyStatistics(postprocessingLatencies),
                endToEnd = getLatencyStatistics(endToEndLatencies)
            ),
            memory = MemoryStatistics(
                current = getCurrentMemoryUsageMB(),
                peak = getPeakMemoryUsage(),
                average = getAverageMemoryUsage(),
                p50 = getPercentileMemory(50),
                p95 = getPercentileMemory(95)
            ),
            thermal = ThermalStatistics(
                currentTemperature = getCurrentTemperature(),
                maxTemperature = getMaxTemperature(),
                throttlingEvents = getThrottlingEventCount()
            )
        )
    }
    
    /**
     * Helper methods for statistics calculation
     */
    private fun getAverageFPS(): Double {
        if (fpsHistory.isEmpty()) return 0.0
        return fpsHistory.average()
    }
    
    private fun getMinFPS(): Int {
        return if (fpsHistory.isEmpty()) 0 else fpsHistory.minOrNull() ?: 0
    }
    
    private fun getMaxFPS(): Int {
        return if (fpsHistory.isEmpty()) 0 else fpsHistory.maxOrNull() ?: 0
    }
    
    private fun getFPSStandardDeviation(): Double {
        if (fpsHistory.size < 2) return 0.0
        val mean = getAverageFPS()
        val variance = fpsHistory.map { (it - mean) * (it - mean) }.average()
        return kotlin.math.sqrt(variance)
    }
    
    private fun getAverageInferenceLatencyMs(): Double {
        if (inferenceLatencies.isEmpty()) return 0.0
        return inferenceLatencies.map { it / 1_000_000.0 }.average()
    }
    
    private fun getPercentileInferenceLatency(percentile: Int): Double {
        if (inferenceLatencies.isEmpty()) return 0.0
        val latenciesMs = inferenceLatencies.map { it / 1_000_000.0 }.sorted()
        val index = (latenciesMs.size * percentile / 100.0).toInt().coerceIn(0, latenciesMs.size - 1)
        return latenciesMs[index]
    }
    
    private fun getAveragePreprocessingLatencyMs(): Double {
        if (preprocessingLatencies.isEmpty()) return 0.0
        return preprocessingLatencies.map { it / 1_000_000.0 }.average()
    }
    
    private fun getAveragePostprocessingLatencyMs(): Double {
        if (postprocessingLatencies.isEmpty()) return 0.0
        return postprocessingLatencies.map { it / 1_000_000.0 }.average()
    }
    
    private fun getAverageEndToEndLatencyMs(): Double {
        if (endToEndLatencies.isEmpty()) return 0.0
        return endToEndLatencies.map { it / 1_000_000.0 }.average()
    }
    
    private fun getLatencyStatistics(latencies: ConcurrentLinkedQueue<Long>): LatencyMetric {
        if (latencies.isEmpty()) return LatencyMetric(0.0, 0.0, 0.0, 0.0)
        
        val latenciesMs = latencies.map { it / 1_000_000.0 }.sorted()
        return LatencyMetric(
            average = latenciesMs.average(),
            p50 = latenciesMs[latenciesMs.size * 50 / 100],
            p95 = latenciesMs[latenciesMs.size * 95 / 100],
            p99 = latenciesMs[latenciesMs.size * 99 / 100]
        )
    }
    
    private fun getCurrentMemoryUsage(): Long {
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
    }
    
    private fun getCurrentMemoryUsageMB(): Long {
        return getCurrentMemoryUsage()
    }
    
    private fun getCurrentMemoryUsagePercent(): Double {
        val runtime = Runtime.getRuntime()
        val used = (runtime.totalMemory() - runtime.freeMemory())
        val max = runtime.maxMemory()
        return if (max > 0) (used * 100.0 / max) else 0.0
    }
    
    private fun getPeakMemoryUsage(): Long {
        return memorySnapshots.maxOfOrNull { it.usedMemoryMB } ?: 0
    }
    
    private fun getAverageMemoryUsage(): Double {
        if (memorySnapshots.isEmpty()) return 0.0
        return memorySnapshots.map { it.usedMemoryMB }.average()
    }
    
    private fun getPercentileMemory(percentile: Int): Long {
        if (memorySnapshots.isEmpty()) return 0
        val memoryValues = memorySnapshots.map { it.usedMemoryMB }.sorted()
        val index = (memoryValues.size * percentile / 100).coerceIn(0, memoryValues.size - 1)
        return memoryValues[index]
    }
    
    private fun getCurrentBatteryLevel(): Int {
        return batteryManager?.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
    }
    
    private fun getCurrentTemperature(): Int {
        return batteryManager?.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_TEMPERATURE) ?: -1
    }
    
    private fun getMaxTemperature(): Int {
        return energySnapshots.maxOfOrNull { it.batteryTemperature } ?: -1
    }
    
    private fun getThrottlingEventCount(): Int {
        // Implementation would depend on thermal service availability
        return 0
    }
    
    private fun getCurrentThermalState(): String {
        return "UNKNOWN" // Placeholder - would need thermal service integration
    }
    
    /**
     * Log performance summary for debugging
     */
    private fun logPerformanceSummary() {
        val metrics = getCurrentMetrics()
        val statistics = getPerformanceStatistics()
        
        Log.i(TAG, """
            Performance Summary:
            FPS: ${metrics.currentFPS} (avg: ${statistics.fps.average.toInt()}, min: ${statistics.fps.min}, max: ${statistics.fps.max})
            Latency: Inference ${metrics.averageInferenceLatencyMs.toInt()}ms, End-to-End ${metrics.averageEndToEndLatencyMs.toInt()}ms
            Memory: ${metrics.currentMemoryUsageMB}MB (${metrics.currentMemoryUsagePercent.toInt()}%), Peak: ${metrics.peakMemoryUsageMB}MB
            Battery: ${metrics.batteryLevel}%
            Thermal: ${metrics.thermalState}
            Status: ${if (metrics.isPerformanceDegraded) "DEGRADED" else "NORMAL"}
        """.trimIndent())
    }
    
    /**
     * Reset performance statistics
     */
    fun resetStatistics() {
        frameCount.set(0)
        currentFPS.set(0)
        fpsHistory.clear()
        inferenceLatencies.clear()
        preprocessingLatencies.clear()
        postprocessingLatencies.clear()
        endToEndLatencies.clear()
        memorySnapshots.clear()
        energySnapshots.clear()
        isPerformanceDegraded.set(false)
        Log.i(TAG, "Performance statistics reset")
    }
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        stopMonitoring()
        scope.cancel()
        resetStatistics()
    }
}

/**
 * Data classes for performance monitoring
 */
data class MemorySnapshot(
    val timestamp: Long,
    val totalMemoryMB: Long,
    val usedMemoryMB: Long,
    val maxMemoryMB: Long,
    val nativeHeapSizeMB: Int,
    val dalvikHeapSizeMB: Int,
    val totalPssMB: Int,
    val totalPrivateCleanMB: Int,
    val totalPrivateDirtyMB: Int
)

data class EnergySnapshot(
    val timestamp: Long,
    val batteryLevel: Int,
    val batteryTemperature: Int,
    val batteryCurrentNow: Long,
    val batteryCurrentAverage: Long,
    val chargingStatus: Boolean
)

data class PerformanceMetrics(
    val currentFPS: Int,
    val averageFPS: Double,
    val minFPS: Int,
    val maxFPS: Int,
    val averageInferenceLatencyMs: Double,
    val percentile95InferenceLatencyMs: Double,
    val averagePreprocessingLatencyMs: Double,
    val averagePostprocessingLatencyMs: Double,
    val averageEndToEndLatencyMs: Double,
    val currentMemoryUsageMB: Long,
    val peakMemoryUsageMB: Long,
    val currentMemoryUsagePercent: Double,
    val batteryLevel: Int,
    val isPerformanceDegraded: Boolean,
    val thermalState: String
)

data class PerformanceStatistics(
    val fps: FPSStatistics,
    val latency: LatencyStatistics,
    val memory: MemoryStatistics,
    val thermal: ThermalStatistics
)

data class FPSStatistics(
    val current: Int,
    val average: Double,
    val min: Int,
    val max: Int,
    val stdDev: Double
)

data class LatencyStatistics(
    val inference: LatencyMetric,
    val preprocessing: LatencyMetric,
    val postprocessing: LatencyMetric,
    val endToEnd: LatencyMetric
)

data class LatencyMetric(
    val average: Double,
    val p50: Double,
    val p95: Double,
    val p99: Double
)

data class MemoryStatistics(
    val current: Long,
    val peak: Long,
    val average: Double,
    val p50: Long,
    val p95: Long
)

data class ThermalStatistics(
    val currentTemperature: Int,
    val maxTemperature: Int,
    val throttlingEvents: Int
)

/**
 * Performance thresholds for monitoring
 */
private class PerformanceThresholds(
    val minimumFPS: Int = 15,
    val maxInferenceLatencyMs: Double = 100.0,
    val maxMemoryUsagePercent: Double = 80.0,
    val maxTemperatureCelsius: Int = 40,
    val lowBatteryThreshold: Int = 20
)