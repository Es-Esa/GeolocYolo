package com.yolodetection.app.utils

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.Process
import timber.log.Timber
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicInteger

/**
 * Performance monitoring and optimization manager
 * 
 * Handles memory management, thermal throttling, and performance adaptation
 */
object PerformanceManager {
    
    // Memory monitoring
    private val memoryUsage = AtomicLong(0)
    private val peakMemoryUsage = AtomicLong(0)
    private val memoryWarnings = AtomicInteger(0)
    
    // Performance tracking
    private val fpsHistory = mutableListOf<Int>()
    private val fpsHistoryMaxSize = 60 // 1 minute of FPS data
    private var isThermalThrottled = false
    private var lastThermalCheck = 0L
    
    // Device capabilities
    private var deviceCapabilities: DeviceCapabilities? = null
    
    /**
     * Initialize performance manager
     */
    fun initialize(context: Context) {
        detectDeviceCapabilities(context)
        startMemoryMonitoring(context)
        
        Timber.i("Performance manager initialized for device: ${Build.MODEL}")
        Timber.i("Device capabilities: ${deviceCapabilities?.toString()}")
    }
    
    /**
     * Detect device capabilities for performance optimization
     */
    private fun detectDeviceCapabilities(context: Context) {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        deviceCapabilities = DeviceCapabilities(
            totalMemory = memoryInfo.totalMem,
            availableMemory = memoryInfo.availMem,
            isLowEndDevice = isLowEndDevice(activityManager),
            cpuCores = Runtime.getRuntime().availableProcessors(),
            deviceModel = Build.MODEL,
            androidVersion = Build.VERSION.SDK_INT,
            isEmulator = isEmulator()
        )
    }
    
    /**
     * Check if device is low-end
     */
    private fun isLowEndDevice(activityManager: ActivityManager): Boolean {
        return try {
            val deviceConfig = activityManager.deviceConfigurationInfo
            val glEsVersion = deviceConfig.glEsVersion.toInt()
            val memoryClass = activityManager.memoryClass
            
            // Consider device low-end if:
            // - Has less than 4GB RAM
            // - OpenGL ES version < 3.0
            // - Has less than 4 CPU cores
            memoryClass < 1024 || glEsVersion < 300000 || Runtime.getRuntime().availableProcessors() < 4
        } catch (e: Exception) {
            true // Default to low-end if we can't determine
        }
    }
    
    /**
     * Check if running on emulator
     */
    private fun isEmulator(): Boolean {
        return Build.FINGERPRINT.startsWith("generic") ||
                Build.FINGERPRINT.startsWith("unknown") ||
                Build.MODEL.contains("google_sdk") ||
                Build.MODEL.contains("Emulator") ||
                Build.MODEL.contains("Android SDK built for x86") ||
                Build.MANUFACTURER.contains("Genymotion") ||
                Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic")
    }
    
    /**
     * Start memory monitoring
     */
    private fun startMemoryMonitoring(context: Context) {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val processInfo = activityManager.myMemoryPid()?.let { 
            activityManager.getRunningAppProcesses()?.find { it.pid == it }
        }
        
        // Start periodic memory monitoring
        Thread {
            while (true) {
                try {
                    updateMemoryUsage()
                    Thread.sleep(5000) // Check every 5 seconds
                } catch (e: InterruptedException) {
                    break
                } catch (e: Exception) {
                    Timber.e(e, "Error in memory monitoring")
                }
            }
        }.start()
    }
    
    /**
     * Update memory usage statistics
     */
    private fun updateMemoryUsage() {
        try {
            val runtime = Runtime.getRuntime()
            val usedMemory = runtime.totalMemory() - runtime.freeMemory()
            memoryUsage.set(usedMemory)
            
            val maxMemory = runtime.maxMemory()
            if (usedMemory > peakMemoryUsage.get()) {
                peakMemoryUsage.set(usedMemory)
            }
            
            val memoryUsageMB = usedMemory / (1024 * 1024)
            val maxMemoryMB = maxMemory / (1024 * 1024)
            val usagePercentage = (usedMemory * 100.0 / maxMemory).toInt()
            
            if (usagePercentage > 80) {
                memoryWarnings.incrementAndGet()
                Timber.w("High memory usage: $memoryUsageMB MB / $maxMemoryMB MB ($usagePercentage%)")
            }
            
        } catch (e: Exception) {
            Timber.e(e, "Error updating memory usage")
        }
    }
    
    /**
     * Handle memory pressure
     */
    fun handleMemoryPressure(level: Int) {
        Timber.i("Memory pressure received, level: $level")
        
        when (level) {
            android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE -> {
                // App is running but system needs memory
                clearCaches()
            }
            android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW -> {
                // App is running but system needs memory badly
                clearCaches()
                clearLogs()
            }
            android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> {
                // App is running but system needs memory very badly
                clearCaches()
                clearLogs()
                forceGarbageCollection()
            }
            android.content.ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN -> {
                // App is no longer visible, good time to clean up
                clearCaches()
            }
        }
    }
    
    /**
     * Handle thermal throttling
     */
    fun handleThermalThrottling() {
        isThermalThrottled = true
        lastThermalCheck = System.currentTimeMillis()
        Timber.w("Thermal throttling detected, reducing performance")
    }
    
    /**
     * Check if device is thermally throttled
     */
    fun isThrottled(): Boolean {
        // Reset throttling after 30 seconds
        if (isThermalThrottled && System.currentTimeMillis() - lastThermalCheck > 30000) {
            isThermalThrottled = false
        }
        return isThermalThrottled
    }
    
    /**
     * Get performance recommendations based on device capabilities
     */
    fun getPerformanceRecommendations(): List<String> {
        val recommendations = mutableListOf<String>()
        
        deviceCapabilities?.let { cap ->
            if (cap.isLowEndDevice) {
                recommendations.add("Consider using smaller input size (416x416)")
                recommendations.add("Enable aggressive frame skipping")
                recommendations.add("Use CPU delegate for best compatibility")
            }
            
            if (cap.totalMemory < 4L * 1024 * 1024 * 1024) { // Less than 4GB
                recommendations.add("Monitor memory usage closely")
                recommendations.add("Enable memory-conscious mode")
            }
            
            if (cap.cpuCores >= 8) {
                recommendations.add("Can utilize more threads for preprocessing")
            }
        }
        
        return recommendations
    }
    
    /**
     * Get performance statistics
     */
    fun getPerformanceStats(): PerformanceStats {
        return PerformanceStats(
            currentMemoryMB = memoryUsage.get() / (1024 * 1024),
            peakMemoryMB = peakMemoryUsage.get() / (1024 * 1024),
            memoryWarnings = memoryWarnings.get(),
            isThrottled = isThrottled(),
            deviceCapabilities = deviceCapabilities,
            fpsHistory = fpsHistory.toList()
        )
    }
    
    /**
     * Record FPS for performance monitoring
     */
    fun recordFPS(fps: Int) {
        synchronized(fpsHistory) {
            fpsHistory.add(fps)
            if (fpsHistory.size > fpsHistoryMaxSize) {
                fpsHistory.removeAt(0)
            }
        }
    }
    
    /**
     * Get average FPS
     */
    fun getAverageFPS(): Int {
        synchronized(fpsHistory) {
            return if (fpsHistory.isNotEmpty()) {
                fpsHistory.average().toInt()
            } else {
                0
            }
        }
    }
    
    private fun clearCaches() {
        // Clear any application caches
        Timber.i("Clearing application caches")
    }
    
    private fun clearLogs() {
        // Clear old logs to free memory
        Timber.i("Clearing old logs")
    }
    
    private fun forceGarbageCollection() {
        // Force garbage collection
        System.gc()
        Timber.i("Forced garbage collection")
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        fpsHistory.clear()
        memoryUsage.set(0)
        peakMemoryUsage.set(0)
        Timber.i("Performance manager cleaned up")
    }
    
    /**
     * Device capabilities data class
     */
    data class DeviceCapabilities(
        val totalMemory: Long,
        val availableMemory: Long,
        val isLowEndDevice: Boolean,
        val cpuCores: Int,
        val deviceModel: String,
        val androidVersion: Int,
        val isEmulator: Boolean
    ) {
        override fun toString(): String {
            return "DeviceCapabilities(totalMemory=${totalMemory/(1024*1024)}MB, " +
                    "cpuCores=$cpuCores, " +
                    "isLowEnd=$isLowEndDevice, " +
                    "model=$deviceModel, " +
                    "android=$androidVersion)"
        }
    }
    
    /**
     * Performance statistics data class
     */
    data class PerformanceStats(
        val currentMemoryMB: Long,
        val peakMemoryMB: Long,
        val memoryWarnings: Int,
        val isThrottled: Boolean,
        val deviceCapabilities: DeviceCapabilities?,
        val fpsHistory: List<Int>
    ) {
        fun getSummary(): String {
            val avgFPS = if (fpsHistory.isNotEmpty()) fpsHistory.average().toInt() else 0
            return "Memory: ${currentMemoryMB}MB | " +
                    "Peak: ${peakMemoryMB}MB | " +
                    "Avg FPS: $avgFPS | " +
                    "Throttled: $isThrottled"
        }
    }
}

// Extension function to get process memory info
private fun ActivityManager.myMemoryPid(): Int? {
    return try {
        Process.myPid()
    } catch (e: Exception) {
        null
    }
}