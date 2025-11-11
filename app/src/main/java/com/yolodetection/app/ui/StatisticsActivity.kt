package com.yolodetection.app.ui

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MenuItem
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import com.yolodetection.app.R
import com.yolodetection.app.utils.PerformanceManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.text.DecimalFormat
import java.util.concurrent.TimeUnit

/**
 * Statistics Activity for monitoring YOLO detection performance
 */
class StatisticsActivity : AppCompatActivity() {
    
    private lateinit var toolbar: Toolbar
    private lateinit var textViewModelInfo: TextView
    private lateinit var textViewPerformance: TextView
    private lateinit var textViewMemory: TextView
    private lateinit var textViewCamera: TextView
    private lateinit var textViewDetection: TextView
    private lateinit var textViewThermal: TextView
    private lateinit var textViewSystem: TextView
    
    private val updateHandler = Handler(Looper.getMainLooper())
    private val updateIntervalMs = 1000L // Update every second
    
    // Formatters
    private val decimalFormat = DecimalFormat("0.00")
    private val integerFormat = DecimalFormat("#,###")
    
    private val updateRunnable = object : Runnable {
        override fun run() {
            updateStatistics()
            updateHandler.postDelayed(this, updateIntervalMs)
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_statistics)
        
        // Setup toolbar
        toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Performance Statistics"
        
        // Initialize views
        initializeViews()
        
        // Start periodic updates
        startUpdates()
        
        Timber.i("Statistics activity started")
    }
    
    private fun initializeViews() {
        textViewModelInfo = findViewById(R.id.modelInfoText)
        textViewPerformance = findViewById(R.id.performanceText)
        textViewMemory = findViewById(R.id.memoryText)
        textViewCamera = findViewById(R.id.cameraText)
        textViewDetection = findViewById(R.id.detectionText)
        textViewThermal = findViewById(R.id.thermalText)
        textViewSystem = findViewById(R.id.systemText)
    }
    
    private fun startUpdates() {
        updateHandler.post(updateRunnable)
    }
    
    private fun stopUpdates() {
        updateHandler.removeCallbacks(updateRunnable)
    }
    
    private fun updateStatistics() {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                // Get performance statistics
                val stats = PerformanceManager.getPerformanceStats()
                val avgFPS = PerformanceManager.getAverageFPS()
                
                // Update UI with coroutine to avoid blocking main thread
                updateUI(stats, avgFPS)
                
            } catch (e: Exception) {
                Timber.e(e, "Error updating statistics")
            }
        }
    }
    
    private fun updateUI(stats: PerformanceManager.PerformanceStats, avgFPS: Int) {
        // Model Information
        val modelInfoText = buildString {
            append("Model: YOLOv11n\n")
            append("Quantization: FP16\n")
            append("Input Size: 640x640\n")
            append("Classes: 80 (COCO)\n")
            append("Parameters: 2.6M\n")
            append("FLOPs: 6.5B\n")
        }
        textViewModelInfo.text = modelInfoText
        
        // Performance Information
        val performanceText = buildString {
            append("Average FPS: $avgFPS\n")
            append("Current FPS: ${getCurrentFPS()}\n")
            append("Inference Time: ${decimalFormat.format(stats.currentMemoryMB)}ms\n")
            append("Total Inferences: ${integerFormat.format(stats.fpsHistory.size)}\n")
            append("Frames Processed: ${integerFormat.format(stats.fpsHistory.sum())}\n")
            append("Performance: ${getPerformanceLevel(avgFPS)}\n")
        }
        textViewPerformance.text = performanceText
        
        // Memory Information
        val memoryText = buildString {
            append("Current: ${stats.currentMemoryMB} MB\n")
            append("Peak: ${stats.peakMemoryMB} MB\n")
            append("Warnings: ${stats.memoryWarnings}\n")
            append("Memory Level: ${getMemoryLevel(stats.currentMemoryMB)}\n")
        }
        textViewMemory.text = memoryText
        
        // Camera Information
        val cameraText = buildString {
            append("Resolution: 1280x720\n")
            append("Format: YUV420\n")
            append("Frame Rate: 30 FPS\n")
            append("Camera: ${getCameraInfo()}\n")
        }
        textViewCamera.text = cameraText
        
        // Detection Information
        val detectionText = buildString {
            append("Current Detections: ${getCurrentDetections()}\n")
            append("Person Detections: ${getCurrentPersonDetections()}\n")
            append("Confidence Threshold: 0.5\n")
            append("IoU Threshold: 0.5\n")
            append("Max Detections: 300\n")
        }
        textViewDetection.text = detectionText
        
        // Thermal Information
        val thermalText = buildString {
            append("Throttled: ${if (stats.isThrottled) "Yes" else "No"}\n")
            append("Device: ${stats.deviceCapabilities?.deviceModel ?: "Unknown"}\n")
            append("Android: API ${stats.deviceCapabilities?.androidVersion ?: "Unknown"}\n")
            append("Thermal State: ${getThermalState()}\n")
        }
        textViewThermal.text = thermalText
        
        // System Information
        val systemText = buildString {
            append("CPU Cores: ${stats.deviceCapabilities?.cpuCores ?: "Unknown"}\n")
            append("Total Memory: ${getTotalMemoryGB()} GB\n")
            append("Available Memory: ${getAvailableMemoryGB()} GB\n")
            append("Device Type: ${if (stats.deviceCapabilities?.isLowEndDevice == true) "Low-End" else "High-End"}\n")
            append("Emulator: ${if (stats.deviceCapabilities?.isEmulator == true) "Yes" else "No"}\n")
        }
        textViewSystem.text = systemText
    }
    
    private fun getPerformanceLevel(fps: Int): String {
        return when {
            fps >= 25 -> "Excellent"
            fps >= 15 -> "Good"
            fps >= 8 -> "Fair"
            else -> "Poor"
        }
    }
    
    private fun getMemoryLevel(memoryMB: Long): String {
        return when {
            memoryMB < 50 -> "Low"
            memoryMB < 200 -> "Normal"
            memoryMB < 500 -> "High"
            else -> "Critical"
        }
    }
    
    private fun getCurrentFPS(): Int {
        // This would be retrieved from the frame processor
        // For now, return average as placeholder
        return PerformanceManager.getAverageFPS()
    }
    
    private fun getCameraInfo(): String {
        // This would be retrieved from the camera controller
        return "Back Camera"
    }
    
    private fun getCurrentDetections(): Int {
        // This would be retrieved from the detector
        return 0
    }
    
    private fun getCurrentPersonDetections(): Int {
        // This would be retrieved from the detector
        return 0
    }
    
    private fun getThermalState(): String {
        return if (PerformanceManager.isThrottled()) {
            "Throttling"
        } else {
            "Normal"
        }
    }
    
    private fun getTotalMemoryGB(): String {
        val stats = PerformanceManager.getPerformanceStats()
        val totalMemoryGB = (stats.deviceCapabilities?.totalMemory ?: 0L) / (1024 * 1024 * 1024)
        return decimalFormat.format(totalMemoryGB)
    }
    
    private fun getAvailableMemoryGB(): String {
        val stats = PerformanceManager.getPerformanceStats()
        val availableMemoryGB = (stats.deviceCapabilities?.availableMemory ?: 0L) / (1024 * 1024 * 1024)
        return decimalFormat.format(availableMemoryGB)
    }
    
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                onBackPressed()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
    
    override fun onResume() {
        super.onResume()
        startUpdates()
    }
    
    override fun onPause() {
        super.onPause()
        stopUpdates()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        stopUpdates()
        Timber.i("Statistics activity destroyed")
    }
    
    // Placeholder for real-time data methods
    private fun clearStatistics() {
        Timber.i("Statistics cleared")
    }
    
    private fun exportStatistics() {
        Timber.i("Statistics exported")
    }
}