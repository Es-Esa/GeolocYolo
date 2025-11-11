package com.yolodetection.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentContainerView
import androidx.lifecycle.lifecycleScope
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.yolodetection.R
import kotlinx.coroutines.launch

/**
 * Main activity orchestrating the human detection app
 * Integrates camera preview, detection overlay, settings, and performance monitoring
 */
class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
    }
    
    // UI Components
    private var cameraContainer: FrameLayout? = null
    private var settingsContainer: FrameLayout? = null
    private var testImageContainer: FrameLayout? = null
    private var statusText: TextView? = null
    private var settingsButton: ImageView? = null
    private var captureButton: FloatingActionButton? = null
    private var testImageButton: FloatingActionButton? = null
    private var pauseButton: FloatingActionButton? = null
    
    // Fragments
    private var cameraFragment: CameraFragment? = null
    private var settingsFragment: SettingsFragment? = null
    private var testImagePicker: TestImagePicker? = null
    
    // State
    private var isPaused = false
    private var isProcessing = false
    private var currentFps = 0f
    
    // Detection overlay
    private var overlayView: DetectionOverlayView? = null
    private var performanceStatsView: PerformanceStatsView? = null
    
    // Permission launcher
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            updateStatus("Camera permission denied")
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initializeViews()
        setupFragments()
        setupListeners()
        checkPermissionsAndStart()
    }
    
    /**
     * Initialize all view references
     */
    private fun initializeViews() {
        cameraContainer = findViewById(R.id.cameraContainer)
        settingsContainer = findViewById(R.id.settingsContainer)
        testImageContainer = findViewById(R.id.testImageContainer)
        statusText = findViewById(R.id.statusText)
        settingsButton = findViewById(R.id.settingsButton)
        captureButton = findViewById(R.id.captureButton)
        testImageButton = findViewById(R.id.testImageButton)
        pauseButton = findViewById(R.id.pauseButton)
        
        // Get overlay and performance stats views
        overlayView = findViewById(R.id.overlayView)
        performanceStatsView = findViewById(R.id.performanceStatsView)
    }
    
    /**
     * Setup fragments
     */
    private fun setupFragments() {
        // Initialize camera fragment
        cameraFragment = CameraFragment().apply {
            setOnImageAvailableListener { bitmap ->
                processImage(bitmap)
            }
            setOnCameraStateListener { state ->
                runOnUiThread {
                    updateStatus(state)
                }
            }
        }
        
        // Initialize settings fragment
        settingsFragment = SettingsFragment().apply {
            setOnSettingsChangedListener { settings ->
                applySettings(settings)
            }
        }
        
        // Initialize test image picker
        testImagePicker = TestImagePicker().apply {
            setOnImageSelectedListener { bitmap, source ->
                processTestImage(bitmap, source)
            }
            setOnCloseListener {
                hideTestImagePicker()
            }
        }
    }
    
    /**
     * Setup click listeners
     */
    private fun setupListeners() {
        // Settings button
        settingsButton?.setOnClickListener {
            showSettings()
        }
        
        // Capture button
        captureButton?.setOnClickListener {
            captureCurrentFrame()
        }
        
        // Test image button
        testImageButton?.setOnClickListener {
            showTestImagePicker()
        }
        
        // Pause button
        pauseButton?.setOnClickListener {
            togglePause()
        }
    }
    
    /**
     * Check permissions and start camera
     */
    private fun checkPermissionsAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
    
    /**
     * Start camera preview
     */
    private fun startCamera() {
        // Add camera fragment to container
        supportFragmentManager.beginTransaction()
            .replace(R.id.cameraContainer, cameraFragment!!)
            .commit()
        
        updateStatus("Camera starting...")
    }
    
    /**
     * Process image for human detection
     */
    private fun processImage(bitmap: android.graphics.Bitmap) {
        if (isPaused || isProcessing) {
            return
        }
        
        isProcessing = true
        
        lifecycleScope.launch {
            try {
                val startTime = System.currentTimeMillis()
                
                // TODO: Implement actual YOLO detection here
                // This is a placeholder implementation
                val mockDetections = generateMockDetections(bitmap)
                
                val processingTime = System.currentTimeMillis() - startTime
                
                // Update UI on main thread
                runOnUiThread {
                    overlayView?.setDetections(mockDetections)
                    performanceStatsView?.updateDetections(
                        totalCount = mockDetections.size,
                        humanCount = mockDetections.count { it.isHuman },
                        processingTimeMs = processingTime.toFloat(),
                        inferenceTimeMs = (processingTime * 0.8).toFloat()
                    )
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing image: ${e.message}")
                updateStatus("Processing error: ${e.message}")
            } finally {
                isProcessing = false
            }
        }
    }
    
    /**
     * Process test image
     */
    private fun processTestImage(bitmap: android.graphics.Bitmap, source: String) {
        Log.d(TAG, "Processing test image from: $source")
        
        // Hide test image picker and show results
        hideTestImagePicker()
        
        // Process the test image
        processImage(bitmap)
        
        updateStatus("Testing with image: $source")
    }
    
    /**
     * Generate mock detections for testing
     */
    private fun generateMockDetections(bitmap: android.graphics.Bitmap): List<DetectionOverlayView.Detection> {
        val detections = mutableListOf<DetectionOverlayView.Detection>()
        val width = bitmap.width.toFloat()
        val height = bitmap.height.toFloat()
        
        // Generate 1-3 mock human detections
        val detectionCount = (1..3).random()
        
        repeat(detectionCount) { index ->
            val centerX = (0.2f + 0.6f * Math.random()).toFloat() * width
            val centerY = (0.2f + 0.6f * Math.random()).toFloat() * height
            val boxWidth = (0.1f + 0.3f * Math.random()).toFloat() * width
            val boxHeight = (0.2f + 0.5f * Math.random()).toFloat() * height
            
            val detection = DetectionOverlayView.Detection(
                boundingBox = android.graphics.RectF(
                    centerX - boxWidth / 2,
                    centerY - boxHeight / 2,
                    centerX + boxWidth / 2,
                    centerY + boxHeight / 2
                ),
                confidence = (0.6f + 0.4f * Math.random().toFloat()),
                className = "person",
                classIndex = 0,
                isHuman = true,
                trackingId = index + 1
            )
            detections.add(detection)
        }
        
        return detections
    }
    
    /**
     * Show settings fragment
     */
    private fun showSettings() {
        settingsContainer?.visibility = View.VISIBLE
        
        supportFragmentManager.beginTransaction()
            .replace(R.id.settingsContainer, settingsFragment!!)
            .addToBackStack("settings")
            .commit()
    }
    
    /**
     * Show test image picker
     */
    private fun showTestImagePicker() {
        testImageContainer?.visibility = View.VISIBLE
        
        supportFragmentManager.beginTransaction()
            .replace(R.id.testImageContainer, testImagePicker!!)
            .addToBackStack("test_image")
            .commit()
    }
    
    /**
     * Hide test image picker
     */
    private fun hideTestImagePicker() {
        testImageContainer?.visibility = View.GONE
        
        supportFragmentManager.popBackStack("test_image", 0)
    }
    
    /**
     * Capture current frame
     */
    private fun captureCurrentFrame() {
        updateStatus("Capturing frame...")
        // TODO: Implement frame capture
    }
    
    /**
     * Toggle pause/resume
     */
    private fun togglePause() {
        isPaused = !isPaused
        
        if (isPaused) {
            pauseButton?.setImageResource(android.R.drawable.ic_media_play)
            updateStatus("Paused")
        } else {
            pauseButton?.setImageResource(android.R.drawable.ic_media_pause)
            updateStatus("Running")
        }
    }
    
    /**
     * Apply settings changes
     */
    private fun applySettings(settings: SettingsFragment.SettingsData) {
        Log.d(TAG, "Applying settings: $settings")
        
        // Update overlay threshold
        overlayView?.setConfidenceThreshold(settings.confidenceThreshold)
        
        // Update performance stats
        performanceStatsView?.setTargetFps(settings.processingFps.toFloat())
        
        // TODO: Apply other settings to detection pipeline
    }
    
    /**
     * Update status text
     */
    private fun updateStatus(status: String) {
        statusText?.text = status
        Log.d(TAG, "Status: $status")
    }
    
    /**
     * Update FPS display
     */
    fun updateFps(fps: Float) {
        currentFps = fps
        performanceStatsView?.updateFps(fps)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Clean up fragments
        supportFragmentManager.popBackStack()
    }
    
    override fun onBackPressed() {
        when {
            settingsContainer?.visibility == View.VISIBLE -> {
                settingsContainer?.visibility = View.GONE
                supportFragmentManager.popBackStack("settings", 0)
            }
            testImageContainer?.visibility == View.VISIBLE -> {
                hideTestImagePicker()
            }
            else -> {
                super.onBackPressed()
            }
        }
    }
}