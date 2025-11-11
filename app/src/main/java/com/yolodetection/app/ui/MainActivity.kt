package com.yolodetection.app.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.rememberMultiplePermissionsState
import com.yolodetection.app.databinding.ActivityMainBinding
import com.yolodetection.app.detection.YoloDetector
import com.yolodetection.app.detection.models.Detection
import com.yolodetection.app.overlay.OverlayView
import com.yolodetection.app.utils.FrameProcessor
import com.yolodetection.app.utils.PermissionUtils
import kotlinx.coroutines.launch
import timber.log.Timber
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Main Activity for YOLOv11n Human Detection
 * 
 * Integrates Camera2 API for real-time camera preview and TFLite inference
 * for human detection with bounding box visualization
 */
class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageAnalyzer: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    
    // Core components
    private var yoloDetector: YoloDetector? = null
    private var frameProcessor: FrameProcessor? = null
    private var overlayView: OverlayView? = null
    
    // Camera components
    private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    
    // Performance monitoring
    private var isProcessing = false
    private var frameCount = 0
    private var lastFpsUpdate = 0L
    
    @OptIn(ExperimentalPermissionsApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Initialize executor
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Setup UI
        setupUI()
        setupButtonListeners()
        
        // Setup permissions
        val permissionState = rememberMultiplePermissionsState(
            permissions = listOf(
                Manifest.permission.CAMERA
            )
        )
        
        if (permissionState.allPermissionsGranted) {
            initializeCamera()
        } else {
            requestCameraPermission(permissionState)
        }
    }
    
    private fun setupUI() {
        // Set up overlay view
        overlayView = binding.overlayView.apply {
            setZOrderOnTop(true)
            setZOrderMediaOverlay(true)
        }
        
        // Set up statistics text
        binding.statsText.apply {
            text = "Initializing..."
            visibility = View.GONE // Hide until model is ready
        }
        
        // Set up resolution selector
        binding.resolutionSelector.setOnClickListener {
            showResolutionDialog()
        }
        
        // Set up settings button
        binding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        
        // Set up statistics button
        binding.statisticsButton.setOnClickListener {
            startActivity(Intent(this, StatisticsActivity::class.java))
        }
    }
    
    private fun setupButtonListeners() {
        binding.switchCameraButton.setOnClickListener {
            switchCamera()
        }
        
        binding.toggleDetectionButton.setOnClickListener {
            toggleDetection()
        }
        
        binding.captureButton.setOnClickListener {
            captureImage()
        }
    }
    
    @OptIn(ExperimentalPermissionsApi::class)
    private fun requestCameraPermission(permissionState: androidx.compose.runtime.State<List<String>>) {
        val launcher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            if (permissions[Manifest.permission.CAMERA] == true) {
                initializeCamera()
            } else {
                Toast.makeText(
                    this,
                    "Camera permission is required for this app",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
        
        launcher.launch(arrayOf(Manifest.permission.CAMERA))
    }
    
    private fun initializeCamera() {
        lifecycleScope.launch {
            try {
                // Initialize YOLO detector
                yoloDetector = YoloDetector(this@MainActivity).apply {
                    initialize()
                }
                
                // Initialize frame processor
                frameProcessor = FrameProcessor(
                    yoloDetector = yoloDetector!!,
                    overlayView = overlayView!!,
                    onDetectionUpdate = { detections ->
                        updateDetections(detections)
                    }
                )
                
                // Setup camera
                setupCamera()
                
                // Update UI
                binding.statsText.visibility = View.VISIBLE
                binding.statsText.text = "Model loaded, ready for detection"
                
                Timber.i("Camera and detector initialized successfully")
                
            } catch (e: Exception) {
                Timber.e(e, "Failed to initialize camera and detector")
                Toast.makeText(
                    this@MainActivity,
                    "Failed to initialize: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }
    
    private fun setupCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            
            // Setup preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(binding.previewView.display.rotation)
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }
            
            // Setup image analysis
            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setImageQueueDepth(3)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        processImage(image)
                    }
                }
            
            // Bind camera lifecycle
            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
                
                Timber.i("Camera bound to lifecycle successfully")
                
            } catch (e: Exception) {
                Timber.e(e, "Failed to bind camera")
                Toast.makeText(this, "Failed to bind camera: ${e.message}", Toast.LENGTH_SHORT).show()
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun processImage(image: ImageProxy) {
        // Skip if already processing
        if (isProcessing) {
            image.close()
            return
        }
        
        isProcessing = true
        
        lifecycleScope.launch {
            try {
                // Convert ImageProxy to Bitmap
                val bitmap = image.toBitmap()
                
                // Process frame
                frameProcessor?.processFrame(bitmap, image.imageInfo.rotationDegrees.toFloat())
                
                // Update FPS counter
                updateFpsCounter()
                
            } catch (e: Exception) {
                Timber.e(e, "Error processing image")
            } finally {
                isProcessing = false
                image.close()
            }
        }
    }
    
    private fun updateDetections(detections: List<Detection>) {
        // Update overlay with detections
        overlayView?.setDetections(detections)
        
        // Update statistics
        val personDetections = detections.filter { it.classId == 0 } // Person class
        val info = "Detections: ${detections.size} | Persons: ${personDetections.size}"
        binding.statsText.text = info
    }
    
    private fun updateFpsCounter() {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastFpsUpdate > 1000) {
            val fps = (frameCount * 1000.0 / (currentTime - lastFpsUpdate)).toInt()
            binding.fpsText.text = "${fps} FPS"
            frameCount = 0
            lastFpsUpdate = currentTime
        }
        frameCount++
    }
    
    private fun switchCamera() {
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
        
        // Re-bind camera with new selector
        setupCamera()
    }
    
    private fun toggleDetection() {
        val isEnabled = frameProcessor?.isEnabled ?: false
        frameProcessor?.setEnabled(!isEnabled)
        
        val buttonText = if (isEnabled) "Detection ON" else "Detection OFF"
        binding.toggleDetectionButton.text = buttonText
    }
    
    private fun captureImage() {
        // Implement image capture functionality
        Toast.makeText(this, "Capture functionality coming soon", Toast.LENGTH_SHORT).show()
    }
    
    private fun showResolutionDialog() {
        // Show resolution selection dialog
        val resolutions = listOf("320x320", "416x416", "640x640", "832x832")
        // TODO: Implement dialog for resolution selection
        Toast.makeText(this, "Resolution selector coming soon", Toast.LENGTH_SHORT).show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
        yoloDetector?.cleanup()
        frameProcessor?.cleanup()
    }
    
    override fun onPause() {
        super.onPause()
        // Pause camera and detection when app is paused
        cameraProvider?.unbind(imageAnalyzer)
    }
    
    override fun onResume() {
        super.onResume()
        // Resume camera and detection when app is resumed
        if (yoloDetector != null) {
            setupCamera()
        }
    }
}