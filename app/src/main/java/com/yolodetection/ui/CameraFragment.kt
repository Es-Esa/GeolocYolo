package com.yolodetection.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import androidx.core.app.ActivityCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import java.util.concurrent.Executor
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

/**
 * Fragment handling Camera2 API integration for real-time human detection
 * Implements proper threading and optimization for smooth camera preview
 */
class CameraFragment : Fragment() {
    
    companion object {
        private const val TAG = "CameraFragment"
        private const val CAMERA_PERMISSION_REQUEST_CODE = 101
        
        // Camera configuration constants
        private const val IMAGE_FORMAT = ImageFormat.YUV_420_888
        private const val MAX_PREVIEW_SIZE = 1920
        private const val PREVIEW_FPS = 30
        
        // Threading constants
        private const val CAMERA_THREAD_NAME = "CameraThread"
        private const val HANDLER_THREAD_NAME = "CameraHandler"
    }

    // Camera related members
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var cameraId: String? = null
    private var previewSize: Size? = null
    private var surfaceTexture: SurfaceTexture? = null
    
    // Threading components
    private var cameraThread: Thread? = null
    private var cameraHandler: Handler? = null
    private var backgroundHandler: Handler? = null
    private val cameraOpenCloseLock = Semaphore(1)
    
    // Callbacks
    private var onImageAvailableCallback: ((android.graphics.Bitmap) -> Unit)? = null
    private var onCameraStateListener: ((String) -> Unit)? = null
    
    // Performance tracking
    private var frameCount = 0
    private var lastFrameTime = 0L
    private var currentFps = 0f
    
    // TextureView for camera preview
    private var textureView: TextureView? = null
    
    // CameraDevice.StateCallback for device lifecycle
    private val cameraStateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraOpenCloseLock.release()
            this@CameraFragment.cameraDevice = camera
            updateCameraState("Camera Opened")
            startCameraCaptureSession()
        }
        
        override fun onDisconnected(camera: CameraDevice) {
            cameraOpenCloseLock.release()
            camera.close()
            this@CameraFragment.cameraDevice = null
            updateCameraState("Camera Disconnected")
        }
        
        override fun onError(camera: CameraDevice, error: Int) {
            cameraOpenCloseLock.release()
            camera.close()
            this@CameraFragment.cameraDevice = null
            updateCameraState("Camera Error: $error")
            Log.e(TAG, "Camera error: $error")
        }
    }
    
    // ImageReader.OnImageAvailableListener for processing frames
    private val imageAvailableListener = ImageReader.OnImageAvailableListener { reader ->
        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener
        
        try {
            // Convert image to bitmap
            val bitmap = convertImageToBitmap(image)
            if (bitmap != null) {
                // Update FPS calculation
                updateFpsCalculation()
                
                // Process on main thread if needed
                activity?.runOnUiThread {
                    onImageAvailableCallback?.invoke(bitmap)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image: ${e.message}")
        } finally {
            image.close()
        }
    }
    
    // TextureView.SurfaceTextureListener for surface lifecycle
    private val surfaceTextureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            surfaceTexture = surface
            startCamera()
        }
        
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
            configureTransform(width, height)
        }
        
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            surfaceTexture = null
            return true
        }
        
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            // Surface texture was updated
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        startCameraThread()
    }
    
    override fun onViewCreated(view: android.view.View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        textureView = view as? TextureView
        setupTextureView()
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        textureView?.surfaceTextureListener = null
    }
    
    override fun onDestroy() {
        super.onDestroy()
        closeCamera()
        stopCameraThread()
    }
    
    /**
     * Start the camera thread for background processing
     */
    private fun startCameraThread() {
        cameraThread = Thread {
            Looper.prepare()
            cameraHandler = Handler(Looper.myLooper()!!)
            val handlerThread = HandlerThread(HANDLER_THREAD_NAME)
            handlerThread.start()
            backgroundHandler = Handler(handlerThread.looper)
            Looper.loop()
        }
        cameraThread?.start()
    }
    
    /**
     * Stop the camera thread
     */
    private fun stopCameraThread() {
        cameraThread?.let { thread ->
            thread.interrupt()
            try {
                thread.join(1000)
            } catch (e: InterruptedException) {
                Log.e(TAG, "Camera thread interrupted: ${e.message}")
            }
        }
        cameraThread = null
        cameraHandler = null
        backgroundHandler = null
    }
    
    /**
     * Setup TextureView with surface texture listener
     */
    private fun setupTextureView() {
        textureView?.let { view ->
            if (view.isAvailable) {
                surfaceTexture = view.surfaceTexture
                startCamera()
            } else {
                view.surfaceTextureListener = surfaceTextureListener
            }
        }
    }
    
    /**
     * Start camera with proper configuration
     */
    private fun startCamera() {
        if (activity?.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission()
            return
        }
        
        if (surfaceTexture == null) {
            updateCameraState("Surface not ready")
            return
        }
        
        updateCameraState("Starting camera...")
        
        cameraHandler?.post {
            try {
                if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                    throw RuntimeException("Time out waiting to lock camera opening.")
                }
                
                val manager = activity?.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                cameraId = selectCamera(manager)
                configureCamera(manager, cameraId!!)
                manager.openCamera(cameraId!!, cameraStateCallback, cameraHandler)
                
            } catch (e: Exception) {
                Log.e(TAG, "Error starting camera: ${e.message}")
                updateCameraState("Camera start failed: ${e.message}")
            }
        }
    }
    
    /**
     * Select the best available camera (prefer back camera for detection)
     */
    private fun selectCamera(manager: CameraManager): String {
        val cameraIds = manager.cameraIdList
        var backCameraId: String? = null
        
        for (id in cameraIds) {
            val characteristics = manager.getCameraCharacteristics(id)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            
            if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                backCameraId = id
                break
            }
        }
        
        return backCameraId ?: cameraIds[0] // Fallback to first available camera
    }
    
    /**
     * Configure camera with optimal settings for human detection
     */
    private fun configureCamera(manager: CameraManager, cameraId: String) {
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        
        if (map == null) {
            throw RuntimeException("Stream configuration map is null")
        }
        
        // Find optimal preview size
        previewSize = findOptimalPreviewSize(map.getOutputSizes(SurfaceTexture::class.java))
        
        // Setup image reader for processing
        imageReader = ImageReader.newInstance(
            previewSize!!.width,
            previewSize!!.height,
            IMAGE_FORMAT,
            2
        ).apply {
            setOnImageAvailableListener(imageAvailableListener, backgroundHandler)
        }
        
        // Configure surface texture
        surfaceTexture?.setDefaultBufferSize(previewSize!!.width, previewSize!!.height)
    }
    
    /**
     * Find optimal preview size for real-time processing
     */
    private fun findOptimalPreviewSize(sizes: Array<Size>): Size {
        var bestSize: Size? = null
        var maxRatio = 0.0
        
        for (size in sizes) {
            val width = size.width
            val height = size.height
            
            if (width <= MAX_PREVIEW_SIZE && height <= MAX_PREVIEW_SIZE) {
                val aspectRatio = width.toDouble() / height
                if (aspectRatio > maxRatio) {
                    maxRatio = aspectRatio
                    bestSize = size
                }
            }
        }
        
        return bestSize ?: sizes[0]
    }
    
    /**
     * Start camera capture session
     */
    private fun startCameraCaptureSession() {
        if (cameraDevice == null || surfaceTexture == null || imageReader == null) {
            return
        }
        
        cameraHandler?.post {
            try {
                val surface = Surface(surfaceTexture)
                val targets = listOf(surface, imageReader!!.surface)
                
                // Create capture session with optimizations
                val outputConfig = OutputConfiguration(surface)
                val sessionConfig = SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR,
                    listOf(outputConfig),
                    { /* executor */ cameraHandler!! as Executor
                    },
                    captureSessionCallback
                )
                
                cameraDevice!!.createCaptureSession(sessionConfig)
                
            } catch (e: Exception) {
                Log.e(TAG, "Error creating capture session: ${e.message}")
            }
        }
    }
    
    // CameraCaptureSession.StateCallback
    private val captureSessionCallback = object : CameraCaptureSession.StateCallback() {
        override fun onConfigured(session: CameraCaptureSession) {
            captureSession = session
            updateCameraState("Camera Ready")
            
            // Start repeating preview request
            val request = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(surfaceTexture as Surface)
                addTarget(imageReader!!.surface)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, PREVIEW_FPS)
            }
            
            session.setRepeatingRequest(request.build(), null, backgroundHandler)
        }
        
        override fun onConfigureFailed(session: CameraCaptureSession) {
            Log.e(TAG, "Failed to configure capture session")
            updateCameraState("Session Configuration Failed")
        }
    }
    
    /**
     * Configure transform matrix for TextureView
     */
    private fun configureTransform(viewWidth: Int, viewHeight: Int) {
        if (surfaceTexture == null || previewSize == null) {
            return
        }
        
        val rotation = activity?.windowManager?.defaultDisplay?.rotation ?: 0
        val matrix = android.graphics.Matrix()
        val viewRect = android.graphics.RectF(0f, 0f, viewWidth.toFloat(), viewHeight.toFloat())
        val bufferRect = android.graphics.RectF(0f, 0f, previewSize!!.width.toFloat(), previewSize!!.height.toFloat())
        val centerX = viewRect.centerX()
        val centerY = viewRect.centerY()
        
        if (rotation == android.view.Surface.ROTATION_90 || rotation == android.view.Surface.ROTATION_270) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())
            matrix.setRectToRect(viewRect, bufferRect, android.graphics.Matrix.ScaleToFit.FILL)
            val scale = kotlin.math.max(
                viewHeight.toFloat() / previewSize!!.height,
                viewWidth.toFloat() / previewSize!!.width
            )
            matrix.postScale(scale, scale, centerX, centerY)
            matrix.postRotate(90 * (rotation - 2), centerX, centerY)
        } else if (rotation == android.view.Surface.ROTATION_180) {
            matrix.postRotate(180f, centerX, centerY)
        }
        
        textureView?.setTransform(matrix)
    }
    
    /**
     * Convert Image to Bitmap for processing
     */
    private fun convertImageToBitmap(image: android.media.Image): android.graphics.Bitmap? {
        try {
            val planes = image.planes
            val yBuffer = planes[0].buffer
            val uBuffer = planes[1].buffer
            val vBuffer = planes[2].buffer
            
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            
            val nv21 = ByteArray(ySize + uSize + vSize)
            
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            
            return YuvToRgbConverter.convertYuvToBitmap(requireContext(), nv21, image.width, image.height)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error converting image: ${e.message}")
            return null
        }
    }
    
    /**
     * Update FPS calculation for performance monitoring
     */
    private fun updateFpsCalculation() {
        val currentTime = System.currentTimeMillis()
        frameCount++
        
        if (currentTime - lastFrameTime >= 1000) {
            currentFps = frameCount * 1000f / (currentTime - lastFrameTime)
            frameCount = 0
            lastFrameTime = currentTime
            
            activity?.runOnUiThread {
                // Update performance stats if available
                (activity as? MainActivity)?.updateFps(currentFps)
            }
        }
    }
    
    /**
     * Update camera state listener
     */
    private fun updateCameraState(state: String) {
        activity?.runOnUiThread {
            onCameraStateListener?.invoke(state)
        }
    }
    
    /**
     * Request camera permission
     */
    private fun requestCameraPermission() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            // Show rationale to user
            updateCameraState("Camera permission required")
        } else {
            requestPermissions(
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE
            )
        }
    }
    
    /**
     * Close camera and release resources
     */
    private fun closeCamera() {
        try {
            cameraOpenCloseLock.acquire()
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            imageReader?.close()
            imageReader = null
            cameraOpenCloseLock.release()
        } catch (e: InterruptedException) {
            Log.e(TAG, "Interrupted while closing camera: ${e.message}")
        }
    }
    
    // Public methods for external control
    fun setOnImageAvailableListener(callback: (android.graphics.Bitmap) -> Unit) {
        onImageAvailableCallback = callback
    }
    
    fun setOnCameraStateListener(callback: (String) -> Unit) {
        onCameraStateListener = callback
    }
    
    fun isCameraReady(): Boolean = cameraDevice != null && captureSession != null
    fun getCurrentFps(): Float = currentFps
    fun getPreviewSize(): Size? = previewSize
}