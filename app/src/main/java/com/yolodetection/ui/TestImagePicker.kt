package com.yolodetection.ui

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.progressindicator.CircularProgressIndicator
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.*
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.net.URL
import javax.net.ssl.HttpsURLConnection

/**
 * Interface for manual frame input testing and image processing
 * Provides multiple input methods and preview capabilities
 */
class TestImagePicker : Fragment() {
    
    companion object {
        private const val TAG = "TestImagePicker"
        
        // Request codes
        private const val REQUEST_GALLERY = 1001
        private const val REQUEST_CAMERA = 1002
        private const val REQUEST_URL_DOWNLOAD = 1003
        
        // Image processing constants
        private const val MAX_IMAGE_SIZE = 1920
        private const val JPEG_QUALITY = 90
        private const val DOWNLOAD_TIMEOUT = 10000 // ms
        
        // Predefined test images
        private val SAMPLE_IMAGE_URLS = arrayOf(
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
            "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=400",
            "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400",
            "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400",
            "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400"
        )
    }
    
    // UI Components
    private var recyclerView: RecyclerView? = null
    private var progressIndicator: CircularProgressIndicator? = null
    private var urlEditText: TextInputEditText? = null
    private var processButton: MaterialButton? = null
    private var clearButton: MaterialButton? = null
    private var closeButton: FloatingActionButton? = null
    
    // State management
    private var currentBitmap: Bitmap? = null
    private var isProcessing = false
    private var testResults = mutableListOf<TestResult>()
    
    // Callbacks
    private var onImageSelectedListener: ((Bitmap, String) -> Unit)? = null
    private var onCloseListener: (() -> Unit)? = null
    
    // RecyclerView adapter
    private var imageAdapter: TestImageAdapter? = null
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_test_image_picker, container, false)
        
        initializeViews(view)
        setupRecyclerView()
        setupListeners()
        loadSampleImages()
        
        return view
    }
    
    /**
     * Initialize view references
     */
    private fun initializeViews(view: View) {
        recyclerView = view.findViewById(R.id.sampleImagesRecyclerView)
        progressIndicator = view.findViewById(R.id.progressIndicator)
        urlEditText = view.findViewById(R.id.urlEditText)
        processButton = view.findViewById(R.id.processButton)
        clearButton = view.findViewById(R.id.clearButton)
        closeButton = view.findViewById(R.id.closeButton)
    }
    
    /**
     * Setup RecyclerView for sample images
     */
    private fun setupRecyclerView() {
        imageAdapter = TestImageAdapter(
            emptyList(),
            onSampleImageClick = { imageUrl, bitmap ->
                handleSampleImageSelection(bitmap, imageUrl)
            }
        )
        
        recyclerView?.apply {
            layoutManager = GridLayoutManager(requireContext(), 2)
            adapter = imageAdapter
        }
    }
    
    /**
     * Setup all event listeners
     */
    private fun setupListeners() {
        // Gallery button
        view?.findViewById<MaterialButton>(R.id.galleryButton)?.setOnClickListener {
            openGallery()
        }
        
        // Camera button
        view?.findViewById<MaterialButton>(R.id.cameraButton)?.setOnClickListener {
            openCamera()
        }
        
        // URL download button
        processButton?.setOnClickListener {
            downloadImageFromUrl()
        }
        
        // Clear button
        clearButton?.setOnClickListener {
            clearSelection()
        }
        
        // Close button
        closeButton?.setOnClickListener {
            onCloseListener?.invoke()
        }
    }
    
    /**
     * Load sample images for testing
     */
    private fun loadSampleImages() {
        showProgress(true)
        
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val bitmaps = mutableListOf<SampleImage>()
                
                // Download sample images
                SAMPLE_IMAGE_URLS.forEach { url ->
                    try {
                        val bitmap = downloadImageFromUrl(url)
                        if (bitmap != null) {
                            bitmaps.add(SampleImage(bitmap, url, "Sample Image"))
                        }
                    } catch (e: Exception) {
                        // Log error but continue with other images
                        android.util.Log.w(TAG, "Failed to download sample image: $url", e)
                    }
                }
                
                // Update UI on main thread
                withContext(Dispatchers.Main) {
                    imageAdapter?.updateImages(bitmaps)
                    showProgress(false)
                }
                
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showProgress(false)
                    showError("Failed to load sample images: ${e.message}")
                }
            }
        }
    }
    
    /**
     * Download image from URL (suspend function for coroutines)
     */
    private suspend fun downloadImageFromUrl(url: String): Bitmap? = withContext(Dispatchers.IO) {
        try {
            val imageUrl = URL(url)
            val connection = imageUrl.openConnection() as HttpsURLConnection
            connection.apply {
                doInput = true
                connectTimeout = DOWNLOAD_TIMEOUT
                readTimeout = DOWNLOAD_TIMEOUT
                requestMethod = "GET"
                connect()
            }
            
            val inputStream = connection.inputStream
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream.close()
            connection.disconnect()
            
            bitmap?.let { scaleImageToFit(it) }
            
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                showError("Download failed: ${e.message}")
            }
            null
        }
    }
    
    /**
     * Open gallery for image selection
     */
    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI).apply {
            type = "image/*"
        }
        startActivityForResult(intent, REQUEST_GALLERY)
    }
    
    /**
     * Open camera for image capture
     */
    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
            // Create a temporary file for the captured image
            val photoFile = createTempImageFile()
            if (photoFile != null) {
                val photoURI = Uri.fromFile(photoFile)
                putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
            }
        }
        startActivityForResult(intent, REQUEST_CAMERA)
    }
    
    /**
     * Create temporary file for camera capture
     */
    private fun createTempImageFile(): File? {
        return try {
            val timeStamp = System.currentTimeMillis()
            val imageFileName = "test_image_$timeStamp"
            val storageDir = requireContext().cacheDir
            File.createTempFile(imageFileName, ".jpg", storageDir)
        } catch (e: IOException) {
            showError("Failed to create temporary file: ${e.message}")
            null
        }
    }
    
    /**
     * Download image from URL edit text
     */
    private fun downloadImageFromUrl() {
        val url = urlEditText?.text?.toString()?.trim()
        
        if (url.isNullOrEmpty()) {
            urlEditText?.error = "Please enter a valid URL"
            return
        }
        
        if (!isValidUrl(url)) {
            urlEditText?.error = "Please enter a valid image URL"
            return
        }
        
        showProgress(true)
        
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val bitmap = downloadImageFromUrl(url)
                withContext(Dispatchers.Main) {
                    showProgress(false)
                    bitmap?.let {
                        handleImageSelection(it, url)
                    } ?: run {
                        showError("Failed to download image from URL")
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showProgress(false)
                    showError("Download error: ${e.message}")
                }
            }
        }
    }
    
    /**
     * Handle sample image selection
     */
    private fun handleSampleImageSelection(bitmap: Bitmap, source: String) {
        handleImageSelection(bitmap, "Sample: $source")
    }
    
    /**
     * Handle image selection from any source
     */
    private fun handleImageSelection(bitmap: Bitmap, source: String) {
        // Scale image if necessary
        val processedBitmap = scaleImageToFit(bitmap)
        
        currentBitmap = processedBitmap
        isProcessing = false
        
        // Show image preview if available
        showImagePreview(processedBitmap, source)
        
        // Notify listener
        onImageSelectedListener?.invoke(processedBitmap, source)
    }
    
    /**
     * Scale image to fit within maximum size while maintaining aspect ratio
     */
    private fun scaleImageToFit(bitmap: Bitmap): Bitmap {
        val maxSize = MAX_IMAGE_SIZE
        val width = bitmap.width
        val height = bitmap.height
        
        return if (width > maxSize || height > maxSize) {
            val aspectRatio = width.toFloat() / height
            val newWidth = if (aspectRatio > 1) maxSize else (maxSize * aspectRatio).toInt()
            val newHeight = if (aspectRatio > 1) (maxSize / aspectRatio).toInt() else maxSize
            
            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        } else {
            bitmap
        }
    }
    
    /**
     * Show image preview
     */
    private fun showImagePreview(bitmap: Bitmap, source: String) {
        // Update any preview ImageView if exists
        val previewImageView = view?.findViewById<ImageView>(R.id.selectedImagePreview)
        previewImageView?.setImageBitmap(bitmap)
        previewImageView?.visibility = View.VISIBLE
        
        // Update source text
        val sourceTextView = view?.findViewById<TextView>(R.id.selectedImageSource)
        sourceTextView?.text = "Source: $source"
        sourceTextView?.visibility = View.VISIBLE
        
        // Enable process button
        processButton?.isEnabled = true
    }
    
    /**
     * Clear current selection
     */
    private fun clearSelection() {
        currentBitmap = null
        isProcessing = false
        
        // Hide preview
        view?.findViewById<ImageView>(R.id.selectedImagePreview)?.visibility = View.GONE
        view?.findViewById<TextView>(R.id.selectedImageSource)?.visibility = View.GONE
        
        // Clear URL text
        urlEditText?.setText("")
        
        // Disable process button
        processButton?.isEnabled = false
    }
    
    /**
     * Show progress indicator
     */
    private fun showProgress(show: Boolean) {
        progressIndicator?.visibility = if (show) View.VISIBLE else View.GONE
        
        // Disable all controls during progress
        processButton?.isEnabled = !show
        clearButton?.isEnabled = !show
        view?.findViewById<MaterialButton>(R.id.galleryButton)?.isEnabled = !show
        view?.findViewById<MaterialButton>(R.id.cameraButton)?.isEnabled = !show
    }
    
    /**
     * Show error message
     */
    private fun showError(message: String) {
        // Show in a toast or snackbar
        Toast.makeText(requireContext(), message, Toast.LENGTH_SHORT).show()
        android.util.Log.e(TAG, message)
    }
    
    /**
     * Validate URL
     */
    private fun isValidUrl(url: String): Boolean {
        return try {
            val urlObj = java.net.URL(url)
            urlObj.protocol == "http" || urlObj.protocol == "https"
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Handle activity result
     */
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                REQUEST_GALLERY -> {
                    val selectedImageUri = data?.data
                    selectedImageUri?.let { uri ->
                        loadImageFromUri(uri, "Gallery Selection")
                    }
                }
                
                REQUEST_CAMERA -> {
                    // Image already saved to temporary file by the camera app
                    // The camera app will return the result here
                    val bitmap = data?.extras?.get("data") as? Bitmap
                    if (bitmap != null) {
                        handleImageSelection(bitmap, "Camera Capture")
                    }
                }
                
                REQUEST_URL_DOWNLOAD -> {
                    // This might be used for future URL handling
                }
            }
        }
    }
    
    /**
     * Load image from URI
     */
    private fun loadImageFromUri(uri: Uri, source: String) {
        showProgress(true)
        
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val inputStream = requireContext().contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()
                
                withContext(Dispatchers.Main) {
                    showProgress(false)
                    bitmap?.let {
                        handleImageSelection(it, source)
                    } ?: run {
                        showError("Failed to load image from gallery")
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showProgress(false)
                    showError("Error loading image: ${e.message}")
                }
            }
        }
    }
    
    /**
     * Add test result to history
     */
    fun addTestResult(result: TestResult) {
        testResults.add(0, result) // Add to beginning
        
        // Keep only last 10 results
        if (testResults.size > 10) {
            testResults.removeAt(testResults.size - 1)
        }
    }
    
    /**
     * Get test results history
     */
    fun getTestResults(): List<TestResult> = testResults.toList()
    
    /**
     * Clear test results
     */
    fun clearTestResults() {
        testResults.clear()
    }
    
    /**
     * Set image selection callback
     */
    fun setOnImageSelectedListener(listener: (Bitmap, String) -> Unit) {
        onImageSelectedListener = listener
    }
    
    /**
     * Set close callback
     */
    fun setOnCloseListener(listener: () -> Unit) {
        onCloseListener = listener
    }
    
    /**
     * Get current selected image
     */
    fun getCurrentImage(): Bitmap? = currentBitmap
    
    /**
     * Check if currently processing
     */
    fun isProcessingImage(): Boolean = isProcessing
    
    // Data classes
    data class SampleImage(
        val bitmap: Bitmap,
        val url: String,
        val title: String
    )
    
    data class TestResult(
        val source: String,
        val processingTime: Long,
        val detectionCount: Int,
        val confidence: Float,
        val imageSize: String,
        val timestamp: Long = System.currentTimeMillis()
    ) {
        fun getFormattedTime(): String {
            val seconds = (processingTime / 1000.0)
            return if (seconds >= 1) {
                "%.1fs".format(seconds)
            } else {
                "${processingTime}ms"
            }
        }
    }
    
    // RecyclerView Adapter for sample images
    private class TestImageAdapter(
        private var images: List<SampleImage>,
        private val onSampleImageClick: (String, Bitmap) -> Unit
    ) : RecyclerView.Adapter<TestImageAdapter.ViewHolder>() {
        
        inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
            val imageView: ImageView = itemView.findViewById(R.id.sampleImageView)
            val titleText: TextView = itemView.findViewById(R.id.sampleImageTitle)
            val cardView: MaterialCardView = itemView as MaterialCardView
        }
        
        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_sample_image, parent, false)
            return ViewHolder(view)
        }
        
        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            val image = images[position]
            holder.imageView.setImageBitmap(image.bitmap)
            holder.titleText.text = image.title
            
            holder.cardView.setOnClickListener {
                onSampleImageClick(image.url, image.bitmap)
            }
        }
        
        override fun getItemCount(): Int = images.size
        
        fun updateImages(newImages: List<SampleImage>) {
            images = newImages
            notifyDataSetChanged()
        }
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        // Clean up any pending coroutines
        lifecycleScope.coroutineContext.cancelChildren()
    }
}