package com.yolodetection.ui

import android.content.SharedPreferences
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.widget.SwitchCompat
import androidx.fragment.app.Fragment
import androidx.preference.PreferenceManager
import com.google.android.material.card.MaterialCardView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.slider.Slider

/**
 * Settings fragment for configuring detection parameters and performance options
 * Provides intuitive controls for threshold adjustment and optimization settings
 */
class SettingsFragment : Fragment() {
    
    companion object {
        private const val TAG = "SettingsFragment"
        
        // Settings keys
        private const val KEY_CONFIDENCE_THRESHOLD = "confidence_threshold"
        private const val KEY_IOU_THRESHOLD = "iou_threshold"
        private const val KEY_MAX_DETECTIONS = "max_detections"
        private const val KEY_ENABLE_TRACKING = "enable_tracking"
        private const val KEY_ENABLE_DEBUG_MODE = "enable_debug_mode"
        private const val KEY_ENABLE_LANDMARKS = "enable_landmarks"
        private const val KEY_MODEL_PRECISION = "model_precision"
        private const val KEY_PROCESSING_FPS = "processing_fps"
        private const val KEY_ENABLE_GPU_ACCELERATION = "enable_gpu_acceleration"
        private const val KEY_ENABLE_IMAGE_PREPROCESSING = "enable_image_preprocessing"
        
        // Default values
        private const val DEFAULT_CONFIDENCE_THRESHOLD = 0.5f
        private const val DEFAULT_IOU_THRESHOLD = 0.45f
        private const val DEFAULT_MAX_DETECTIONS = 10
        private const val DEFAULT_PROCESSING_FPS = 30
        
        // UI update delay for smooth interactions
        private const val UI_UPDATE_DELAY = 100L
    }
    
    // UI Components
    private var confidenceSlider: Slider? = null
    private var confidenceValue: TextView? = null
    private var iouSlider: Slider? = null
    private var iouValue: TextView? = null
    private var maxDetectionsSlider: Slider? = null
    private var maxDetectionsValue: TextView? = null
    private var processingFpsSlider: Slider? = null
    private var processingFpsValue: TextView? = null
    
    private var trackingSwitch: SwitchCompat? = null
    private var debugSwitch: SwitchCompat? = null
    private var landmarksSwitch: SwitchCompat? = null
    private var gpuSwitch: SwitchCompat? = null
    private var preprocessingSwitch: SwitchCompat? = null
    
    private var modelPrecisionSpinner: TextView? = null
    private var closeButton: FloatingActionButton? = null
    
    // Settings change listener
    private var onSettingsChangedListener: ((SettingsData) -> Unit)? = null
    
    // Shared preferences
    private var sharedPreferences: SharedPreferences? = null
    
    // Data class for settings
    data class SettingsData(
        val confidenceThreshold: Float,
        val iouThreshold: Float,
        val maxDetections: Int,
        val enableTracking: Boolean,
        val enableDebugMode: Boolean,
        val enableLandmarks: Boolean,
        val modelPrecision: String,
        val processingFps: Int,
        val enableGpuAcceleration: Boolean,
        val enableImagePreprocessing: Boolean
    )
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_settings, container, false)
        
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(requireContext())
        initializeViews(view)
        loadSettings()
        setupListeners()
        
        return view
    }
    
    /**
     * Initialize all view references
     */
    private fun initializeViews(view: View) {
        // Sliders and their value displays
        confidenceSlider = view.findViewById(R.id.confidenceSlider)
        confidenceValue = view.findViewById(R.id.confidenceValue)
        
        iouSlider = view.findViewById(R.id.iouSlider)
        iouValue = view.findViewById(R.id.iouValue)
        
        maxDetectionsSlider = view.findViewById(R.id.maxDetectionsSlider)
        maxDetectionsValue = view.findViewById(R.id.maxDetectionsValue)
        
        processingFpsSlider = view.findViewById(R.id.processingFpsSlider)
        processingFpsValue = view.findViewById(R.id.processingFpsValue)
        
        // Switches
        trackingSwitch = view.findViewById(R.id.trackingSwitch)
        debugSwitch = view.findViewById(R.id.debugSwitch)
        landmarksSwitch = view.findViewById(R.id.landmarksSwitch)
        gpuSwitch = view.findViewById(R.id.gpuSwitch)
        preprocessingSwitch = view.findViewById(R.id.preprocessingSwitch)
        
        // Other controls
        modelPrecisionSpinner = view.findViewById(R.id.modelPrecisionText)
        closeButton = view.findViewById(R.id.closeButton)
        
        // Configure slider ranges and steps
        setupSliderRanges()
    }
    
    /**
     * Configure slider ranges and step values
     */
    private fun setupSliderRanges() {
        // Confidence threshold: 0.1 to 1.0 with 0.05 step
        confidenceSlider?.apply {
            valueFrom = 0.1f
            valueTo = 1.0f
            stepSize = 0.05f
        }
        
        // IoU threshold: 0.3 to 0.8 with 0.05 step
        iouSlider?.apply {
            valueFrom = 0.3f
            valueTo = 0.8f
            stepSize = 0.05f
        }
        
        // Max detections: 1 to 50
        maxDetectionsSlider?.apply {
            valueFrom = 1f
            valueTo = 50f
            stepSize = 1f
        }
        
        // Processing FPS: 10 to 60
        processingFpsSlider?.apply {
            valueFrom = 10f
            valueTo = 60f
            stepSize = 5f
        }
    }
    
    /**
     * Load settings from SharedPreferences
     */
    private fun loadSettings() {
        sharedPreferences?.let { prefs ->
            // Load slider values
            val confidence = prefs.getFloat(KEY_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD)
            val iou = prefs.getFloat(KEY_IOU_THRESHOLD, DEFAULT_IOU_THRESHOLD)
            val maxDetections = prefs.getInt(KEY_MAX_DETECTIONS, DEFAULT_MAX_DETECTIONS)
            val processingFps = prefs.getInt(KEY_PROCESSING_FPS, DEFAULT_PROCESSING_FPS)
            
            // Update UI with loaded values
            confidenceSlider?.value = confidence
            iouSlider?.value = iou
            maxDetectionsSlider?.value = maxDetections.toFloat()
            processingFpsSlider?.value = processingFps.toFloat()
            
            // Update value displays
            updateValueDisplays()
            
            // Load switch states
            trackingSwitch?.isChecked = prefs.getBoolean(KEY_ENABLE_TRACKING, true)
            debugSwitch?.isChecked = prefs.getBoolean(KEY_ENABLE_DEBUG_MODE, false)
            landmarksSwitch?.isChecked = prefs.getBoolean(KEY_ENABLE_LANDMARKS, false)
            gpuSwitch?.isChecked = prefs.getBoolean(KEY_ENABLE_GPU_ACCELERATION, true)
            preprocessingSwitch?.isChecked = prefs.getBoolean(KEY_ENABLE_IMAGE_PREPROCESSING, true)
            
            // Load model precision
            val precision = prefs.getString(KEY_MODEL_PRECISION, "FP32")
            modelPrecisionSpinner?.text = precision
        }
    }
    
    /**
     * Setup click listeners and change listeners
     */
    private fun setupListeners() {
        // Slider change listeners with debouncing
        val debounceHandler = android.os.Handler(requireContext().mainLooper)
        
        confidenceSlider?.addOnChangeListener { _, value, _ ->
            debounceHandler.removeCallbacksAndMessages(null)
            debounceHandler.postDelayed({
                updateValueDisplays()
                saveSetting(KEY_CONFIDENCE_THRESHOLD, value)
                notifySettingsChanged()
            }, UI_UPDATE_DELAY)
        }
        
        iouSlider?.addOnChangeListener { _, value, _ ->
            debounceHandler.removeCallbacksAndMessages(null)
            debounceHandler.postDelayed({
                updateValueDisplays()
                saveSetting(KEY_IOU_THRESHOLD, value)
                notifySettingsChanged()
            }, UI_UPDATE_DELAY)
        }
        
        maxDetectionsSlider?.addOnChangeListener { _, value, _ ->
            debounceHandler.removeCallbacksAndMessages(null)
            debounceHandler.postDelayed({
                updateValueDisplays()
                saveSetting(KEY_MAX_DETECTIONS, value.toInt())
                notifySettingsChanged()
            }, UI_UPDATE_DELAY)
        }
        
        processingFpsSlider?.addOnChangeListener { _, value, _ ->
            debounceHandler.removeCallbacksAndMessages(null)
            debounceHandler.postDelayed({
                updateValueDisplays()
                saveSetting(KEY_PROCESSING_FPS, value.toInt())
                notifySettingsChanged()
            }, UI_UPDATE_DELAY)
        }
        
        // Switch listeners
        trackingSwitch?.setOnCheckedChangeListener { _, isChecked ->
            saveSetting(KEY_ENABLE_TRACKING, isChecked)
            notifySettingsChanged()
        }
        
        debugSwitch?.setOnCheckedChangeListener { _, isChecked ->
            saveSetting(KEY_ENABLE_DEBUG_MODE, isChecked)
            notifySettingsChanged()
        }
        
        landmarksSwitch?.setOnCheckedChangeListener { _, isChecked ->
            saveSetting(KEY_ENABLE_LANDMARKS, isChecked)
            notifySettingsChanged()
        }
        
        gpuSwitch?.setOnCheckedChangeListener { _, isChecked ->
            saveSetting(KEY_ENABLE_GPU_ACCELERATION, isChecked)
            notifySettingsChanged()
        }
        
        preprocessingSwitch?.setOnCheckedChangeListener { _, isChecked ->
            saveSetting(KEY_ENABLE_IMAGE_PREPROCESSING, isChecked)
            notifySettingsChanged()
        }
        
        // Model precision spinner
        modelPrecisionSpinner?.setOnClickListener {
            showModelPrecisionDialog()
        }
        
        // Close button
        closeButton?.setOnClickListener {
            parentFragmentManager.popBackStack()
        }
        
        // Reset to defaults button (if exists)
        view?.findViewById<View>(R.id.resetToDefaultsButton)?.setOnClickListener {
            resetToDefaults()
        }
    }
    
    /**
     * Update all value displays with current slider values
     */
    private fun updateValueDisplays() {
        confidenceValue?.text = "%.2f".format(confidenceSlider?.value ?: DEFAULT_CONFIDENCE_THRESHOLD)
        iouValue?.text = "%.2f".format(iouSlider?.value ?: DEFAULT_IOU_THRESHOLD)
        maxDetectionsValue?.text = "${maxDetectionsSlider?.value?.toInt() ?: DEFAULT_MAX_DETECTIONS}"
        processingFpsValue?.text = "${processingFpsSlider?.value?.toInt() ?: DEFAULT_PROCESSING_FPS} FPS"
    }
    
    /**
     * Show dialog for model precision selection
     */
    private fun showModelPrecisionDialog() {
        val options = arrayOf("FP32", "FP16", "INT8")
        val currentSelection = sharedPreferences?.getString(KEY_MODEL_PRECISION, "FP32") ?: "FP32"
        val selectedIndex = options.indexOf(currentSelection)
        
        androidx.appcompat.app.AlertDialog.Builder(requireContext())
            .setTitle("Model Precision")
            .setSingleChoiceItems(options, selectedIndex) { dialog, which ->
                val selectedPrecision = options[which]
                modelPrecisionSpinner?.text = selectedPrecision
                saveSetting(KEY_MODEL_PRECISION, selectedPrecision)
                notifySettingsChanged()
                dialog.dismiss()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
    
    /**
     * Save individual setting to SharedPreferences
     */
    private fun saveSetting(key: String, value: Any) {
        sharedPreferences?.edit()?.let { editor ->
            when (value) {
                is Float -> editor.putFloat(key, value)
                is Int -> editor.putInt(key, value)
                is Boolean -> editor.putBoolean(key, value)
                is String -> editor.putString(key, value)
            }
            editor.apply()
        }
    }
    
    /**
     * Reset all settings to default values
     */
    private fun resetToDefaults() {
        sharedPreferences?.edit()?.let { editor ->
            editor.putFloat(KEY_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD)
            editor.putFloat(KEY_IOU_THRESHOLD, DEFAULT_IOU_THRESHOLD)
            editor.putInt(KEY_MAX_DETECTIONS, DEFAULT_MAX_DETECTIONS)
            editor.putInt(KEY_PROCESSING_FPS, DEFAULT_PROCESSING_FPS)
            editor.putBoolean(KEY_ENABLE_TRACKING, true)
            editor.putBoolean(KEY_ENABLE_DEBUG_MODE, false)
            editor.putBoolean(KEY_ENABLE_LANDMARKS, false)
            editor.putBoolean(KEY_ENABLE_GPU_ACCELERATION, true)
            editor.putBoolean(KEY_ENABLE_IMAGE_PREPROCESSING, true)
            editor.putString(KEY_MODEL_PRECISION, "FP32")
            editor.apply()
        }
        
        // Reload the UI
        loadSettings()
        notifySettingsChanged()
    }
    
    /**
     * Get current settings as SettingsData object
     */
    fun getCurrentSettings(): SettingsData? {
        sharedPreferences?.let { prefs ->
            return SettingsData(
                confidenceThreshold = prefs.getFloat(KEY_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD),
                iouThreshold = prefs.getFloat(KEY_IOU_THRESHOLD, DEFAULT_IOU_THRESHOLD),
                maxDetections = prefs.getInt(KEY_MAX_DETECTIONS, DEFAULT_MAX_DETECTIONS),
                enableTracking = prefs.getBoolean(KEY_ENABLE_TRACKING, true),
                enableDebugMode = prefs.getBoolean(KEY_ENABLE_DEBUG_MODE, false),
                enableLandmarks = prefs.getBoolean(KEY_ENABLE_LANDMARKS, false),
                modelPrecision = prefs.getString(KEY_MODEL_PRECISION, "FP32") ?: "FP32",
                processingFps = prefs.getInt(KEY_PROCESSING_FPS, DEFAULT_PROCESSING_FPS),
                enableGpuAcceleration = prefs.getBoolean(KEY_ENABLE_GPU_ACCELERATION, true),
                enableImagePreprocessing = prefs.getBoolean(KEY_ENABLE_IMAGE_PREPROCESSING, true)
            )
        }
        return null
    }
    
    /**
     * Notify listeners about settings changes
     */
    private fun notifySettingsChanged() {
        getCurrentSettings()?.let { settings ->
            onSettingsChangedListener?.invoke(settings)
        }
    }
    
    /**
     * Set listener for settings changes
     */
    fun setOnSettingsChangedListener(listener: (SettingsData) -> Unit) {
        onSettingsChangedListener = listener
    }
    
    /**
     * Apply settings programmatically
     */
    fun applySettings(settings: SettingsData) {
        confidenceSlider?.value = settings.confidenceThreshold
        iouSlider?.value = settings.iouThreshold
        maxDetectionsSlider?.value = settings.maxDetections.toFloat()
        processingFpsSlider?.value = settings.processingFps.toFloat()
        trackingSwitch?.isChecked = settings.enableTracking
        debugSwitch?.isChecked = settings.enableDebugMode
        landmarksSwitch?.isChecked = settings.enableLandmarks
        gpuSwitch?.isChecked = settings.enableGpuAcceleration
        preprocessingSwitch?.isChecked = settings.enableImagePreprocessing
        modelPrecisionSpinner?.text = settings.modelPrecision
        
        // Save to shared preferences
        sharedPreferences?.edit()?.let { editor ->
            editor.putFloat(KEY_CONFIDENCE_THRESHOLD, settings.confidenceThreshold)
            editor.putFloat(KEY_IOU_THRESHOLD, settings.iouThreshold)
            editor.putInt(KEY_MAX_DETECTIONS, settings.maxDetections)
            editor.putInt(KEY_PROCESSING_FPS, settings.processingFps)
            editor.putBoolean(KEY_ENABLE_TRACKING, settings.enableTracking)
            editor.putBoolean(KEY_ENABLE_DEBUG_MODE, settings.enableDebugMode)
            editor.putBoolean(KEY_ENABLE_LANDMARKS, settings.enableLandmarks)
            editor.putBoolean(KEY_ENABLE_GPU_ACCELERATION, settings.enableGpuAcceleration)
            editor.putBoolean(KEY_ENABLE_IMAGE_PREPROCESSING, settings.enableImagePreprocessing)
            editor.putString(KEY_MODEL_PRECISION, settings.modelPrecision)
            editor.apply()
        }
        
        updateValueDisplays()
        notifySettingsChanged()
    }
    
    /**
     * Get performance impact description for a setting
     */
    fun getPerformanceImpactDescription(): String {
        val currentSettings = getCurrentSettings() ?: return "No settings available"
        
        val impactFactors = mutableListOf<String>()
        
        if (currentSettings.confidenceThreshold < 0.5f) {
            impactFactors.add("More false positives")
        }
        if (currentSettings.maxDetections > 20) {
            impactFactors.add("Higher processing load")
        }
        if (currentSettings.processingFps > 30) {
            impactFactors.add("Higher battery usage")
        }
        if (!currentSettings.enableGpuAcceleration) {
            impactFactors.add("CPU-only processing")
        }
        if (currentSettings.enableLandmarks) {
            impactFactors.add("Additional processing for landmarks")
        }
        
        return if (impactFactors.isEmpty()) {
            "Balanced performance settings"
        } else {
            "Performance factors: ${impactFactors.joinToString(", ")}"
        }
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        onSettingsChangedListener = null
    }
}