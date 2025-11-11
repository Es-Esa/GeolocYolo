package com.yolodetection.app.ui

import android.content.SharedPreferences
import android.os.Bundle
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import com.google.android.material.switchmaterial.SwitchMaterial
import com.yolodetection.app.R
import timber.log.Timber

/**
 * Settings Activity for configuring YOLO detection parameters
 */
class SettingsActivity : AppCompatActivity() {
    
    private lateinit var sharedPreferences: SharedPreferences
    
    // UI Components
    private lateinit var confidenceSeekBar: SeekBar
    private lateinit var confidenceValueText: TextView
    private lateinit var iouSeekBar: SeekBar
    private lateinit var iouValueText: TextView
    private lateinit var maxDetectionsSeekBar: SeekBar
    private lateinit var maxDetectionsValueText: TextView
    private lateinit var inputSizeSeekBar: SeekBar
    private lateinit var inputSizeValueText: TextView
    private lateinit var personOnlySwitch: SwitchMaterial
    private lateinit var showConfidenceSwitch: SwitchMaterial
    private lateinit var showLabelsSwitch: SwitchMaterial
    private lateinit var highPerformanceSwitch: SwitchMaterial
    private lateinit var batterySaverSwitch: SwitchMaterial
    
    // Input size options
    private val inputSizes = intArrayOf(320, 416, 512, 640, 832, 1024)
    private val inputSizeLabels = inputSizes.map { "${it}x${it}" }.toTypedArray()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        
        // Setup toolbar
        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Detection Settings"
        
        // Get shared preferences
        sharedPreferences = getSharedPreferences("yolo_settings", MODE_PRIVATE)
        
        // Initialize UI components
        initializeViews()
        setupSeekBarListeners()
        setupSwitchListeners()
        loadSettings()
    }
    
    private fun initializeViews() {
        confidenceSeekBar = findViewById(R.id.confidenceSeekBar)
        confidenceValueText = findViewById(R.id.confidenceValue)
        iouSeekBar = findViewById(R.id.iouSeekBar)
        iouValueText = findViewById(R.id.iouValue)
        maxDetectionsSeekBar = findViewById(R.id.maxDetectionsSeekBar)
        maxDetectionsValueText = findViewById(R.id.maxDetectionsValue)
        inputSizeSeekBar = findViewById(R.id.inputSizeSeekBar)
        inputSizeValueText = findViewById(R.id.inputSizeValue)
        personOnlySwitch = findViewById(R.id.personOnlySwitch)
        showConfidenceSwitch = findViewById(R.id.showConfidenceSwitch)
        showLabelsSwitch = findViewById(R.id.showLabelsSwitch)
        highPerformanceSwitch = findViewById(R.id.highPerformanceSwitch)
        batterySaverSwitch = findViewById(R.id.batterySaverSwitch)
    }
    
    private fun setupSeekBarListeners() {
        confidenceSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    val confidence = progress / 100.0f
                    confidenceValueText.text = "%.2f".format(confidence)
                    saveSetting("confidence_threshold", confidence)
                }
            }
            
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        iouSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    val iou = progress / 100.0f
                    iouValueText.text = "%.2f".format(iou)
                    saveSetting("iou_threshold", iou)
                }
            }
            
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        maxDetectionsSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    val maxDetections = 10 + progress * 10 // 10 to 300
                    maxDetectionsValueText.text = maxDetections.toString()
                    saveSetting("max_detections", maxDetections)
                }
            }
            
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        inputSizeSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    val size = inputSizes[progress]
                    inputSizeValueText.text = "${size}x${size}"
                    saveSetting("input_size", size)
                }
            }
            
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }
    
    private fun setupSwitchListeners() {
        personOnlySwitch.setOnCheckedChangeListener { _, isChecked ->
            saveSetting("person_only", isChecked)
        }
        
        showConfidenceSwitch.setOnCheckedChangeListener { _, isChecked ->
            saveSetting("show_confidence", isChecked)
        }
        
        showLabelsSwitch.setOnCheckedChangeListener { _, isChecked ->
            saveSetting("show_labels", isChecked)
        }
        
        highPerformanceSwitch.setOnCheckedChangeListener { _, isChecked ->
            saveSetting("high_performance", isChecked)
            if (isChecked) {
                batterySaverSwitch.isChecked = false
                saveSetting("battery_saver", false)
            }
        }
        
        batterySaverSwitch.setOnCheckedChangeListener { _, isChecked ->
            saveSetting("battery_saver", isChecked)
            if (isChecked) {
                highPerformanceSwitch.isChecked = false
                saveSetting("high_performance", false)
            }
        }
    }
    
    private fun loadSettings() {
        // Load confidence threshold
        val confidence = sharedPreferences.getFloat("confidence_threshold", 0.5f)
        val confidenceProgress = (confidence * 100).toInt()
        confidenceSeekBar.progress = confidenceProgress
        confidenceValueText.text = "%.2f".format(confidence)
        
        // Load IoU threshold
        val iou = sharedPreferences.getFloat("iou_threshold", 0.5f)
        val iouProgress = (iou * 100).toInt()
        iouSeekBar.progress = iouProgress
        iouValueText.text = "%.2f".format(iou)
        
        // Load max detections
        val maxDetections = sharedPreferences.getInt("max_detections", 100)
        val maxDetectionsProgress = (maxDetections - 10) / 10
        maxDetectionsSeekBar.progress = maxDetectionsProgress
        maxDetectionsValueText.text = maxDetections.toString()
        
        // Load input size
        val inputSize = sharedPreferences.getInt("input_size", 640)
        val inputSizeIndex = inputSizes.indexOf(inputSize).coerceAtLeast(0)
        inputSizeSeekBar.progress = inputSizeIndex
        inputSizeValueText.text = "${inputSize}x${inputSize}"
        
        // Load boolean settings
        personOnlySwitch.isChecked = sharedPreferences.getBoolean("person_only", true)
        showConfidenceSwitch.isChecked = sharedPreferences.getBoolean("show_confidence", true)
        showLabelsSwitch.isChecked = sharedPreferences.getBoolean("show_labels", true)
        highPerformanceSwitch.isChecked = sharedPreferences.getBoolean("high_performance", false)
        batterySaverSwitch.isChecked = sharedPreferences.getBoolean("battery_saver", false)
        
        Timber.i("Settings loaded from shared preferences")
    }
    
    private fun saveSetting(key: String, value: Any) {
        val editor = sharedPreferences.edit()
        when (value) {
            is Float -> editor.putFloat(key, value)
            is Int -> editor.putInt(key, value)
            is Boolean -> editor.putBoolean(key, value)
            is String -> editor.putString(key, value)
        }
        editor.apply()
        Timber.d("Setting saved: $key = $value")
    }
    
    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Timber.i("Settings activity destroyed")
    }
}