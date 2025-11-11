package com.yolov11n.detection

import android.content.Context
import android.content.res.AssetManager
import android.os.Build
import android.util.Log
import org.json.JSONObject
import java.io.IOException

/**
 * Model Configuration and Optimization Settings
 * Centralizes all configuration parameters for YOLOv11n human detection
 * Implements optimization strategies based on mobile performance research
 */
class ModelConfig(
    private val context: Context,
    private val configFileName: String = "yolov11n_config.json"
) {
    private val TAG = "ModelConfig"
    
    // Model file configuration
    val modelFileName: String
    val labelsFileName: String
    val metadataFileName: String?
    
    // Model specifications
    val inputImageSize: Int
    val inputTensorFormat: InputTensorFormat
    val outputFormat: OutputFormat
    val expectedOutputShape: IntArray
    
    // Detection parameters
    val confidenceThreshold: Float
    val nmsThreshold: Float
    val humanClassId: Int
    val maxDetections: Int
    val applyNMS: Boolean
    
    // Performance optimization settings
    val preferredDelegate: TFLiteInterpreter.DelegateType
    val threadCount: Int
    val targetFPS: Int
    val frameSkippingEnabled: Boolean
    val adaptiveFrameSkipping: Boolean
    val enableMemoryOptimization: Boolean
    val enableThreadingOptimization: Boolean
    
    // Preprocessing options
    val keepAspectRatio: Boolean
    val convertToGrayscale: Boolean
    val normalizationMethod: NormalizationMethod
    val resizeMethod: ResizeMethod
    
    // Quantization settings
    val quantizationType: QuantizationType
    val enableQuantization: Boolean
    val calibrationDatasetPath: String?
    
    // Memory management
    val maxBufferPoolSize: Int
    val enableMemoryPooling: Boolean
    val maxHeapMemoryMB: Int
    
    // Device-specific optimizations
    val enableDynamicModelSelection: Boolean
    val devicePerformanceLevel: DevicePerformance
    val thermalThrottlingEnabled: Boolean
    val batteryAwareOptimization: Boolean
    
    // Debug and monitoring
    val enablePerformanceLogging: Boolean
    val enableDetailedProfiling: Boolean
    val debugOutputEnabled: Boolean
    
    // Hardware acceleration specific
    val gpuOptions: GPUOptions
    val nnapiOptions: NNAPIOptions
    
    init {
        val config = loadConfiguration()
        
        // Model files
        modelFileName = config.getString("model_file_name", "yolov11n_human.tflite")
        labelsFileName = config.getString("labels_file_name", "labels.txt")
        metadataFileName = config.optString("metadata_file_name", null)
        
        // Model specs
        inputImageSize = config.getInt("input_image_size", 640)
        inputTensorFormat = InputTensorFormat.fromString(
            config.getString("input_tensor_format", "NCHW")
        )
        outputFormat = OutputFormat.fromString(
            config.getString("output_format", "YOLO_DETECTIONS")
        )
        expectedOutputShape = config.getJSONArray("expected_output_shape")
            .let { jsonArray ->
                IntArray(jsonArray.length()) { jsonArray.getInt(it) }
            }
        
        // Detection parameters
        confidenceThreshold = config.getDouble("confidence_threshold", 0.5).toFloat()
        nmsThreshold = config.getDouble("nms_threshold", 0.4).toFloat()
        humanClassId = config.getInt("human_class_id", 0)
        maxDetections = config.getInt("max_detections", 100)
        applyNMS = config.getBoolean("apply_nms", true)
        
        // Performance optimization
        preferredDelegate = TFLiteInterpreter.DelegateType.fromString(
            config.getString("preferred_delegate", getDefaultDelegate())
        )
        threadCount = config.getInt("thread_count", getOptimalThreadCount())
        targetFPS = config.getInt("target_fps", 30)
        frameSkippingEnabled = config.getBoolean("frame_skipping_enabled", true)
        adaptiveFrameSkipping = config.getBoolean("adaptive_frame_skipping", true)
        enableMemoryOptimization = config.getBoolean("enable_memory_optimization", true)
        enableThreadingOptimization = config.getBoolean("enable_threading_optimization", true)
        
        // Preprocessing
        keepAspectRatio = config.getBoolean("keep_aspect_ratio", true)
        convertToGrayscale = config.getBoolean("convert_to_grayscale", true)
        normalizationMethod = NormalizationMethod.fromString(
            config.getString("normalization_method", "DIVIDE_BY_255")
        )
        resizeMethod = ResizeMethod.fromString(
            config.getString("resize_method", "NEAREST_NEIGHBOR")
        )
        
        // Quantization
        quantizationType = QuantizationType.fromString(
            config.getString("quantization_type", "FP16")
        )
        enableQuantization = config.getBoolean("enable_quantization", true)
        calibrationDatasetPath = config.optString("calibration_dataset_path", null)
        
        // Memory management
        maxBufferPoolSize = config.getInt("max_buffer_pool_size", 5)
        enableMemoryPooling = config.getBoolean("enable_memory_pooling", true)
        maxHeapMemoryMB = config.getInt("max_heap_memory_mb", 256)
        
        // Device-specific
        enableDynamicModelSelection = config.getBoolean("enable_dynamic_model_selection", true)
        devicePerformanceLevel = detectDevicePerformance()
        thermalThrottlingEnabled = config.getBoolean("thermal_throttling_enabled", true)
        batteryAwareOptimization = config.getBoolean("battery_aware_optimization", true)
        
        // Debug
        enablePerformanceLogging = config.getBoolean("enable_performance_logging", true)
        enableDetailedProfiling = config.getBoolean("enable_detailed_profiling", false)
        debugOutputEnabled = config.getBoolean("debug_output_enabled", false)
        
        // Hardware options
        gpuOptions = GPUOptions.fromConfig(config.getJSONObject("gpu_options"))
        nnapiOptions = NNAPIOptions.fromConfig(config.getJSONObject("nnapi_options"))
        
        Log.i(TAG, "Model configuration loaded successfully")
        Log.d(TAG, "Config summary: InputSize=$inputImageSize, Delegate=$preferredDelegate, TargetFPS=$targetFPS")
    }
    
    /**
     * Load configuration from JSON file in assets
     */
    private fun loadConfiguration(): JSONObject {
        return try {
            val inputStream = context.assets.open(configFileName)
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            JSONObject(jsonString)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load config from $configFileName, using defaults", e)
            getDefaultConfiguration()
        }
    }
    
    /**
     * Get default delegate based on device capabilities
     */
    private fun getDefaultDelegate(): String {
        return when {
            // High-end devices with good GPU support
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1 && 
            Build.HARDWARE.contains("adreno") -> "GPU"
            
            // Devices with NNAPI support
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1 -> "NNAPI"
            
            // Fallback to CPU
            else -> "CPU"
        }
    }
    
    /**
     * Get optimal thread count based on device cores
     */
    private fun getOptimalThreadCount(): Int {
        return Runtime.getRuntime().availableProcessors().coerceIn(1, 8)
    }
    
    /**
     * Detect device performance level
     */
    private fun detectDevicePerformance(): DevicePerformance {
        val cores = Runtime.getRuntime().availableProcessors()
        val maxMemory = Runtime.getRuntime().maxMemory() / (1024 * 1024) // MB
        
        return when {
            cores >= 8 && maxMemory > 100 -> DevicePerformance.HIGH
            cores >= 4 && maxMemory > 50 -> DevicePerformance.MEDIUM
            else -> DevicePerformance.LOW
        }
    }
    
    /**
     * Get default configuration when file loading fails
     */
    private fun getDefaultConfiguration(): JSONObject {
        return JSONObject().apply {
            put("model_file_name", "yolov11n_human.tflite")
            put("labels_file_name", "labels.txt")
            put("input_image_size", 640)
            put("input_tensor_format", "NCHW")
            put("output_format", "YOLO_DETECTIONS")
            put("expected_output_shape", org.json.JSONArray("[1, 300, 6]"))
            put("confidence_threshold", 0.5)
            put("nms_threshold", 0.4)
            put("human_class_id", 0)
            put("max_detections", 100)
            put("apply_nms", true)
            put("preferred_delegate", getDefaultDelegate())
            put("thread_count", getOptimalThreadCount())
            put("target_fps", 30)
            put("frame_skipping_enabled", true)
            put("adaptive_frame_skipping", true)
            put("enable_memory_optimization", true)
            put("enable_threading_optimization", true)
            put("keep_aspect_ratio", true)
            put("convert_to_grayscale", true)
            put("normalization_method", "DIVIDE_BY_255")
            put("resize_method", "NEAREST_NEIGHBOR")
            put("quantization_type", "FP16")
            put("enable_quantization", true)
            put("max_buffer_pool_size", 5)
            put("enable_memory_pooling", true)
            put("max_heap_memory_mb", 256)
            put("enable_dynamic_model_selection", true)
            put("thermal_throttling_enabled", true)
            put("battery_aware_optimization", true)
            put("enable_performance_logging", true)
            put("enable_detailed_profiling", false)
            put("debug_output_enabled", false)
            put("gpu_options", JSONObject().apply {
                put("precision_loss_allowed", false)
                put("inference_preference", "FAST_SINGLE_ANSWER")
                put("enable_serialization", true)
                put("max_delegates", 1)
            })
            put("nnapi_options", JSONObject().apply {
                put("execution_preference", "SUSTAINED_SPEED")
                put("enable_burst_execution", true)
                put("enable_memory_domains", true)
                put("enable_qos_hints", true)
            })
        }
    }
    
    /**
     * Update configuration at runtime
     */
    fun updateConfiguration(updates: Map<String, Any>) {
        updates.forEach { (key, value) ->
            when (key) {
                "confidence_threshold" -> if (value is Number) confidenceThreshold
                "nms_threshold" -> if (value is Number) nmsThreshold
                "target_fps" -> if (value is Number) targetFPS
                "preferred_delegate" -> if (value is String) preferredDelegate
                "thread_count" -> if (value is Number) threadCount
                else -> Log.d(TAG, "Runtime config update not implemented for: $key")
            }
        }
    }
    
    /**
     * Get configuration summary for logging
     */
    fun getConfigurationSummary(): String {
        return """
            Model: $modelFileName
            Input Size: ${inputImageSize}x${inputImageSize}
            Delegate: $preferredDelegate (${getDefaultDelegate()})
            Thread Count: $threadCount
            Target FPS: $targetFPS
            Quantization: $quantizationType
            Device Performance: $devicePerformanceLevel
            Memory Optimization: $enableMemoryOptimization
            Frame Skipping: $frameSkippingEnabled
        """.trimIndent()
    }
    
    /**
     * Check if device supports specific optimization
     */
    fun supportsOptimization(optimization: String): Boolean {
        return when (optimization) {
            "gpu" -> Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1 && 
                    (Build.HARDWARE.contains("adreno") || Build.HARDWARE.contains("mali"))
            "nnapi" -> Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1
            "hexagon" -> Build.HARDWARE.contains("qcom")
            "advanced_memory" -> Build.VERSION.SDK_INT >= Build.VERSION_CODES.P
            else -> false
        }
    }
}

/**
 * Supporting enums and data classes
 */
enum class InputTensorFormat(val description: String) {
    NCHW("Channels First"),
    NHWC("Channels Last"),
    NCDHW("3D Channels First"),
    NDHWC("3D Channels Last");
    
    companion object {
        fun fromString(format: String): InputTensorFormat {
            return values().find { it.name.equals(format, ignoreCase = true) } ?: NCHW
        }
    }
}

enum class OutputFormat(val description: String) {
    YOLO_DETECTIONS("YOLO Detection Format"),
    COCO_DETECTIONS("COCO Detection Format"),
    RAW_TENSORS("Raw Tensor Output");
    
    companion object {
        fun fromString(format: String): OutputFormat {
            return values().find { it.name.equals(format, ignoreCase = true) } ?: YOLO_DETECTIONS
        }
    }
}

enum class NormalizationMethod(val description: String) {
    DIVIDE_BY_255("Divide by 255.0"),
    STANDARDIZATION("Mean=0, Std=1"),
    MIN_MAX("Min-Max Normalization"),
    RAW("No Normalization");
    
    companion object {
        fun fromString(method: String): NormalizationMethod {
            return values().find { it.name.equals(method, ignoreCase = true) } ?: DIVIDE_BY_255
        }
    }
}

enum class ResizeMethod(val description: String) {
    NEAREST_NEIGHBOR("Nearest Neighbor"),
    BILINEAR("Bilinear Interpolation"),
    BICUBIC("Bicubic Interpolation"),
    LANCZOS("Lanczos Resampling");
    
    companion object {
        fun fromString(method: String): ResizeMethod {
            return values().find { it.name.equals(method, ignoreCase = true) } ?: NEAREST_NEIGHBOR
        }
    }
}

enum class QuantizationType(val description: String) {
    FP32("32-bit Floating Point"),
    FP16("16-bit Floating Point"),
    INT8("8-bit Integer"),
    DYNAMIC("Dynamic Quantization");
    
    companion object {
        fun fromString(type: String): QuantizationType {
            return values().find { it.name.equals(type, ignoreCase = true) } ?: FP16
        }
    }
}

data class GPUOptions(
    val precisionLossAllowed: Boolean,
    val inferencePreference: String,
    val enableSerialization: Boolean,
    val maxDelegates: Int,
    val enableBackendGpu: Boolean = true
) {
    companion object {
        fun fromConfig(config: JSONObject): GPUOptions {
            return GPUOptions(
                precisionLossAllowed = config.optBoolean("precision_loss_allowed", false),
                inferencePreference = config.getString("inference_preference", "FAST_SINGLE_ANSWER"),
                enableSerialization = config.optBoolean("enable_serialization", true),
                maxDelegates = config.optInt("max_delegates", 1)
            )
        }
    }
}

data class NNAPIOptions(
    val executionPreference: String,
    val enableBurstExecution: Boolean,
    val enableMemoryDomains: Boolean,
    val enableQosHints: Boolean,
    val enableFencedExecution: Boolean = true
) {
    companion object {
        fun fromConfig(config: JSONObject): NNAPIOptions {
            return NNAPIOptions(
                executionPreference = config.getString("execution_preference", "SUSTAINED_SPEED"),
                enableBurstExecution = config.optBoolean("enable_burst_execution", true),
                enableMemoryDomains = config.optBoolean("enable_memory_domains", true),
                enableQosHints = config.optBoolean("enable_qos_hints", true)
            )
        }
    }
}

enum class DevicePerformance(val description: String, val recommendedSettings: String) {
    HIGH("High Performance", "640x640, GPU delegate, 4 threads"),
    MEDIUM("Medium Performance", "480x480, NNAPI/CPU, 2 threads"),
    LOW("Low Performance", "320x320, CPU only, 1 thread")
}