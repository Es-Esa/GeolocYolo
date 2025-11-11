package com.yolodetection.app.utils

/**
 * Application-wide constants and configuration
 */
object Constants {
    
    // Model configuration
    const val MODEL_INPUT_SIZE = 640  // YOLOv11n default input size
    const val CONFIDENCE_THRESHOLD = 0.5f  // Default confidence threshold
    const val IOU_THRESHOLD = 0.5f  // Default IoU threshold for NMS
    const val MAX_DETECTIONS = 300  // Maximum detections per frame
    
    // Model files
    const val MODEL_FP32_PATH = "models/yolov11n_fp32.tflite"
    const val MODEL_FP16_PATH = "models/yolov11n_fp16.tflite"
    const val MODEL_INT8_PATH = "models/yolov11n_int8.tflite"
    const val LABELS_PATH = "models/labels.txt"
    
    // Performance settings
    const val TARGET_FPS = 30  // Target frames per second
    const val MAX_QUEUE_SIZE = 3  // Maximum frame queue size
    const val PROCESSING_TIMEOUT_MS = 100L  // Processing timeout
    const val THREAD_COUNT = 4  // Number of inference threads
    
    // Camera settings
    const val CAMERA_PREVIEW_SIZE_WIDTH = 1280
    const val CAMERA_PREVIEW_SIZE_HEIGHT = 720
    const val CAMERA_IMAGE_SIZE_WIDTH = 640
    const val CAMERA_IMAGE_SIZE_HEIGHT = 640
    
    // Visualization settings
    const val BOUNDING_BOX_ALPHA = 220
    const val LABEL_BACKGROUND_ALPHA = 200
    const val LABEL_TEXT_SIZE = 32f
    const val LABEL_PADDING = 8f
    const val CORNER_RADIUS = 8f
    const val STROKE_WIDTH = 4f
    
    // Person detection specific
    const val PERSON_CLASS_ID = 0  // Person class in COCO dataset
    const val PERSON_CONFIDENCE_THRESHOLD = 0.3f  // Lower threshold for persons
    
    // Colors
    const val PERSON_COLOR = android.graphics.Color.GREEN
    const val OTHER_COLOR = android.graphics.Color.RED
    const val LABEL_TEXT_COLOR = android.graphics.Color.WHITE
    const val SHADOW_COLOR = android.graphics.Color.argb(100, 0, 0, 0)
    
    // Animation settings
    const val ANIMATION_DURATION_MS = 300L
    const val FADE_IN_DURATION_MS = 150L
    
    // Performance monitoring
    const val PERFORMANCE_SAMPLE_INTERVAL = 1000L  // Sample performance every second
    const val MAX_LOG_SIZE = 1000  // Maximum log entries to keep
    
    // File system
    const val CACHE_DIR = "cache"
    const val MODELS_DIR = "models"
    const val LOGS_DIR = "logs"
    const val TEMP_DIR = "temp"
    
    // API versions
    const val MIN_ANDROID_SDK = 24  // Android 7.0
    const val TARGET_ANDROID_SDK = 34  // Android 14
    const val COMPILE_ANDROID_SDK = 34
    
    // Delegates
    const val USE_CPU_DELEGATE = true
    const val USE_GPU_DELEGATE = true
    const val USE_NNAPI_DELEGATE = true
    
    // Memory management
    const val MAX_BITMAP_SIZE = 50 * 1024 * 1024  // 50MB
    const val MAX_INPUT_BUFFER_SIZE = 10 * 1024 * 1024  // 10MB
    const val TRIM_MEMORY_THRESHOLD = 100  // MB
    
    // Error codes
    const val ERROR_CODE_MODEL_LOAD_FAILED = 1001
    const val ERROR_CODE_CAMERA_INIT_FAILED = 1002
    const val ERROR_CODE_INFERENCE_FAILED = 1003
    const val ERROR_CODE_PERMISSION_DENIED = 1004
    const val ERROR_CODE_OUT_OF_MEMORY = 1005
    
    // Debug settings
    const val DEBUG_MODE = false
    const val VERBOSE_LOGGING = false
    const val SHOW_FPS_OVERLAY = true
    const val SHOW_PERFORMANCE_METRICS = true
    
    /**
     * Model quantization options
     */
    object Quantization {
        const val FP32 = "fp32"
        const val FP16 = "fp16"
        const val INT8 = "int8"
        
        fun getRecommended(): String = FP16  // Best balance of speed and accuracy
    }
    
    /**
     * Performance profiles
     */
    object PerformanceProfiles {
        const val LOW_POWER = "low_power"
        const val BALANCED = "balanced"
        const val HIGH_PERFORMANCE = "high_performance"
        
        fun getRecommended(): String = BALANCED
    }
    
    /**
     * Camera configurations
     */
    object CameraConfig {
        const val BACK_CAMERA = 0
        const val FRONT_CAMERA = 1
        const val EXTERNAL_CAMERA = 2
        
        const val ASPECT_RATIO_16_9 = "16:9"
        const val ASPECT_RATIO_4_3 = "4:3"
        const val ASPECT_RATIO_1_1 = "1:1"
    }
    
    /**
     * Supported model resolutions
     */
    object ModelResolutions {
        val SUPPORTED_SIZES = listOf(320, 416, 512, 640, 832, 1024)
        val RECOMMENDED = 640  // YOLOv11n default
    }
}