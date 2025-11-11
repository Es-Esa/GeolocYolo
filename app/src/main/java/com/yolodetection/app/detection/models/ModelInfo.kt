package com.yolodetection.app.detection.models

/**
 * Model information and configuration
 */
data class ModelInfo(
    val name: String,
    val version: String,
    val inputSize: Int,
    val numClasses: Int,
    val hasNMS: Boolean, // Whether NMS is embedded in the model
    val confidenceThreshold: Float,
    val iouThreshold: Float,
    val description: String = "YOLOv11n Human Detection Model",
    val labels: List<String> = generateCOCOLabels(),
    val quantization: Quantization = Quantization.FP16
) {
    
    /**
     * Get the label for a class ID
     */
    fun getClassLabel(classId: Int): String {
        return if (classId in labels.indices) {
            labels[classId]
        } else {
            "Unknown"
        }
    }
    
    /**
     * Check if a class is a person class
     */
    fun isPersonClass(classId: Int): Boolean {
        return classId == 0 // In COCO, person is typically class 0
    }
    
    /**
     * Get person class ID
     */
    fun getPersonClassId(): Int = 0
    
    companion object {
        private fun generateCOCOLabels(): List<String> {
            return listOf(
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush"
            )
        }
    }
    
    /**
     * Model quantization types
     */
    enum class Quantization(val displayName: String, val fileSuffix: String) {
        FP32("Float32", "_fp32"),
        FP16("Float16", "_fp16"),
        INT8("Int8", "_int8");
        
        override fun toString(): String = displayName
    }
}