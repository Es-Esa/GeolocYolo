package com.yolodetection.app.detection.models

/**
 * Detection result from YOLO model
 */
data class Detection(
    val boundingBox: BoundingBox,
    val confidence: Float,
    val classId: Int,
    val className: String,
    val trackId: Int = -1, // For tracking across frames
    val landmark: PointF? = null // Optional landmark for specific classes
) {
    val center: PointF
        get() = PointF(
            (boundingBox.x1 + boundingBox.x2) / 2f,
            (boundingBox.y1 + boundingBox.y2) / 2f
        )
    
    val width: Float
        get() = boundingBox.x2 - boundingBox.x1
    
    val height: Float
        get() = boundingBox.y2 - boundingBox.y1
    
    val area: Float
        get() = width * height
}