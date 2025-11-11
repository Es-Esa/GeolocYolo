package com.yolodetection.app.detection.models

import android.graphics.RectF

/**
 * Bounding box representation for object detection
 */
data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
) {
    
    /**
     * Convert to RectF for drawing
     */
    fun toRectF(): RectF = RectF(x1, y1, x2, y2)
    
    /**
     * Calculate intersection with another bounding box
     */
    fun intersection(other: BoundingBox): BoundingBox {
        return BoundingBox(
            x1 = maxOf(this.x1, other.x1),
            y1 = maxOf(this.y1, other.y1),
            x2 = minOf(this.x2, other.x2),
            y2 = minOf(this.y2, other.y2)
        )
    }
    
    /**
     * Calculate union with another bounding box
     */
    fun union(other: BoundingBox): BoundingBox {
        return BoundingBox(
            x1 = minOf(this.x1, other.x1),
            y1 = minOf(this.y1, other.y1),
            x2 = maxOf(this.x2, other.x2),
            y2 = maxOf(this.y2, other.y2)
        )
    }
    
    /**
     * Calculate intersection over union (IoU)
     */
    fun iou(other: BoundingBox): Float {
        val intersectionBox = intersection(other)
        val intersectionArea = intersectionBox.area
        val unionArea = this.area + other.area - intersectionArea
        
        return if (unionArea > 0) {
            intersectionArea / unionArea
        } else {
            0f
        }
    }
    
    /**
     * Check if this box contains a point
     */
    fun contains(point: PointF): Boolean {
        return point.x >= x1 && point.x <= x2 &&
               point.y >= y1 && point.y <= y2
    }
    
    /**
     * Scale the bounding box by a factor
     */
    fun scale(scale: Float): BoundingBox {
        val centerX = (x1 + x2) / 2f
        val centerY = (y1 + y2) / 2f
        val width = (x2 - x1) * scale
        val height = (y2 - y1) * scale
        
        return BoundingBox(
            x1 = centerX - width / 2f,
            y1 = centerY - height / 2f,
            x2 = centerX + width / 2f,
            y2 = centerY + height / 2f
        )
    }
    
    /**
     * Normalize coordinates to [0, 1] range
     */
    fun normalize(imageWidth: Float, imageHeight: Float): BoundingBox {
        return BoundingBox(
            x1 = x1 / imageWidth,
            y1 = y1 / imageHeight,
            x2 = x2 / imageWidth,
            y2 = y2 / imageHeight
        )
    }
    
    /**
     * Denormalize from [0, 1] range to actual pixel coordinates
     */
    fun denormalize(imageWidth: Float, imageHeight: Float): BoundingBox {
        return BoundingBox(
            x1 = x1 * imageWidth,
            y1 = y1 * imageHeight,
            x2 = x2 * imageWidth,
            y2 = y2 * imageHeight
        )
    }
}