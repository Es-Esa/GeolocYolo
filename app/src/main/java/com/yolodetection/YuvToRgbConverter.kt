package com.yolodetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream

/**
 * Utility class for converting YUV image data to RGB Bitmap
 * Optimized for real-time camera preview processing
 */
object YuvToRgbConverter {
    
    /**
     * Convert YUV image to RGB Bitmap
     * 
     * @param context Application context
     * @param yuvData YUV 420 888 image data
     * @param width Image width
     * @param height Image height
     * @return Converted RGB Bitmap
     */
    fun convertYuvToBitmap(
        context: Context,
        yuvData: ByteArray,
        width: Int,
        height: Int
    ): Bitmap? {
        return try {
            // Create YuvImage from YUV data
            val yuvImage = YuvImage(
                yuvData,
                ImageFormat.YUV_420_888,
                width,
                height,
                null
            )
            
            // Create output stream
            val outputStream = ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                Rect(0, 0, width, height),
                80, // Quality
                outputStream
            )
            
            // Convert to byte array and decode
            val imageBytes = outputStream.toByteArray()
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
    
    /**
     * Alternative YUV to RGB conversion using direct conversion
     * This is faster but requires more memory
     */
    fun convertYuvToRgbDirect(
        yuvData: ByteArray,
        width: Int,
        height: Int,
        output: IntArray
    ): IntArray {
        val frameSize = width * height
        var yp = 0
        
        for (j in 0 until height) {
            var u = 0
            var v = 0
            
            val uvp = frameSize + (j shr 1) * width
            val uvg = uvp + width / 2
            
            for (i in 0 until width) {
                val y = (0xff and yuvData[yp].toInt()) - 16
                
                if (i and 1 == 0) {
                    v = 0xff and yuvData[uvg].toInt() - 128
                    u = 0xff and yuvData[uvp].toInt() - 128
                    uvp += 2
                    uvg += 2
                } else {
                    v = 0xff and yuvData[uvg].toInt() - 128
                    u = 0xff and yuvData[uvp].toInt() - 128
                }
                
                val y1192 = 1192 * y
                val r = (y1192 + 1634 * v)
                val g = (y1192 - 833 * v - 400 * u)
                val b = (y1192 + 1296 * u)
                
                var rr = r shr 10
                var gg = g shr 10
                var bb = b shr 10
                
                rr = rr.coerceIn(0, 255)
                gg = gg.coerceIn(0, 255)
                bb = bb.coerceIn(0, 255)
                
                output[yp++] = -0x1000000 + (rr shl 16) + (gg shl 8) + bb
            }
        }
        
        return output
    }
    
    /**
     * Get optimal YUV conversion method based on device capabilities
     */
    fun getOptimalConversionMethod(
        context: Context,
        imageSize: Int
    ): ConversionMethod {
        // Use direct conversion for smaller images, JPEG compression for larger ones
        return when {
            imageSize < 500 * 500 -> ConversionMethod.DIRECT
            else -> ConversionMethod.JPEG_COMPRESSION
        }
    }
    
    /**
     * Conversion method enum
     */
    enum class ConversionMethod {
        JPEG_COMPRESSION,
        DIRECT
    }
}