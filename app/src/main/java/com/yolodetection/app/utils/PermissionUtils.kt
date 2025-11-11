package com.yolodetection.app.utils

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import timber.log.Timber

/**
 * Utility class for handling runtime permissions
 */
class PermissionUtils(private val context: Context) {
    
    /**
     * Check if camera permission is granted
     */
    fun hasCameraPermission(): Boolean {
        val result = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
        return result == PackageManager.PERMISSION_GRANTED
    }
    
    /**
     * Check if storage permission is granted
     */
    fun hasStoragePermission(): Boolean {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ uses media permissions
            true // For now, assume granted as we handle media permissions separately
        } else {
            val result = ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            result == PackageManager.PERMISSION_GRANTED
        }
    }
    
    /**
     * Get all required permissions
     */
    fun getRequiredPermissions(): Array<String> {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ uses granular media permissions
            arrayOf(Manifest.permission.CAMERA)
        } else {
            // Android 12 and below
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
        }
    }
    
    /**
     * Get all granted permissions
     */
    fun getGrantedPermissions(): Array<String> {
        val required = getRequiredPermissions()
        return required.filter { permission ->
            ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        }.toTypedArray()
    }
    
    /**
     * Check if all required permissions are granted
     */
    fun areAllPermissionsGranted(): Boolean {
        val required = getRequiredPermissions()
        return required.all { permission ->
            ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        }
    }
    
    /**
     * Get missing permissions
     */
    fun getMissingPermissions(): Array<String> {
        val required = getRequiredPermissions()
        return required.filter { permission ->
            ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED
        }.toTypedArray()
    }
    
    /**
     * Log permission status for debugging
     */
    fun logPermissionStatus() {
        val granted = getGrantedPermissions()
        val missing = getMissingPermissions()
        
        Timber.d("Permission Status - Granted: ${granted.joinToString(", ")}")
        Timber.d("Permission Status - Missing: ${missing.joinToString(", ")}")
        Timber.d("All permissions granted: ${areAllPermissionsGranted()}")
    }
    
    /**
     * Check if we should show permission rationale
     */
    fun shouldShowPermissionRationale(permission: String): Boolean {
        return android.app.ActivityCompat.shouldShowRequestPermissionRationale(
            context as? android.app.Activity ?: return false,
            permission
        )
    }
    
    /**
     * Get permission explanation
     */
    fun getPermissionExplanation(permission: String): String {
        return when (permission) {
            Manifest.permission.CAMERA -> "Camera permission is required to detect humans in real-time"
            Manifest.permission.WRITE_EXTERNAL_STORAGE -> "Storage permission is required to save captured images"
            else -> "This permission is required for the app to function properly"
        }
    }
    
    /**
     * Get max resolution for preview based on permissions
     */
    fun getMaxPreviewResolution(): Pair<Int, Int> {
        return if (hasCameraPermission()) {
            // Camera2 API supports high resolution
            Pair(1920, 1080) // 1080p
        } else {
            // Fallback to lower resolution if no camera permission
            Pair(640, 480)
        }
    }
}