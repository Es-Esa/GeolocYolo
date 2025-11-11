package com.yolodetection.app

import android.app.Application
import android.content.Context
import com.yolodetection.app.utils.PermissionUtils
import timber.log.Timber

/**
 * Global Application class for YOLOv11n Human Detection App
 * 
 * Handles global initialization, permission setup, and logging configuration
 */
class YoloDetectionApp : Application() {
    
    companion object {
        private lateinit var instance: YoloDetectionApp
        lateinit var permissionUtils: PermissionUtils
        
        fun getInstance(): Context = instance
    }
    
    override fun onCreate() {
        super.onCreate()
        instance = this
        
        // Initialize logging
        initLogging()
        
        // Initialize permission manager
        permissionUtils = PermissionUtils(this)
        
        // Initialize global configuration
        initConfiguration()
        
        // Setup crash reporting
        setupCrashHandling()
    }
    
    private fun initLogging() {
        if (BuildConfig.DEBUG) {
            Timber.plant(Timber.DebugTree())
        } else {
            // In production, plant a production logging tree
            // Timber.plant(ProductionTree())
        }
        Timber.d("YOLOv11n Human Detection App initialized")
    }
    
    private fun initConfiguration() {
        // Load any global configuration
        Timber.d("Loading global configuration")
        
        // Set performance settings based on device capabilities
        PerformanceManager.initialize(this)
    }
    
    private fun setupCrashHandling() {
        if (!BuildConfig.DEBUG) {
            // Setup crash reporting service in production
            // Crashlytics.initialize(this)
            Timber.d("Crash reporting initialized for production")
        }
    }
    
    override fun onTerminate() {
        super.onTerminate()
        Timber.d("Application terminated, cleaning up resources")
        
        // Clean up any global resources
        PerformanceManager.cleanup()
    }
    
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        Timber.d("Memory trim requested, level: $level")
        
        // Trigger memory cleanup
        PerformanceManager.handleMemoryPressure(level)
    }
}