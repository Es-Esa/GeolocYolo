# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# Keep TensorFlow Lite classes
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.support.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.nnapi.** { *; }

# Keep YOLO model classes
-keep class com.yolodetection.app.detection.** { *; }
-keep class com.yolodetection.app.overlay.** { *; }
-keep class com.yolodetection.app.utils.** { *; }

# Keep model loading methods
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# Keep reflection methods used by TensorFlow Lite
-keepattributes *Annotation*
-keepclassmembers class * {
    @org.tensorflow.lite.annotations.** <methods>;
}

# Keep data classes
-keep class com.yolodetection.app.detection.models.** { *; }
-keep class com.yolodetection.app.utils.PerformanceManager.** { *; }

# Keep Camera2 classes
-keep class androidx.camera.** { *; }
-keep class android.hardware.camera2.** { *; }

# Keep coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}

# Remove logging in release builds
-assumenosideeffects class android.util.Log {
    public static *** d(...);
    public static *** v(...);
    public static *** i(...);
}

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep enums
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Keep Parcelable implementations
-keepclassmembers class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator CREATOR;
}

# Keep generic types
-keepattributes Signature
-keepattributes InnerClasses
-keepattributes EnclosingMethod

# Optimize and obfuscate
-optimizations !code/simplification/arithmetic,!code/simplification/cast,!field/*,!class/merging/*
-optimizationpasses 5
-allowaccessmodification
-dontpreverify

# Remove debug information
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile