# YOLOv11n Human Detection - Build Instructions

## 🏗️ Complete Build and Deployment Guide

This comprehensive guide covers building, configuring, and deploying the YOLOv11n Human Detection Android app from source code.

## 📋 Prerequisites

### Required Software
```bash
# Java Development Kit
JDK 8 or later (JDK 11 recommended)
java -version  # Verify installation

# Android Studio and SDK
Android Studio 4.0+
Android SDK API 24+ (Android 7.0+)
Android SDK Build-Tools 34.0.0+

# Version Control
Git (for source code management)
```

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux Ubuntu 18.04+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for SDK and build tools
- **Internet**: Stable connection for dependency downloads

### Environment Variables
```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)

# Android SDK
export ANDROID_HOME=$HOME/Android/Sdk  # Update path for your system
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools

# Java (if not using Android Studio's bundled version)
export JAVA_HOME=/path/to/your/jdk  # Update path

# Verify setup
echo $ANDROID_HOME
echo $JAVA_HOME
```

## 🔧 Development Environment Setup

### Step 1: Install Android Studio
1. **Download** Android Studio from [developer.android.com](https://developer.android.com/studio)
2. **Install** following installation wizard
3. **Launch** Android Studio
4. **Complete** initial setup wizard
5. **Install** required SDK components

### Step 2: Configure SDK
```bash
# Open SDK Manager (via Android Studio or command line)
sdkmanager --list  # List available packages

# Install required packages
sdkmanager "platforms;android-34"
sdkmanager "build-tools;34.0.0"
sdkmanager "platform-tools"
sdkmanager "extras;android;m2repository"
```

### Step 3: Setup Command Line Tools
```bash
# Create commandline-tools directory
mkdir -p $ANDROID_HOME/cmdline-tools/latest
cd $ANDROID_HOME/cmdline-tools/latest

# Download command line tools
# Visit: https://developer.android.com/studio/command-line

# Add to PATH (add to your shell profile)
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin
```

### Step 4: Verify Environment
```bash
# Test Android SDK
adb version

# Should show something like:
# Android Debug Bridge version 1.0.41

# Test Gradle (if using wrapper)
./gradlew --version

# Should show Gradle 8.x and Kotlin 1.9.x
```

## 📦 Project Setup

### Step 1: Clone or Extract Project
```bash
# If from Git
git clone <repository-url>
cd android-yolov11-detection

# Or extract from archive
unzip android-project.zip
cd android-yolov11-detection
```

### Step 2: Project Structure Verification
```bash
# Verify key files exist
ls -la android/
# Should show:
# build.gradle
# settings.gradle
# gradle.properties
# app/
#   build.gradle
#   src/
#   assets/

# Check model files
ls -la android/app/src/main/assets/models/
# Should show:
# yolov11n_fp16.tflite
# yolov11n_int8.tflite
# labels.txt
```

### Step 3: Update Project Configuration
```bash
# Edit android/app/build.gradle if needed
# Update minSdkVersion, targetSdkVersion, versionCode, versionName

# Check dependencies
grep -n "implementation" app/build.gradle
```

## 🔨 Building the Project

### Method 1: Using Build Script (Recommended)
```bash
# Make script executable
chmod +x build.sh

# Setup environment
./build.sh setup

# Full build process (clean, test, lint, debug)
./build.sh build-all

# Or build debug APK only
./build.sh debug

# Or build release APK
./build.sh release
```

### Method 2: Manual Gradle Build
```bash
# Navigate to project directory
cd android/

# Clean previous builds
./gradlew clean

# Build debug APK
./gradlew assembleDebug

# Build release APK (requires signing)
./gradlew assembleRelease

# Run tests
./gradlew test

# Run lint checks
./gradlew lint
```

### Method 3: Android Studio Build
1. **Open** Android Studio
2. **Select** "Open an Existing Project"
3. **Navigate** to the `android/` directory
4. **Click** "OK"
5. **Wait** for project synchronization
6. **Build** menu → "Build Bundle(s) / APK(s)" → "Build APK(s)"

## 🔑 Signing Configuration

### For Release Builds
```bash
# Generate keystore (one-time setup)
keytool -genkey -v -keystore release-key.keystore \
  -alias yolo-release -keyalg RSA -keysize 2048 -validity 10000

# Configure signing in app/build.gradle
android {
    signingConfigs {
        release {
            storeFile file('release-key.keystore')
            storePassword 'your_keystore_password'
            keyAlias 'yolo-release'
            keyPassword 'your_key_password'
        }
    }
    
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
}
```

### Using Gradle Properties
```bash
# Create gradle.properties (already exists in project)
# Add signing configuration
YOLO_STORE_FILE=release-key.keystore
YOLO_STORE_PASSWORD=your_store_password
YOLO_KEY_ALIAS=yolo-release
YOLO_KEY_PASSWORD=your_key_password

# Reference in build.gradle
android {
    signingConfigs {
        release {
            if (project.hasProperty('YOLO_STORE_FILE')) {
                storeFile file(YOLO_STORE_FILE)
                storePassword YOLO_STORE_PASSWORD
                keyAlias YOLO_KEY_ALIAS
                keyPassword YOLO_KEY_PASSWORD
            }
        }
    }
}
```

## 📱 Device Setup and Installation

### Enable Developer Options
1. **Settings** → **About Phone**
2. **Tap** "Build Number" 7 times
3. **Go back** to Settings
4. **Developer Options** → **Enable USB Debugging**

### Connect Device
```bash
# Check device connection
adb devices

# Should show:
# List of devices attached
# [device-id]    device

# If device not detected:
# - Enable USB debugging
# - Install device drivers (Windows)
# - Try different USB cable
# - Try different USB port
```

### Install APK
```bash
# Install via ADB
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Install with APK name
adb install -r path/to/your/app.apk

# Force install (replace existing)
adb install -r -d app-debug.apk

# Install for specific device
adb -s [device-id] install app-debug.apk
```

## 🧪 Testing During Development

### Unit Tests
```bash
# Run unit tests
./gradlew test

# Run specific test class
./gradlew test --tests "com.yolodetection.app.detection.YoloDetectorTest"

# Generate test report
./gradlew testDebugUnitTest
# Report location: app/build/reports/tests/testDebugUnitTest/index.html
```

### Instrumented Tests
```bash
# Run on connected device
./gradlew connectedAndroidTest

# Run specific test
./gradlew connectedAndroidTest --tests "com.yolodetection.app.detection.IntegrationTest"

# Generate coverage report
./gradlew jacocoTestReport
```

### Performance Profiling
```bash
# Enable profiling (debug build only)
# In MainActivity.kt, set:
val USE_PROFILING = true

# Run with profiling
adb shell am start -n com.yolodetection.app/.ui.MainActivity

# Capture system traces
adb shell cat /sys/kernel/debug/tracing/trace > performance-trace.txt
```

## 🔍 Build Troubleshooting

### Common Build Issues

#### Gradle Build Failed
```bash
# Clear Gradle cache
./gradlew clean

# Delete build directories
rm -rf app/build
rm -rf .gradle

# Rebuild
./gradlew assembleDebug
```

#### SDK Not Found
```bash
# Check ANDROID_HOME
echo $ANDROID_HOME

# If not set, find SDK path:
# Windows: %LOCALAPPDATA%/Android/Sdk
# Mac: ~/Library/Android/sdk
# Linux: ~/Android/Sdk

# Set environment variable
export ANDROID_HOME=~/Android/Sdk
```

#### Dependencies Failed
```bash
# Clean Gradle cache
./gradlew --refresh-dependencies

# Check network connectivity
# Try building in offline mode
./gradlew assembleDebug --offline
```

#### Model Files Missing
```bash
# Check if model files exist
ls -la app/src/main/assets/models/

# If missing, copy from backup or regenerate:
# Place TFLite model files in assets/models/
# Ensure names match: yolov11n_fp16.tflite, yolov11n_int8.tflite
```

#### Build Memory Issues
```bash
# Increase Gradle heap size
# Add to gradle.properties:
org.gradle.jvmargs=-Xmx4g -XX:MaxMetaspaceSize=512m

# Enable parallel builds
org.gradle.parallel=true
org.gradle.daemon=true
```

### Build Log Analysis
```bash
# Save build log
./gradlew assembleDebug > build.log 2>&1

# Filter for errors
grep -i error build.log

# Filter for warnings
grep -i warning build.log
```

## 📊 Build Output Analysis

### APK Information
```bash
# Analyze APK size and contents
aapt dump badging app/build/outputs/apk/debug/app-debug.apk

# Check APK size
ls -lh app/build/outputs/apk/debug/app-debug.apk

# View APK structure
unzip -l app-debug.apk | head -20
```

### Performance Metrics
```bash
# APK size targets
# Debug: < 100MB
# Release: < 50MB (with ProGuard)

# Build time targets
# Debug: < 5 minutes
# Release: < 10 minutes

# Memory usage during build
# Monitor with system tools
top -p $(pgrep -f gradle)
```

## 🚀 Deployment Options

### Option 1: Direct Device Installation
```bash
# Install via ADB
adb install -r app-debug.apk

# Install with app name
adb install -r "YOLOv11n Detection.apk"
```

### Option 2: Google Play Store
```bash
# Generate signed bundle
./gradlew bundleRelease

# Upload to Google Play Console
# Location: app/build/outputs/bundle/release/app-release.aab
```

### Option 3: Firebase App Distribution
```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login to Firebase
firebase login

# Distribute to testers
firebase appdistribution:distribute \
  app/build/outputs/apk/release/app-release.apk \
  --app "your-firebase-app-id" \
  --groups "testers" \
  --release-notes "Bug fixes and performance improvements"
```

### Option 4: Custom Distribution
```bash
# Create distribution directory
mkdir -p distribution
cp app/build/outputs/apk/debug/app-debug.apk distribution/

# Generate release notes
cat > distribution/RELEASE_NOTES.txt << EOF
YOLOv11n Human Detection v1.0
============================

Features:
- Real-time human detection
- Multiple model formats (FP16, INT8)
- Performance optimization
- Modern UI/UX

Installation:
1. Download and install the APK
2. Grant camera permissions
3. Start detecting!

Known Issues:
- None

For support: [contact information]
EOF
```

## 📈 Build Optimization

### Speed Up Builds
```gradle
// gradle.properties
org.gradle.parallel=true
org.gradle.caching=true
org.gradle.daemon=true
org.gradle.configureondemand=true
android.enableBuildCache=true
android.enableR8.fullMode=true
```

### Reduce APK Size
```gradle
// app/build.gradle
android {
    buildTypes {
        release {
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    
    packagingOptions {
        pickFirst '**/libc++_shared.so'
        pickFirst '**/libjsc.so'
    }
}
```

### Enable R8 Optimization
```gradle
// gradle.properties
android.enableR8=true
android.enableR8.fullMode=true
```

## 🔄 Continuous Integration

### GitHub Actions Example
```yaml
# .github/workflows/android.yml
name: Android CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up JDK 11
      uses: actions/setup-java@v2
      with:
        java-version: '11'
        distribution: 'adopt'
        
    - name: Setup Android SDK
      uses: android-actions/setup-android@v2
      
    - name: Grant execute permission for gradlew
      run: chmod +x gradlew
      
    - name: Run tests
      run: ./gradlew test
      
    - name: Build debug APK
      run: ./gradlew assembleDebug
```

### Jenkins Pipeline Example
```groovy
pipeline {
    agent any
    tools {
        gradle 'Gradle 8.0'
        jdk 'JDK 11'
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Build') {
            steps {
                sh './gradlew assembleDebug'
            }
        }
        stage('Test') {
            steps {
                sh './gradlew test'
            }
        }
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'adb install -r app-debug.apk'
            }
        }
    }
}
```

## ✅ Build Success Verification

### Post-Build Checklist
- [ ] APK generated successfully
- [ ] APK size within acceptable limits (< 50MB for release)
- [ ] All tests pass
- [ ] No lint errors (or only acceptable warnings)
- [ ] Model files included in APK
- [ ] App installs on test device
- [ ] App launches without crashes
- [ ] Basic functionality works (camera, detection)

### Final Verification
```bash
# Final build test
./build.sh build-all

# Install and test
./build.sh install

# Verify on device
adb shell am start -n com.yolodetection.app/.ui.MainActivity
```

## 📞 Support and Resources

### Build Issues
- **Gradle Issues**: Check Gradle wrapper version
- **SDK Issues**: Verify ANDROID_HOME and PATH
- **Memory Issues**: Increase Gradle heap size
- **Network Issues**: Check proxy/firewall settings

### Documentation
- [Android Build System](https://developer.android.com/studio/build)
- [Gradle User Guide](https://docs.gradle.org/current/userguide/userguide.html)
- [TensorFlow Lite Android](https://www.tensorflow.org/lite/guide/android)

### Getting Help
- Check build logs for specific error messages
- Verify all prerequisites are installed
- Test with minimal project (no custom code)
- Try different Gradle version
- Consult Android development forums

This build guide provides a comprehensive foundation for building and deploying the YOLOv11n Human Detection Android app in any development environment.