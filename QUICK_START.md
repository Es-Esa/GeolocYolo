# YOLOv11n Human Detection - Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will help you get the YOLOv11n Human Detection app up and running on your Android device in just a few minutes.

## 📋 Prerequisites

### Required Software
- **Android Studio** (Giraffe or later) or command-line tools
- **Android SDK** API 24+ (Android 7.0)
- **JDK 8+** (Java Development Kit)
- **Android device** with API 24+ or Android emulator

### Hardware Requirements
- **Minimum**: 2GB RAM, ARM64 processor
- **Recommended**: 4GB+ RAM, Snapdragon 660+ or equivalent
- **Storage**: 500MB free space

## 🔧 Environment Setup

### 1. Set Android SDK Path
```bash
export ANDROID_HOME=/path/to/your/android/sdk
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
```

### 2. Verify Setup
```bash
# Check Android SDK
echo $ANDROID_HOME

# Check connected devices
adb devices

# Should show connected devices or emulator
```

## 🏗️ Build the App

### Option A: Automated Build Script
```bash
# Make script executable
chmod +x build.sh

# Build and install
./build.sh build-all
./build.sh install
```

### Option B: Manual Build
```bash
# Clean and build debug APK
./gradlew clean assembleDebug

# Install on connected device
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## 📱 Install and Run

### 1. Install on Device
```bash
# Install via ADB
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Or drag APK to emulator in Android Studio
```

### 2. Grant Permissions
When the app starts for the first time:
- **Allow camera access** - Required for detection
- **Allow storage access** - For saving images/results
- **Allow overlay permission** - For detection visualization

### 3. Launch App
```bash
# Via ADB
adb shell am start -n com.yolodetection.app/.ui.MainActivity

# Or tap the app icon on your device
```

## ⚡ First Run - Basic Test

### 1. Basic Camera Test
1. Open the app
2. You should see **camera preview** immediately
3. Check that **camera switch button** works (front/back)

### 2. Enable Detection
1. Tap the **detection toggle** button
2. You should see **bounding boxes** appear around people
3. **Green boxes** indicate detected humans
4. **Confidence scores** appear above boxes

### 3. Settings Test
1. Tap **settings icon** (gear)
2. Adjust **confidence threshold** (0.3-0.9)
3. Try different **input sizes** (320, 416, 640)
4. Test **person-only mode**

## 📊 Performance Check

### Quick Performance Test
- **High-end devices**: Should achieve 25-30 FPS
- **Mid-range devices**: 15-20 FPS expected
- **Low-end devices**: 8-15 FPS is normal

### Check Statistics
1. Tap **statistics icon** in main screen
2. View **real-time metrics**:
   - FPS (frames per second)
   - Memory usage
   - Detection count
   - Model load time

## 🎯 First Successful Detection

### Test Scenario
1. **Position yourself** in front of the camera
2. **Wait 2-3 seconds** for model initialization
3. **Look for green bounding box** around your body
4. **Confidence score** should be > 0.5

### Expected Results
- ✅ **Green bounding box** around detected person
- ✅ **Confidence score** 0.5-0.95
- ✅ **Real-time updates** at target FPS
- ✅ **Smooth performance** without lag

## 🔍 Common First-Run Issues

### Camera Not Working
**Problem**: Black screen or camera error
**Solution**:
- Check camera permissions
- Restart app
- Try different camera (front/back)

### No Detections
**Problem**: No bounding boxes appear
**Solution**:
- Ensure good lighting
- Check confidence threshold (try 0.3)
- Verify person is clearly visible
- Wait for model initialization (5-10 seconds)

### Slow Performance
**Problem**: Low FPS or lag
**Solution**:
- Reduce input size (320x320)
- Close other apps
- Switch to person-only mode
- Check device temperature

### App Crashes
**Problem**: App closes unexpectedly
**Solution**:
- Clear app cache
- Check available memory
- Update Android OS
- Try different model quantization

## 🎉 Success Checklist

After your first run, you should have:
- [ ] App installed and running
- [ ] Camera preview working
- [ ] Permissions granted
- [ ] Human detection working
- [ ] Bounding boxes displayed
- [ ] Reasonable FPS achieved
- [ ] Settings menu accessible
- [ ] Statistics view working

## 📞 Need Help?

### Next Steps
- **Performance Issues**: See [PERFORMANCE_TESTING.md](PERFORMANCE_TESTING.md)
- **Feature Details**: See [FEATURES_GUIDE.md](FEATURES_GUIDE.md)
- **Detailed Testing**: See [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Build Problems**: See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)

### Support Resources
- **Logs**: Check `adb logcat` for detailed error messages
- **Device Info**: Note your device model and Android version
- **Performance**: Record FPS and memory usage stats

## 🏃‍♂️ What Next?

Once basic functionality works:
1. **Test different lighting conditions**
2. **Try various camera angles and distances**
3. **Experiment with settings**
4. **Test on different devices**
5. **Run performance benchmarks**

You're now ready to explore the full capabilities of the YOLOv11n Human Detection app! 🚀

---

*For detailed troubleshooting and advanced features, refer to the other guides in this documentation.*