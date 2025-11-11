#!/bin/bash

# YOLOv11n Human Detection Android App Build Script
# This script helps build and test the application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Android SDK
check_android_sdk() {
    if [ -z "$ANDROID_HOME" ]; then
        print_error "ANDROID_HOME environment variable not set"
        print_status "Please set ANDROID_HOME to your Android SDK directory"
        exit 1
    fi
    print_success "Android SDK found at: $ANDROID_HOME"
}

# Function to check Java version
check_java() {
    if command_exists java; then
        JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1-2)
        print_success "Java version: $JAVA_VERSION"
    else
        print_error "Java not found. Please install JDK 8 or later"
        exit 1
    fi
}

# Function to check Gradle
check_gradle() {
    if [ -f "gradlew" ]; then
        print_success "Gradle wrapper found"
        GRADLE_CMD="./gradlew"
    elif command_exists gradle; then
        print_success "Gradle found in PATH"
        GRADLE_CMD="gradle"
    else
        print_error "Gradle not found. Please install Gradle or use the wrapper"
        exit 1
    fi
}

# Function to clean previous builds
clean_build() {
    print_status "Cleaning previous builds..."
    $GRADLE_CMD clean
    print_success "Clean completed"
}

# Function to run tests
run_tests() {
    print_status "Running unit tests..."
    $GRADLE_CMD test
    if [ $? -eq 0 ]; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        return 1
    fi
}

# Function to build debug APK
build_debug() {
    print_status "Building debug APK..."
    $GRADLE_CMD assembleDebug
    if [ $? -eq 0 ]; then
        print_success "Debug APK built successfully"
        print_status "APK location: app/build/outputs/apk/debug/app-debug.apk"
    else
        print_error "Debug build failed"
        return 1
    fi
}

# Function to build release APK
build_release() {
    print_status "Building release APK..."
    $GRADLE_CMD assembleRelease
    if [ $? -eq 0 ]; then
        print_success "Release APK built successfully"
        print_status "APK location: app/build/outputs/apk/release/app-release.apk"
    else
        print_error "Release build failed"
        return 1
    fi
}

# Function to run lint
run_lint() {
    print_status "Running Android lint..."
    $GRADLE_CMD lint
    if [ $? -eq 0 ]; then
        print_success "Lint checks passed"
    else
        print_warning "Lint checks found issues"
    fi
}

# Function to install on device
install_on_device() {
    if [ -z "$1" ]; then
        print_error "Please specify APK path"
        return 1
    fi
    
    print_status "Installing APK on connected device..."
    adb install -r "$1"
    if [ $? -eq 0 ]; then
        print_success "APK installed successfully"
        print_status "Starting app..."
        adb shell am start -n com.yolodetection.app/.ui.MainActivity
    else
        print_error "Failed to install APK"
        return 1
    fi
}

# Function to show help
show_help() {
    echo "YOLOv11n Human Detection Android App Build Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup       - Check environment and setup"
    echo "  clean       - Clean build directories"
    echo "  test        - Run unit tests"
    echo "  debug       - Build debug APK"
    echo "  release     - Build release APK"
    echo "  install     - Install debug APK on device"
    echo "  lint        - Run Android lint checks"
    echo "  build-all   - Run clean, test, lint, and build debug"
    echo "  help        - Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  ANDROID_HOME - Path to Android SDK"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 build-all"
    echo "  $0 install app/build/outputs/apk/debug/app-debug.apk"
}

# Main script logic
main() {
    case "${1:-help}" in
        "setup")
            print_status "Setting up build environment..."
            check_android_sdk
            check_java
            check_gradle
            print_success "Environment setup complete"
            ;;
        "clean")
            check_gradle
            clean_build
            ;;
        "test")
            check_gradle
            run_tests
            ;;
        "debug")
            check_gradle
            build_debug
            ;;
        "release")
            check_gradle
            build_release
            ;;
        "install")
            if [ -z "$2" ]; then
                print_status "Building and installing debug APK..."
                check_gradle
                build_debug
                install_on_device "app/build/outputs/apk/debug/app-debug.apk"
            else
                install_on_device "$2"
            fi
            ;;
        "lint")
            check_gradle
            run_lint
            ;;
        "build-all")
            check_gradle
            clean_build
            run_tests
            run_lint
            build_debug
            print_success "Build process complete!"
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"