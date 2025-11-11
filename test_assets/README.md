# Android Human Detection Test Assets

This directory contains comprehensive test assets for evaluating human detection models on Android devices. The test suite includes sample images, performance testing scripts, model comparisons, and benchmark templates.

## Directory Structure

```
test_assets/
├── sample_test_images/          # Test images for human detection
├── performance_test_suite/      # Scripts for performance testing
├── model_comparison/           # Different model variants and configs
├── benchmark_results/          # Templates and results storage
└── demo_videos/                # Video testing materials
```

## 1. Sample Test Images (`sample_test_images/`)

Contains 24 test images covering various scenarios for human detection testing:

### Image Categories:
- **Single Person**: 3 images - `single_person_standing_*.jpg/png`
- **Multiple People**: 3 images - `multiple_people_group_*.jpg/png/jpeg`
- **Indoor Lighting**: 3 images - `indoor_lighting_*.jpg`
- **Outdoor Daylight**: 3 images - `outdoor_daylight_*.jpg/png`
- **Low Light**: 3 images - `low_light_*.jpg`
- **Motion/Action**: 3 images - `person_motion_*.jpg/png`
- **Close-up Portraits**: 3 images - `close_up_portrait_*.jpg`
- **Backlit/Silhouette**: 3 images - `backlit_silhouette_*.jpg`

### Usage:
```python
import cv2
import numpy as np

# Load and test an image
image_path = "sample_test_images/single_person_standing_2.jpg"
image = cv2.imread(image_path)
# Your detection code here...
```

## 2. Performance Test Suite (`performance_test_suite/`)

### Scripts Included:

#### `performance_benchmark.py`
- Measures inference time, memory usage, and accuracy
- Supports multiple model testing
- Generates detailed performance reports
- **Usage**: `python performance_benchmark.py`

#### `stress_test.py`
- Tests model under concurrent load
- Memory stress testing
- Batch processing performance
- **Usage**: `python stress_test.py`

#### `android_performance_test.py`
- Simulates different Android device classes
- Battery impact testing
- Thermal throttling simulation
- **Usage**: `python android_performance_test.py`

### Example Usage:
```python
from performance_benchmark import PerformanceBenchmark

# Initialize benchmark
benchmark = PerformanceBenchmark()
model = YourHumanDetectionModel()

# Run performance tests
results = benchmark.run_benchmark_suite(
    model=model,
    test_images_dir="../sample_test_images",
    output_file="../benchmark_results/performance_results.json"
)
```

## 3. Model Comparison (`model_comparison/`)

### Files:
- **`README.md`**: Comprehensive guide to different YOLO variants
- **`comparison_config.json`**: Configuration for testing multiple models

### Model Variants:
1. **YOLOv11n (Nano)**: ~6MB, 320x320, INT8 optimization
2. **YOLOv11s (Small)**: ~14MB, 416x416, FP16 optimization
3. **YOLOv11m (Medium)**: ~25MB, 416x416, FP32 optimization
4. **YOLOv11l (Large)**: ~45MB, 512x512, FP32 optimization

### Target Device Classes:
- **Low-end**: 2GB RAM, quad-core → Use YOLOv11n
- **Mid-range**: 4GB RAM, octa-core → Use YOLOv11s
- **High-end**: 8GB RAM, flagship → Use YOLOv11m
- **Flagship**: 12GB RAM, latest processor → Use YOLOv11l

## 4. Benchmark Results (`benchmark_results/`)

### Template Files:
- **`benchmark_results_template.json`**: Complete benchmark results template
- **`model_comparison_summary.csv`**: Quick comparison spreadsheet

### Performance Metrics Tracked:
- **Speed**: Inference time (ms), FPS, batch processing time
- **Accuracy**: Precision, recall, F1-score, mAP
- **Efficiency**: Memory usage, battery drain, CPU utilization
- **Scenario Performance**: Breakdown by test conditions

### Example Results Analysis:
```python
import json
import pandas as pd

# Load benchmark results
with open('benchmark_results_template.json', 'r') as f:
    results = json.load(f)

# Quick comparison
comparison_df = pd.read_csv('model_comparison_summary.csv')
print(comparison_df[['Model Name', 'FPS', 'Accuracy Category']])
```

## 5. Demo Videos (`demo_videos/`)

### Files:
- **`README.md`**: Guide to video testing
- **`video_metadata_template.json`**: Template for video test metadata
- **`video_performance_test.py`**: Script for video performance testing

### Recommended Video Characteristics:
- **Duration**: 10-30 seconds (quick test) or 1-5 minutes (comprehensive)
- **Resolution**: 720p to 1080p
- **Frame Rate**: 30 FPS
- **Format**: MP4 (H.264 codec)

### Video Categories:
1. **Single Person**: Walking, standing, different poses
2. **Multiple People**: Groups, conversations, interactions
3. **Crowd Scenes**: Dense populations, movement patterns
4. **Motion/Action**: Running, fast movement, motion blur
5. **Challenging Conditions**: Low light, night, shadows
6. **Mixed Activities**: Compilation of various scenarios

### Video Testing:
```python
from video_performance_test import VideoPerformanceTest

# Initialize video test
video_test = VideoPerformanceTest(model)

# Test single video
config = {"process_every_n_frames": 1}
result = video_test.process_video_file(
    video_path="demo_videos/single_person_walking.mp4",
    output_dir="benchmark_results",
    config=config
)

# Test real-time camera
camera_result = video_test.real_time_video_test(duration=30)
```

## Getting Started

### 1. Quick Setup
```bash
# Navigate to test assets directory
cd android/test_assets

# Install required packages (if using Python scripts)
pip install opencv-python numpy psutil pandas
```

### 2. Basic Testing Workflow
1. **Start with sample images**: Test your model on provided images
2. **Run performance benchmarks**: Use performance test suite scripts
3. **Compare model variants**: Use model comparison configurations
4. **Test video performance**: Use video testing scripts
5. **Analyze results**: Use benchmark results templates

### 3. Example Complete Test
```python
# Complete testing example
from performance_benchmark import PerformanceBenchmark
from video_performance_test import VideoPerformanceTest
import json

# Initialize components
benchmark = PerformanceBenchmark()
video_test = VideoPerformanceTest(model)

# Run image benchmarks
image_results = benchmark.run_benchmark_suite(
    model=model,
    test_images_dir="sample_test_images",
    output_file="benchmark_results/image_results.json"
)

# Run video tests
video_results = video_test.run_video_test_suite(
    video_dir="demo_videos",
    output_dir="benchmark_results",
    config={"process_every_n_frames": 1}
)

# Load and analyze results
with open("benchmark_results/image_results.json") as f:
    results = json.load(f)

print(f"Average FPS: {results['benchmarks'][0]['fps']}")
print(f"Memory Usage: {results['benchmarks'][1]['peak_memory_mb']} MB")
```

## Customization

### Adding Your Own Test Images
1. Place images in `sample_test_images/`
2. Follow naming convention: `{scenario}_{description}_{id}.{ext}`
3. Recommended scenarios: single_person, multiple_people, indoor, outdoor, etc.

### Creating Custom Test Configurations
1. Copy `model_comparison/comparison_config.json`
2. Modify model paths, thresholds, and test scenarios
3. Update `benchmark_results/benchmark_results_template.json` with your metrics

### Extending Video Tests
1. Add video files to `demo_videos/`
2. Create metadata using `video_metadata_template.json`
3. Run `video_performance_test.py` with custom parameters

## Performance Tips

### For Mobile Devices:
- Use YOLOv11n for low-end devices
- Consider frame skipping for real-time processing
- Monitor thermal throttling during extended use
- Optimize memory management for longer sessions

### For Benchmarking:
- Test on actual target devices
- Run multiple iterations for statistical significance
- Include thermal and battery impact measurements
- Document all test conditions and environment

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce input resolution or use smaller model
2. **Low FPS**: Enable frame skipping or use optimized model
3. **Poor Accuracy**: Check input preprocessing and model parameters
4. **Battery Drain**: Use smaller model or reduce processing frequency

### Getting Help:
- Check logs in `benchmark_results/` directory
- Review `performance_benchmark.py` for detailed metrics
- Use `stress_test.py` to identify bottlenecks
- Consult Android-specific optimizations in `android_performance_test.py`

## Contributing

When adding new test assets:
1. Follow existing naming conventions
2. Update this README with new features
3. Include example usage code
4. Add appropriate metadata files
5. Test on multiple device classes

---

**Note**: This test suite is designed for comprehensive evaluation of human detection models on Android. Adjust configurations based on your specific requirements and target device capabilities.