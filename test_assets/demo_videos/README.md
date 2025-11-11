# Demo Videos for Human Detection Testing
# Sample video files for testing detection on video streams

## Video Test Suite

This directory contains information about video files used for testing human detection on video streams rather than static images.

### Recommended Video Characteristics

- **Duration**: 10-30 seconds for quick testing, 1-5 minutes for comprehensive testing
- **Resolution**: 720p to 1080p (mobile device resolution)
- **Frame Rate**: 30 FPS (standard mobile video)
- **Format**: MP4, MOV, AVI (H.264 codec recommended)
- **Content**: Mix of different human detection scenarios

### Video Categories

#### 1. Single Person Videos
- **File**: `single_person_walking.mp4`
- **Description**: One person walking in a straight line
- **Duration**: 15 seconds
- **Resolution**: 1080p
- **Scenarios**: Indoor/outdoor, different speeds

#### 2. Multiple People Videos  
- **File**: `group_conversation.mp4`
- **Description**: 3-5 people in conversation
- **Duration**: 20 seconds
- **Resolution**: 1080p
- **Scenarios**: Different distances, overlapping people

#### 3. Crowd Scene Video
- **File**: `crowd_movement.mp4`
- **Description**: Large group of people in motion
- **Duration**: 30 seconds
- **Resolution**: 1080p
- **Scenarios**: Dense crowd, varying motion patterns

#### 4. Motion and Action Video
- **File**: `people_running.mp4`
- **Description**: People running and fast motion
- **Duration**: 10 seconds
- **Resolution**: 1080p
- **Scenarios**: Motion blur, fast moving objects

#### 5. Challenging Conditions Video
- **File**: `low_light_night.mp4`
- **Description**: People in low light conditions
- **Duration**: 15 seconds
- **Resolution**: 1080p
- **Scenarios**: Night time, indoor low light, shadows

#### 6. Mixed Scenarios Video
- **File**: `mixed_activities.mp4`
- **Description**: Compilation of different activities
- **Duration**: 60 seconds
- **Resolution**: 1080p
- **Scenarios**: Various lighting, poses, and conditions

### Video Testing Metrics

#### Performance Metrics
- **Processing FPS**: Actual FPS the model can process
- **Detection Latency**: Time from frame capture to detection
- **Memory Usage**: Peak memory during video processing
- **Battery Impact**: Battery drain during video analysis

#### Accuracy Metrics
- **Frame-by-frame accuracy**: Accuracy per frame
- **Tracking stability**: How stable the detections are across frames
- **False positives**: Incorrect detections in video
- **Missed detections**: People not detected in frames

### Mobile-Specific Considerations

#### Performance Optimization
- **Frame Skipping**: Process every Nth frame for speed
- **Dynamic Resolution**: Adjust frame size based on device capability  
- **Memory Management**: Proper buffer management for video frames
- **Thermal Throttling**: Monitor CPU temperature during processing

#### User Experience
- **Real-time Preview**: Show detections as they happen
- **Tracking Trails**: Visual trails of detected people
- **Confidence Indicators**: Show detection confidence levels
- **Performance Display**: Show FPS and processing stats

### Video Generation Tools

#### For Creating Test Videos
1. **FFmpeg**: Command-line video processing
2. **OpenCV**: Python-based video capture and processing
3. **Mobile Apps**: Screen recording on Android devices
4. **Stock Video**: Royalty-free video databases

#### Sample FFmpeg Commands
```bash
# Extract 15-second clip
ffmpeg -i input_video.mp4 -t 15 -c copy clip_15s.mp4

# Resize to 720p
ffmpeg -i input_video.mp4 -vf scale=1280:720 output_720p.mp4

# Extract specific frames for testing
ffmpeg -i input_video.mp4 -vf fps=1 frames/frame_%03d.jpg
```

### Testing Scenarios

#### Real-time Processing
- Process video at original frame rate
- Measure actual vs. target FPS
- Check for dropped frames
- Monitor performance degradation over time

#### Offline Analysis
- Process video file completely
- Generate detection report
- Create annotated video with detections
- Measure total processing time

#### Continuous Processing
- Loop video for sustained testing
- Monitor memory leaks
- Test thermal throttling
- Measure battery drain over time

### Sample Video Metadata JSON

```json
{
  "video_info": {
    "filename": "single_person_walking.mp4",
    "duration_seconds": 15,
    "resolution": [1920, 1080],
    "fps": 30,
    "file_size_mb": 45.2,
    "codec": "H.264",
    "total_frames": 450,
    "has_audio": false,
    "scenarios": ["single_person", "walking", "outdoor"]
  },
  "ground_truth": {
    "total_people_appearances": 15,
    "scenes": [
      {
        "start_frame": 0,
        "end_frame": 450,
        "person_count": 1,
        "description": "Single person walking left to right"
      }
    ]
  },
  "expected_results": {
    "min_detections_per_second": 0.8,
    "max_processing_fps": 30,
    "target_accuracy": 0.85
  }
}
```

### Usage Instructions

1. **Download or create test videos** matching the criteria above
2. **Use video processing tools** to analyze performance
3. **Compare results** across different models and conditions
4. **Optimize parameters** based on video testing results
5. **Create annotated videos** showing detection results

### Note for Implementation

When implementing video processing in the Android app:
- Use SurfaceView for efficient video rendering
- Implement frame extraction for processing
- Consider using GPU acceleration (if available)
- Provide options for processing quality vs. speed
- Include performance monitoring in the UI