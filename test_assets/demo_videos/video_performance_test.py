#!/usr/bin/env python3
"""
Video Performance Testing for Human Detection
Tests model performance on video streams and files
"""

import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import psutil

class VideoPerformanceTest:
    def __init__(self, model):
        self.model = model
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "video_tests": []
        }
    
    def process_video_file(self, video_path, output_dir, config):
        """Process a video file and measure performance"""
        print(f"Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames, {duration:.1f}s")
        
        # Initialize tracking
        detection_results = []
        processing_times = []
        frame_times = []
        
        # Setup output video writer
        output_video_path = output_dir / f"annotated_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.perf_counter()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if configured
            if config.get("process_every_n_frames", 1) > 1 and frame_idx % config["process_every_n_frames"] != 0:
                frame_idx += 1
                continue
            
            frame_start = time.perf_counter()
            
            # Perform detection
            detections = self.model.detect(frame)
            
            # Draw detections on frame
            for detection in detections:
                x, y, w, h = detection["bbox"]
                confidence = detection["confidence"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Write frame
            out.write(frame)
            
            # Store results
            frame_end = time.perf_counter()
            processing_time = frame_end - frame_start
            processing_times.append(processing_time)
            
            detection_results.append({
                "frame": frame_idx,
                "processing_time": processing_time,
                "num_detections": len(detections),
                "detections": detections
            })
            
            frame_times.append(frame_end - start_time)
            frame_idx += 1
            
            # Progress update
            if frame_idx % 30 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{frame_count} frames)")
        
        # Clean up
        cap.release()
        out.release()
        
        # Calculate metrics
        total_time = time.perf_counter() - start_time
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        min_processing_time = np.min(processing_times)
        actual_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        # Memory usage
        process = psutil.Process()
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Compile results
        test_result = {
            "video_info": {
                "filename": video_path.name,
                "resolution": [width, height],
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "processed_frames": len(detection_results)
            },
            "performance_metrics": {
                "total_processing_time": total_time,
                "average_processing_time_per_frame": avg_processing_time,
                "min_processing_time": min_processing_time,
                "max_processing_time": max_processing_time,
                "actual_fps": actual_fps,
                "peak_memory_usage_mb": peak_memory,
                "cpu_utilization": psutil.cpu_percent()
            },
            "detection_metrics": {
                "total_detections": sum(r["num_detections"] for r in detection_results),
                "average_detections_per_frame": np.mean([r["num_detections"] for r in detection_results]),
                "max_detections_per_frame": max(r["num_detections"] for r in detection_results)
            },
            "output_files": {
                "annotated_video": str(output_video_path),
                "detection_log": f"detection_log_{video_path.stem}.json"
            }
        }
        
        # Save detection log
        detection_log_path = output_dir / f"detection_log_{video_path.stem}.json"
        with open(detection_log_path, 'w') as f:
            json.dump(detection_results, f, indent=2)
        
        return test_result
    
    def real_time_video_test(self, camera_id=0, duration=30, config=None):
        """Test real-time video processing from camera"""
        if config is None:
            config = {"process_every_n_frames": 1}
        
        print(f"Starting real-time video test for {duration} seconds...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError("Could not open camera")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        start_time = time.perf_counter()
        frame_count = 0
        processing_times = []
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        while (time.perf_counter() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.perf_counter()
            
            # Process frame
            detections = self.model.detect(frame)
            
            # Draw detections
            for detection in detections:
                x, y, w, h = detection["bbox"]
                confidence = detection["confidence"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {confidence:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Real-time Human Detection', frame)
            
            # Store metrics
            frame_end = time.perf_counter()
            processing_time = frame_end - frame_start
            processing_times.append(processing_time)
            frame_count += 1
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate results
        total_time = time.perf_counter() - start_time
        peak_memory = process.memory_info().rss / 1024 / 1024
        
        test_result = {
            "test_type": "real_time_camera",
            "duration": duration,
            "frames_processed": frame_count,
            "average_fps": frame_count / total_time,
            "average_processing_time": np.mean(processing_times),
            "peak_memory_usage_mb": peak_memory,
            "initial_memory_mb": initial_memory,
            "memory_increase_mb": peak_memory - initial_memory
        }
        
        return test_result
    
    def run_video_test_suite(self, video_dir, output_dir, config):
        """Run complete video test suite"""
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
        
        print(f"Found {len(video_files)} video files to test")
        
        # Process each video
        for video_file in video_files:
            try:
                result = self.process_video_file(video_file, output_dir, config)
                self.test_results["video_tests"].append(result)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue
        
        # Save test results
        results_file = output_dir / "video_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"Video test results saved to {results_file}")
        return self.test_results

# Example usage
if __name__ == "__main__":
    # Mock model class
    class MockHumanDetector:
        def __init__(self):
            self.model_name = "Mock YOLO Model"
        
        def detect(self, frame):
            # Simulate detection
            height, width = frame.shape[:2]
            # Random detections for testing
            import random
            num_detections = random.randint(0, 3)
            detections = []
            
            for i in range(num_detections):
                x = random.randint(0, width - 100)
                y = random.randint(0, height - 100)
                w = random.randint(50, 100)
                h = random.randint(100, 200)
                confidence = random.uniform(0.5, 0.95)
                
                detections.append({
                    "bbox": [x, y, w, h],
                    "confidence": confidence,
                    "class": "person"
                })
            
            return detections
    
    # Run video tests
    model = MockHumanDetector()
    video_test = VideoPerformanceTest(model)
    
    # Configuration
    config = {
        "process_every_n_frames": 1,  # Process every frame
        "target_fps": 30
    }
    
    # Test video files (if they exist)
    video_directory = "../demo_videos"
    output_directory = "../benchmark_results"
    
    try:
        results = video_test.run_video_test_suite(video_directory, output_directory, config)
        print("Video performance testing completed!")
    except FileNotFoundError:
        print("Video directory not found. Please add video files to test.")
    
    # Test real-time camera (commented out for safety)
    # camera_result = video_test.real_time_video_test(duration=10)
    # print("Real-time camera test completed!")