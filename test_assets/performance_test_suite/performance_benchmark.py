#!/usr/bin/env python3
"""
Performance Benchmark Suite for Human Detection
Measures inference time, accuracy, and memory usage
"""

import time
import psutil
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

class PerformanceBenchmark:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "python_version": os.sys.version
            },
            "benchmarks": []
        }
    
    def measure_inference_time(self, model, test_images, iterations=10):
        """Measure inference time for a model on test images"""
        print(f"Measuring inference time for {len(test_images)} images...")
        times = []
        
        for iteration in range(iterations):
            for image_path in test_images:
                start_time = time.perf_counter()
                # Simulate model inference (replace with actual inference call)
                _ = model.detect(image_path)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        return {
            "test_type": "inference_time",
            "images_tested": len(test_images),
            "iterations": iterations,
            "mean_time": np.mean(times),
            "median_time": np.median(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "std_dev": np.std(times),
            "fps": 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }
    
    def measure_memory_usage(self, model, test_images):
        """Measure memory usage during inference"""
        print("Measuring memory usage...")
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run inference on all images
        for image_path in test_images:
            _ = model.detect(image_path)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "test_type": "memory_usage",
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "memory_increase_mb": peak_memory - initial_memory,
            "images_tested": len(test_images)
        }
    
    def measure_accuracy(self, model, test_images, ground_truth_labels):
        """Measure detection accuracy"""
        print("Measuring accuracy...")
        correct_detections = 0
        total_detections = 0
        
        for i, image_path in enumerate(test_images):
            if i < len(ground_truth_labels):
                detections = model.detect(image_path)
                # Compare with ground truth (implement according to your use case)
                # This is a simplified example
                if len(detections) > 0:
                    correct_detections += 1
                total_detections += 1
        
        accuracy = (correct_detections / total_detections) * 100 if total_detections > 0 else 0
        
        return {
            "test_type": "accuracy",
            "correct_detections": correct_detections,
            "total_detections": total_detections,
            "accuracy_percentage": accuracy
        }
    
    def run_benchmark_suite(self, model, test_images_dir, output_file):
        """Run complete benchmark suite"""
        test_images = list(Path(test_images_dir).glob("*"))
        print(f"Found {len(test_images)} test images")
        
        # Run benchmarks
        inference_benchmark = self.measure_inference_time(model, test_images)
        memory_benchmark = self.measure_memory_usage(model, test_images)
        
        # Add benchmarks to results
        self.results["benchmarks"].extend([inference_benchmark, memory_benchmark])
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark results saved to {output_file}")
        return self.results

# Example usage
if __name__ == "__main__":
    # Mock model class for demonstration
    class MockHumanDetector:
        def __init__(self):
            self.model_name = "Mock YOLO Model"
        
        def detect(self, image_path):
            # Simulate inference time
            time.sleep(0.05)  # 50ms simulation
            return [{"confidence": 0.95, "bbox": [100, 100, 200, 300]}]
    
    # Run benchmark
    benchmark = PerformanceBenchmark()
    model = MockHumanDetector()
    test_dir = "../sample_test_images"
    output_file = "../benchmark_results/performance_results.json"
    
    results = benchmark.run_benchmark_suite(model, test_dir, output_file)
    print("Benchmark completed!")