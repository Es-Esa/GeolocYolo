#!/usr/bin/env python3
"""
Stress Testing Suite for Human Detection Models
Tests model performance under various stress conditions
"""

import time
import threading
import concurrent.futures
import random
from pathlib import Path
import json
from datetime import datetime

class StressTestSuite:
    def __init__(self):
        self.stress_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
    
    def concurrent_load_test(self, model, test_images, num_threads=4, requests_per_thread=10):
        """Test model under concurrent load"""
        print(f"Running concurrent load test with {num_threads} threads...")
        
        start_time = time.perf_counter()
        results = []
        
        def worker_thread(thread_id):
            thread_results = []
            for i in range(requests_per_thread):
                image_path = random.choice(test_images)
                thread_start = time.perf_counter()
                detection_result = model.detect(image_path)
                thread_end = time.perf_counter()
                
                thread_results.append({
                    "thread_id": thread_id,
                    "request_id": i,
                    "inference_time": thread_end - thread_start,
                    "detections_count": len(detection_result)
                })
            return thread_results
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        inference_times = [r["inference_time"] for r in results]
        
        test_result = {
            "test_name": "concurrent_load_test",
            "num_threads": num_threads,
            "requests_per_thread": requests_per_thread,
            "total_requests": num_threads * requests_per_thread,
            "total_time": total_time,
            "avg_inference_time": sum(inference_times) / len(inference_times),
            "min_inference_time": min(inference_times),
            "max_inference_time": max(inference_times),
            "requests_per_second": (num_threads * requests_per_thread) / total_time
        }
        
        self.stress_results["tests"].append(test_result)
        return test_result
    
    def memory_stress_test(self, model, test_images, max_memory_mb=500):
        """Test memory usage and limits"""
        print("Running memory stress test...")
        
        # Simulate memory-intensive operations
        memory_usage = []
        
        for image_path in test_images:
            # Simulate model loading and inference
            detection = model.detect(image_path)
            current_memory = self._get_memory_usage()
            memory_usage.append(current_memory)
            
            if current_memory > max_memory_mb:
                break
        
        return {
            "test_name": "memory_stress_test",
            "max_memory_limit_mb": max_memory_mb,
            "peak_memory_usage_mb": max(memory_usage),
            "images_processed": len(memory_usage),
            "memory_exceeded": max(memory_usage) > max_memory_mb
        }
    
    def batch_processing_test(self, model, test_images, batch_sizes=[1, 4, 8, 16]):
        """Test batch processing performance"""
        print("Running batch processing test...")
        
        batch_results = []
        
        for batch_size in batch_sizes:
            if batch_size > len(test_images):
                continue
                
            # Simulate batch inference
            batch_start = time.perf_counter()
            batch_results_local = []
            
            for i in range(0, len(test_images), batch_size):
                batch = test_images[i:i + batch_size]
                batch_detection = model.detect_batch(batch)
                batch_results_local.extend(batch_detection)
            
            batch_end = time.perf_counter()
            batch_time = batch_end - batch_start
            
            # Calculate per-image time
            per_image_time = batch_time / len(test_images)
            throughput = len(test_images) / batch_time
            
            batch_results.append({
                "batch_size": batch_size,
                "total_images": len(test_images),
                "batch_time": batch_time,
                "per_image_time": per_image_time,
                "throughput_fps": throughput
            })
        
        return {
            "test_name": "batch_processing_test",
            "batch_results": batch_results
        }
    
    def _get_memory_usage(self):
        """Get current memory usage in MB (simplified)"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def run_full_stress_suite(self, model, test_images_dir, output_file):
        """Run complete stress test suite"""
        test_images = list(Path(test_images_dir).glob("*"))
        
        print("Starting stress test suite...")
        
        # Run different stress tests
        concurrent_result = self.concurrent_load_test(model, test_images, num_threads=4)
        memory_result = self.memory_stress_test(model, test_images)
        batch_result = self.batch_processing_test(model, test_images)
        
        # Add all results
        self.stress_results["tests"].extend([concurrent_result, memory_result, batch_result])
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(self.stress_results, f, indent=2)
        
        print(f"Stress test results saved to {output_file}")
        return self.stress_results

# Example usage
if __name__ == "__main__":
    # Mock model class
    class MockHumanDetector:
        def __init__(self):
            self.model_name = "Mock YOLO Model"
        
        def detect(self, image_path):
            time.sleep(random.uniform(0.01, 0.05))  # Random inference time
            return [{"confidence": 0.95, "bbox": [100, 100, 200, 300]}]
        
        def detect_batch(self, batch):
            time.sleep(0.1)  # Simulate batch processing
            return [{"confidence": 0.95, "bbox": [100, 100, 200, 300]} for _ in batch]
    
    # Run stress tests
    stress_suite = StressTestSuite()
    model = MockHumanDetector()
    test_dir = "../sample_test_images"
    output_file = "../benchmark_results/stress_test_results.json"
    
    results = stress_suite.run_full_stress_suite(model, test_dir, output_file)
    print("Stress testing completed!")