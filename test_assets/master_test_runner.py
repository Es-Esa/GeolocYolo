#!/usr/bin/env python3
"""
Master Test Runner for Android Human Detection
Comprehensive test suite that runs all performance tests and generates reports
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add the test assets directory to path
current_dir = Path(__file__).parent
test_assets_dir = current_dir / "test_assets"
sys.path.append(str(test_assets_dir))

class MasterTestRunner:
    def __init__(self, model, output_dir="test_results"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.master_results = {
            "test_session_info": {
                "start_time": datetime.now().isoformat(),
                "tester": "Master Test Runner",
                "model_name": getattr(model, 'model_name', 'Unknown Model')
            },
            "test_results": {},
            "summary": {}
        }
    
    def run_image_benchmarks(self):
        """Run all image-based benchmarks"""
        print("🖼️  Running image benchmarks...")
        
        try:
            from performance_test_suite.performance_benchmark import PerformanceBenchmark
            from performance_test_suite.stress_test import StressTestSuite
            from performance_test_suite.android_performance_test import AndroidPerformanceTest
            
            # Initialize test suites
            benchmark = PerformanceBenchmark()
            stress_suite = StressTestSuite()
            android_test = AndroidPerformanceTest()
            
            test_images_dir = test_assets_dir / "sample_test_images"
            results = {}
            
            # Run performance benchmark
            perf_result = benchmark.run_benchmark_suite(
                model=self.model,
                test_images_dir=str(test_images_dir),
                output_file=str(self.output_dir / "image_performance.json")
            )
            results["performance_benchmark"] = perf_result
            
            # Run stress test
            stress_result = stress_suite.run_full_stress_suite(
                model=self.model,
                test_images_dir=str(test_images_dir),
                output_file=str(self.output_dir / "stress_test_results.json")
            )
            results["stress_test"] = stress_result
            
            # Run Android-specific tests
            android_result = android_test.run_android_benchmark_suite(
                model=self.model,
                test_images_dir=str(test_images_dir),
                output_file=str(self.output_dir / "android_performance.json")
            )
            results["android_test"] = android_result
            
            self.master_results["test_results"]["image_benchmarks"] = results
            print("✅ Image benchmarks completed")
            return results
            
        except ImportError as e:
            print(f"❌ Error importing test modules: {e}")
            return {}
    
    def run_video_tests(self):
        """Run video performance tests"""
        print("🎥 Running video tests...")
        
        try:
            from demo_videos.video_performance_test import VideoPerformanceTest
            
            video_test = VideoPerformanceTest(self.model)
            video_dir = test_assets_dir / "demo_videos"
            results = {}
            
            # Check if video files exist
            video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
            
            if video_files:
                config = {"process_every_n_frames": 1}
                video_result = video_test.run_video_test_suite(
                    video_dir=str(video_dir),
                    output_dir=str(self.output_dir),
                    config=config
                )
                results["video_files"] = video_result
            else:
                print("⚠️  No video files found, skipping video tests")
                results["video_files"] = {"status": "no_video_files"}
            
            # Optional: Test real-time camera (uncomment if you want to test)
            # camera_result = video_test.real_time_video_test(duration=10)
            # results["real_time_camera"] = camera_result
            
            self.master_results["test_results"]["video_tests"] = results
            print("✅ Video tests completed")
            return results
            
        except ImportError as e:
            print(f"❌ Error importing video test module: {e}")
            return {}
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("📊 Generating summary report...")
        
        # Collect key metrics from all tests
        summary = {
            "test_session": {
                "end_time": datetime.now().isoformat(),
                "duration": "See test_session_info.start_time for calculation",
                "total_tests_run": len(self.master_results["test_results"])
            },
            "performance_summary": {},
            "model_recommendations": {},
            "device_suitability": {}
        }
        
        # Extract performance metrics if available
        if "image_benchmarks" in self.master_results["test_results"]:
            img_results = self.master_results["test_results"]["image_benchmarks"]
            
            if "performance_benchmark" in img_results:
                perf_bench = img_results["performance_benchmark"]
                for benchmark in perf_bench.get("benchmarks", []):
                    if benchmark.get("test_type") == "inference_time":
                        summary["performance_summary"]["speed"] = {
                            "average_inference_time_ms": benchmark.get("mean_time", 0) * 1000,
                            "fps": benchmark.get("fps", 0)
                        }
                    elif benchmark.get("test_type") == "memory_usage":
                        summary["performance_summary"]["memory"] = {
                            "peak_memory_mb": benchmark.get("peak_memory_mb", 0)
                        }
        
        # Generate recommendations based on results
        speed_fps = summary["performance_summary"].get("speed", {}).get("fps", 0)
        memory_mb = summary["performance_summary"].get("memory", {}).get("peak_memory_mb", 0)
        
        if speed_fps > 30:
            summary["model_recommendations"]["real_time"] = "Excellent for real-time applications"
        elif speed_fps > 15:
            summary["model_recommendations"]["real_time"] = "Good for near real-time applications"
        else:
            summary["model_recommendations"]["real_time"] = "Suitable for batch processing only"
        
        if memory_mb < 200:
            summary["device_suitability"]["low_end"] = "Suitable"
        elif memory_mb < 500:
            summary["device_suitability"]["mid_range"] = "Suitable"
        else:
            summary["device_suitability"]["high_end"] = "Recommended"
        
        self.master_results["summary"] = summary
        return summary
    
    def save_master_results(self):
        """Save all results to master file"""
        # Save master results
        master_file = self.output_dir / "master_test_results.json"
        with open(master_file, 'w') as f:
            json.dump(self.master_results, f, indent=2)
        
        print(f"📁 Master results saved to: {master_file}")
        return str(master_file)
    
    def run_complete_test_suite(self):
        """Run the complete test suite"""
        print("🚀 Starting comprehensive test suite...")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Model: {getattr(self.model, 'model_name', 'Unknown')}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Run all tests
        self.run_image_benchmarks()
        self.run_video_tests()
        
        # Generate summary
        self.generate_summary_report()
        
        # Save results
        master_file = self.save_master_results()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("-" * 50)
        print("🎉 Test suite completed!")
        print(f"⏱️  Total duration: {duration:.1f} seconds")
        print(f"📁 Results saved to: {self.output_dir}")
        print("📋 Generated files:")
        
        # List generated files
        for file_path in self.output_dir.glob("*.json"):
            print(f"   - {file_path.name}")
        
        return self.master_results

# Mock model for demonstration
class MockHumanDetectionModel:
    def __init__(self):
        self.model_name = "Mock YOLO Model for Testing"
    
    def detect(self, image):
        """Mock detection method"""
        import random
        import time
        time.sleep(random.uniform(0.02, 0.08))  # Simulate inference time
        
        # Simulate random detections
        height, width = image.shape[:2] if len(image.shape) == 3 else (480, 640)
        num_detections = random.randint(0, 3)
        detections = []
        
        for i in range(num_detections):
            x = random.randint(0, width - 100)
            y = random.randint(0, height - 100)
            w = random.randint(50, 100)
            h = random.randint(100, 200)
            confidence = random.uniform(0.6, 0.95)
            
            detections.append({
                "bbox": [x, y, w, h],
                "confidence": confidence,
                "class": "person"
            })
        
        return detections
    
    def detect_batch(self, images):
        """Mock batch detection"""
        return [self.detect(img) for img in images]

def main():
    parser = argparse.ArgumentParser(description="Master Test Runner for Human Detection")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for results")
    parser.add_argument("--model", default="mock", help="Model type (mock or your_model)")
    parser.add_argument("--skip-video", action="store_true", help="Skip video tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    # Initialize model (mock for demonstration)
    if args.model == "mock":
        model = MockHumanDetectionModel()
        print("🔧 Using mock model for testing")
    else:
        # Import your actual model here
        # from your_model_module import YourHumanDetectionModel
        # model = YourHumanDetectionModel()
        model = MockHumanDetectionModel()
        print("🔧 Using custom model (using mock for demo)")
    
    # Initialize test runner
    test_runner = MasterTestRunner(model, args.output_dir)
    
    try:
        # Run complete test suite
        results = test_runner.run_complete_test_suite()
        
        # Print quick summary
        print("\n" + "="*50)
        print("📊 QUICK SUMMARY")
        print("="*50)
        
        summary = results.get("summary", {})
        perf_summary = summary.get("performance_summary", {})
        
        if "speed" in perf_summary:
            fps = perf_summary["speed"].get("fps", 0)
            print(f"🚀 Performance: {fps:.1f} FPS")
        
        if "memory" in perf_summary:
            memory = perf_summary["memory"].get("peak_memory_mb", 0)
            print(f"💾 Memory Usage: {memory:.1f} MB")
        
        recommendations = summary.get("model_recommendations", {})
        if "real_time" in recommendations:
            print(f"⚡ Real-time: {recommendations['real_time']}")
        
        print("✅ Test suite completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running test suite: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())