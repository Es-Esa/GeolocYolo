# YOLOv11n Android Performance Benchmarking Guide

## Overview

This comprehensive guide provides methodologies, tools, and best practices for benchmarking YOLOv11n TFLite models on Android devices. It covers performance metrics, testing frameworks, automated benchmarking, and analysis techniques to ensure optimal model deployment.

## Table of Contents

1. [Performance Metrics](#performance-metrics)
2. [Testing Infrastructure](#testing-infrastructure)
3. [Automated Benchmarking Framework](#automated-benchmarking-framework)
4. [Device Matrix Testing](#device-matrix-testing)
5. [Comparison Testing](#comparison-testing)
6. [Thermal and Power Analysis](#thermal-and-power-analysis)
7. [Memory and Resource Analysis](#memory-and-resource-analysis)
8. [Automated Reporting](#automated-reporting)
9. [Continuous Performance Monitoring](#continuous-performance-monitoring)

## Performance Metrics

### Core Performance Metrics

| Metric | Definition | Target | Measurement Method |
|--------|------------|--------|-------------------|
| **End-to-End Latency** | Time from frame capture to detection rendering | <50ms for real-time | High-resolution timestamps |
| **Inference Latency** | Model execution time only | <20ms for YOLOv11n | TFLite timing APIs |
| **FPS (Frames Per Second)** | Detection rate | 20-30 FPS typical | Frame counter with timing |
| **Memory Usage** | Peak memory during inference | <100MB for YOLOv11n | Android Profiler |
| **Model Size** | Storage requirements | <5MB for INT8 | File size analysis |
| **Battery Consumption** | Power usage per inference | <50mW average | PowerProfiler integration |
| **Accuracy (mAP)** | Detection quality | >35% for YOLOv11n | Validation dataset |
| **Cold Start Time** | Model loading time | <2 seconds | Profiling system |

### Advanced Metrics

| Metric | Purpose | Collection Method |
|--------|---------|-------------------|
| **P95/P99 Latency** | Performance consistency | Percentile analysis |
| **Thermal Throttling** | Sustained performance | Thermal state monitoring |
| **Delegate Performance** | Hardware acceleration effectiveness | Comparison across delegates |
| **Frame Skipping Rate** | Real-time adaptability | Queue monitoring |
| **Memory Allocation Rate** | GC pressure | Allocation tracking |

## Testing Infrastructure

### Test Environment Setup

```bash
# Required tools and libraries
adb install -r app-debug.apk
adb shell settings put global device_provisioning_mobile_data 1

# Enable profiling
adb shell setprop debug.performance.timing 1
adb shell setprop debug.nn.trace 1
```

### Benchmark Test Suite Structure

```kotlin
// Main benchmarking coordinator
class YOLOv11nBenchmarkSuite {
    
    data class BenchmarkConfig(
        val modelVariants: List<ModelVariant>,
        val testDevices: List<TestDevice>,
        val testScenarios: List<TestScenario>,
        val duration: Duration = Duration.ofMinutes(5)
    )
    
    data class ModelVariant(
        val name: String,
        val modelPath: String,
        val quantization: QuantizationType,
        val inputSize: Int
    )
    
    data class TestDevice(
        val name: String,
        val model: String,
        val androidVersion: Int,
        val specifications: DeviceSpecs
    )
    
    enum class QuantizationType { FP32, FP16, INT8 }
    enum class TestScenario { REAL_TIME, POWER_SAVER, HIGH_ACCURACY, THERMAL_STRESS }
}
```

### Test Data Preparation

```kotlin
object TestDataManager {
    
    // Standard test images for consistency
    private val testImageCategories = mapOf(
        "single_person" to listOf(
            "test_images/person_close.jpg",
            "test_images/person_medium.jpg", 
            "test_images/person_far.jpg"
        ),
        "crowded_scene" to listOf(
            "test_images/crowd_sparse.jpg",
            "test_images/crowd_medium.jpg",
            "test_images/crowd_dense.jpg"
        ),
        "challenging_conditions" to listOf(
            "test_images/low_light.jpg",
            "test_images/high_light.jpg",
            "test_images/occluded.jpg"
        )
    )
    
    // Generate synthetic test data
    fun generateSyntheticTestData(count: Int): List<Bitmap> {
        val syntheticData = mutableListOf<Bitmap>()
        
        repeat(count) { index ->
            val bitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            
            // Add random human-like shapes
            val personPaint = Paint().apply { 
                color = Color.rgb(100, 150, 200)
                style = Paint.Style.FILL
            }
            
            val random = Random(index.toLong())
            val personCount = random.nextInt(3) + 1
            
            repeat(personCount) { personIndex ->
                val x = random.nextInt(480) + 80f
                val y = random.nextInt(480) + 80f
                val width = random.nextInt(100) + 50f
                val height = random.nextInt(150) + 100f
                
                canvas.drawRect(x, y, x + width, y + height, personPaint)
            }
            
            syntheticData.add(bitmap)
        }
        
        return syntheticData
    }
}
```

## Automated Benchmarking Framework

### Core Benchmark Runner

```kotlin
class YOLOv11nPerformanceBenchmark {
    
    private val benchmarkResults = mutableListOf<BenchmarkResult>()
    private lateinit var currentDevice: TestDevice
    private lateinit var currentModel: ModelVariant
    
    suspend fun runComprehensiveBenchmark(config: BenchmarkConfig): BenchmarkReport {
        val report = BenchmarkReport()
        
        for (device in config.testDevices) {
            currentDevice = device
            setupDeviceForTesting(device)
            
            for (model in config.modelVariants) {
                currentModel = model
                
                for (scenario in config.testScenarios) {
                    val result = runScenario(model, device, scenario, config.duration)
                    report.addResult(result)
                }
            }
        }
        
        return report
    }
    
    private suspend fun runScenario(
        model: ModelVariant,
        device: TestDevice,
        scenario: TestScenario,
        duration: Duration
    ): BenchmarkResult {
        
        val detector = createDetector(model)
        val frameProcessor = FrameProcessor()
        val performanceMonitor = PerformanceMonitor()
        
        val measurements = mutableListOf<PerformanceMeasurement>()
        val startTime = System.currentTimeMillis()
        
        // Initialize performance monitoring
        performanceMonitor.startMonitoring()
        
        when (scenario) {
            TestScenario.REAL_TIME -> {
                measurements.addAll(runRealTimeTest(detector, frameProcessor, duration))
            }
            TestScenario.POWER_SAVER -> {
                measurements.addAll(runPowerSaverTest(detector, frameProcessor, duration))
            }
            TestScenario.HIGH_ACCURACY -> {
                measurements.addAll(runHighAccuracyTest(detector, frameProcessor, duration))
            }
            TestScenario.THERMAL_STRESS -> {
                measurements.addAll(runThermalStressTest(detector, frameProcessor, duration))
            }
        }
        
        performanceMonitor.stopMonitoring()
        
        return BenchmarkResult(
            model = model,
            device = device,
            scenario = scenario,
            measurements = measurements,
            performanceSummary = analyzeMeasurements(measurements)
        )
    }
    
    private suspend fun runRealTimeTest(
        detector: YOLOv11nHumanDetector,
        frameProcessor: FrameProcessor,
        duration: Duration
    ): List<PerformanceMeasurement> {
        
        val measurements = mutableListOf<PerformanceMeasurement>()
        val testImages = TestDataManager.generateSyntheticTestData(100)
        var imageIndex = 0
        
        val endTime = System.currentTimeMillis() + duration.toMillis()
        
        while (System.currentTimeMillis() < endTime) {
            val image = testImages[imageIndex % testImages.size]
            imageIndex++
            
            val measurement = measureSingleInference(detector, image)
            measurements.add(measurement)
            
            // Simulate real-time processing rate
            delay((1000 / 30).toLong()) // 30 FPS target
        }
        
        return measurements
    }
}
```

### Performance Measurement Utilities

```kotlin
class PerformanceMonitor {
    
    private val measurements = mutableListOf<PerformanceMetric>()
    private var isMonitoring = false
    
    data class PerformanceMetric(
        val timestamp: Long,
        val cpuUsage: Float,
        val memoryUsage: Long,
        val temperature: Float?,
        val batteryLevel: Int
    )
    
    fun startMonitoring() {
        isMonitoring = true
        
        CoroutineScope(Dispatchers.Default).launch {
            while (isMonitoring) {
                val metric = collectCurrentMetrics()
                measurements.add(metric)
                delay(100) // Sample every 100ms
            }
        }
    }
    
    fun stopMonitoring() {
        isMonitoring = false
    }
    
    private fun collectCurrentMetrics(): PerformanceMetric {
        return PerformanceMetric(
            timestamp = System.currentTimeMillis(),
            cpuUsage = getCpuUsage(),
            memoryUsage = getMemoryUsage(),
            temperature = getDeviceTemperature(),
            batteryLevel = getBatteryLevel()
        )
    }
    
    private fun getCpuUsage(): Float {
        // Implementation to read CPU usage from system
        return try {
            val stats = StringTokenizer(
                File("/proc/stat").readText()
            ).let { tokenizer ->
                repeat(8) { tokenizer.nextToken() } // Skip first 8 tokens
                tokenizer.nextToken() // idle time
            }
            
            val totalCpuTime = File("/proc/stat").readLines()
                .first { it.startsWith("cpu ") }
                .split("\\s+")
                .drop(1)
                .map { it.toLong() }
                .sum()
            
            val idleTime = stats.toLong()
            val activeTime = totalCpuTime - idleTime
            (activeTime.toFloat() / totalCpuTime) * 100
        } catch (e: Exception) {
            0f
        }
    }
    
    private fun getMemoryUsage(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }
    
    private fun getDeviceTemperature(): Float? {
        return try {
            val thermalPath = "/sys/class/thermal/thermal_zone0/temp"
            if (File(thermalPath).exists()) {
                File(thermalPath).readText().trim().toFloat() / 1000.0f // Convert to Celsius
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }
    
    private fun getBatteryLevel(): Int {
        return try {
            val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            batteryIntent?.let { intent ->
                val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                if (level != -1 && scale != -1) {
                    ((level.toFloat() / scale.toFloat()) * 100).toInt()
                } else -1
            } ?: -1
        } catch (e: Exception) {
            -1
        }
    }
}
```

### Detailed Performance Profiling

```kotlin
class DetailedProfiler {
    
    data class InferenceProfile(
        val totalTime: Long,
        val preprocessTime: Long,
        val inferenceTime: Long,
        val postprocessTime: Long,
        val renderTime: Long,
        val memoryPeak: Long,
        val allocations: Int
    )
    
    fun profileInference(
        detector: YOLOv11nHumanDetector,
        testImage: Bitmap,
        iterations: Int = 100
    ): InferenceProfile {
        
        val profiles = mutableListOf<InferenceProfile>()
        
        repeat(iterations) { iteration ->
            val profile = profileSingleInference(detector, testImage)
            profiles.add(profile)
        }
        
        return analyzeProfiles(profiles)
    }
    
    private fun profileSingleInference(
        detector: YOLOv11nHumanDetector,
        testImage: Bitmap
    ): InferenceProfile {
        
        val startMemory = getMemoryUsage()
        val initialAllocations = getAllocationCount()
        
        val totalStart = System.nanoTime()
        
        // Preprocess
        val preprocessStart = System.nanoTime()
        val preprocessed = detector.preprocessImage(testImage)
        val preprocessEnd = System.nanoTime()
        
        // Inference
        val inferenceStart = System.nanoTime()
        val results = detector.detectHumans(preprocessed)
        val inferenceEnd = System.nanoTime()
        
        // Post-process
        val postprocessStart = System.nanoTime()
        val processedResults = detector.postprocessResults(results)
        val postprocessEnd = System.nanoTime()
        
        // Render (simulated)
        val renderStart = System.nanoTime()
        val rendered = renderResults(processedResults)
        val renderEnd = System.nanoTime()
        
        val totalEnd = System.nanoTime()
        
        val endMemory = getMemoryUsage()
        val finalAllocations = getAllocationCount()
        
        return InferenceProfile(
            totalTime = totalEnd - totalStart,
            preprocessTime = preprocessEnd - preprocessStart,
            inferenceTime = inferenceEnd - inferenceStart,
            postprocessTime = postprocessEnd - postprocessStart,
            renderTime = renderEnd - renderStart,
            memoryPeak = endMemory - startMemory,
            allocations = finalAllocations - initialAllocations
        )
    }
    
    private fun getMemoryUsage(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }
    
    private fun getAllocationCount(): Int {
        // Use Android Profiler API or DDMS for allocation counting
        return 0 // Placeholder
    }
}
```

## Device Matrix Testing

### Device Capability Detection

```kotlin
object DeviceCapabilityAnalyzer {
    
    data class DeviceCapabilities(
        val model: String,
        val manufacturer: String,
        val androidVersion: Int,
        val apiLevel: Int,
        val hasNpu: Boolean,
        val hasGpu: Boolean,
        val hasGpuDriver: Boolean,
        val cpuArchitecture: String,
        val cpuCores: Int,
        val ramSize: Long,
        val supportedDelegates: List<DelegateType>
    )
    
    fun analyzeDeviceCapabilities(context: Context): DeviceCapabilities {
        val packageManager = context.packageManager
        
        return DeviceCapabilities(
            model = Build.MODEL,
            manufacturer = Build.MANUFACTURER,
            androidVersion = Build.VERSION.SDK_INT,
            apiLevel = Build.VERSION_CODES.currentApi(),
            hasNpu = detectNPU(),
            hasGpu = detectGPU(),
            hasGpuDriver = detectGpuDriver(),
            cpuArchitecture = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown",
            cpuCores = Runtime.getRuntime().availableProcessors(),
            ramSize = getTotalRAM(),
            supportedDelegates = detectSupportedDelegates()
        )
    }
    
    private fun detectNPU(): Boolean {
        // Check for NPU support through various methods
        return try {
            // Method 1: Check system properties
            val npuProp = SystemProperties.get("ro.hardware.npu", "")
            npuProp.isNotEmpty()
        } catch (e: Exception) {
            false
        }
    }
    
    private fun detectGPU(): Boolean {
        return try {
            val gpuDelegateHelper = Class.forName("org.tensorflow.lite.gpu.GpuDelegateHelper")
            val method = gpuDelegateHelper.getMethod("isGpuDelegateSupported")
            method.invoke(null) as Boolean
        } catch (e: Exception) {
            false
        }
    }
    
    private fun getTotalRAM(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.totalMem
    }
    
    private fun detectSupportedDelegates(): List<DelegateType> {
        val delegates = mutableListOf<DelegateType>()
        
        delegates.add(DelegateType.CPU) // Always supported
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            delegates.add(DelegateType.NNAPI)
        }
        
        if (detectGPU()) {
            delegates.add(DelegateType.GPU)
        }
        
        return delegates
    }
}
```

### Automated Device Testing Framework

```kotlin
class DeviceMatrixTestRunner {
    
    data class TestMatrix(
        val deviceCapabilities: List<DeviceCapabilities>,
        val modelVariants: List<ModelVariant>,
        val testScenarios: List<TestScenario>
    )
    
    suspend fun runDeviceMatrixTest(testMatrix: TestMatrix): MatrixTestReport {
        val report = MatrixTestReport()
        
        for (capabilities in testMatrix.deviceCapabilities) {
            val deviceReport = runTestsOnDevice(capabilities, testMatrix)
            report.addDeviceReport(deviceReport)
        }
        
        return report
    }
    
    private suspend fun runTestsOnDevice(
        capabilities: DeviceCapabilities,
        testMatrix: TestMatrix
    ): DeviceTestReport {
        
        val deviceReport = DeviceTestReport(capabilities)
        
        for (model in testMatrix.modelVariants) {
            if (!isModelSupportedOnDevice(model, capabilities)) continue
            
            for (scenario in testMatrix.testScenarios) {
                if (!isScenarioSupportedOnDevice(scenario, capabilities)) continue
                
                val result = runTestOnDevice(model, scenario, capabilities)
                deviceReport.addResult(result)
            }
        }
        
        return deviceReport
    }
    
    private fun isModelSupportedOnDevice(
        model: ModelVariant,
        capabilities: DeviceCapabilities
    ): Boolean {
        return when (model.quantization) {
            QuantizationType.INT8 -> capabilities.hasNpu || capabilities.hasGpu
            else -> true // FP16 and FP32 are universally supported
        }
    }
}
```

## Comparison Testing

### A/B Testing Framework

```kotlin
class ABTestFramework {
    
    data class ABTestConfig(
        val modelA: ModelVariant,
        val modelB: ModelVariant,
        val trafficSplit: Float = 0.5f, // 50/50 split
        val duration: Duration = Duration.ofDays(1),
        val metrics: List<Metric> = listOf(
            Metric.LATENCY,
            Metric.ACCURACY,
            Metric.MEMORY,
            Metric.BATTERY
        )
    )
    
    data class ABTestResult(
        val modelAResults: List<PerformanceMeasurement>,
        val modelBResults: List<PerformanceMeasurement>,
        val statisticalAnalysis: StatisticalAnalysis,
        val recommendation: String
    )
    
    suspend fun runABTest(config: ABTestConfig): ABTestResult {
        val modelAResults = mutableListOf<PerformanceMeasurement>()
        val modelBResults = mutableListOf<PerformanceMeasurement>()
        
        val startTime = System.currentTimeMillis()
        val endTime = startTime + config.duration.toMillis()
        
        while (System.currentTimeMillis() < endTime) {
            val testImage = getRandomTestImage()
            
            // Alternate between models based on traffic split
            val useModelA = Random().nextFloat() < config.trafficSplit
            
            if (useModelA) {
                val detectorA = createDetector(config.modelA)
                val result = measurePerformance(detectorA, testImage)
                modelAResults.add(result)
            } else {
                val detectorB = createDetector(config.modelB)
                val result = measurePerformance(detectorB, testImage)
                modelBResults.add(result)
            }
        }
        
        val analysis = analyzeABTestResults(modelAResults, modelBResults, config.metrics)
        val recommendation = generateRecommendation(analysis)
        
        return ABTestResult(
            modelAResults = modelAResults,
            modelBResults = modelBResults,
            statisticalAnalysis = analysis,
            recommendation = recommendation
        )
    }
    
    private fun analyzeABTestResults(
        modelAResults: List<PerformanceMeasurement>,
        modelBResults: List<PerformanceMeasurement>,
        metrics: List<Metric>
    ): StatisticalAnalysis {
        
        val analysis = StatisticalAnalysis()
        
        for (metric in metrics) {
            val modelAValues = modelAResults.map { it.getMetricValue(metric) }
            val modelBValues = modelBResults.map { it.getMetricValue(metric) }
            
            val testResult = performTTest(modelAValues, modelBValues)
            val effectSize = calculateEffectSize(modelAValues, modelBValues)
            
            analysis.addMetricAnalysis(metric, testResult, effectSize)
        }
        
        return analysis
    }
    
    private fun performTTest(groupA: List<Double>, groupB: List<Double>): TTestResult {
        val tStatistic = calculateTStatistic(groupA, groupB)
        val pValue = calculatePValue(tStatistic, groupA.size + groupB.size - 2)
        val confidenceInterval = calculateConfidenceInterval(groupA, groupB, 0.95)
        
        return TTestResult(
            tStatistic = tStatistic,
            pValue = pValue,
            confidenceInterval = confidenceInterval,
            isSignificant = pValue < 0.05
        )
    }
}
```

### Performance Comparison Dashboard

```kotlin
class PerformanceComparisonDashboard {
    
    fun generateComparisonReport(
        baselineResults: BenchmarkResult,
        comparisonResults: List<BenchmarkResult>
    ): ComparisonReport {
        
        val report = ComparisonReport()
        
        // Key metrics comparison
        val metricsComparison = compareKeyMetrics(baselineResults, comparisonResults)
        report.metricsComparison = metricsComparison
        
        // Performance gains/losses
        val performanceGains = calculatePerformanceGains(baselineResults, comparisonResults)
        report.performanceGains = performanceGains
        
        // Recommendations
        report.recommendations = generateRecommendations(performanceGains)
        
        return report
    }
    
    private fun compareKeyMetrics(
        baseline: BenchmarkResult,
        comparisons: List<BenchmarkResult>
    ): MetricsComparison {
        
        val baselineMetrics = baseline.performanceSummary
        
        return MetricsComparison(
            latencyComparison = comparisons.map { comparison ->
                MetricComparison(
                    model = comparison.model,
                    improvement = ((baselineMetrics.avgLatencyMs - comparison.performanceSummary.avgLatencyMs) 
                                 / baselineMetrics.avgLatencyMs * 100)
                )
            },
            memoryComparison = comparisons.map { comparison ->
                MetricComparison(
                    model = comparison.model,
                    improvement = ((baselineMetrics.memoryUsageMb - comparison.performanceSummary.memoryUsageMb) 
                                 / baselineMetrics.memoryUsageMb * 100)
                )
            },
            accuracyComparison = comparisons.map { comparison ->
                MetricComparison(
                    model = comparison.model,
                    improvement = ((comparison.performanceSummary.accuracyMap - baselineMetrics.accuracyMap) 
                                 / baselineMetrics.accuracyMap * 100)
                )
            }
        )
    }
}
```

## Thermal and Power Analysis

### Thermal Monitoring

```kotlin
class ThermalMonitor {
    
    data class ThermalProfile(
        val ambientTemperature: Float,
        val deviceTemperature: Float,
        val thermalState: ThermalState,
        val throttlingEvents: Int,
        val performanceDegradation: Float
    )
    
    enum class ThermalState {
        NORMAL, WARM, HOT, THROTTLING
    }
    
    fun monitorThermalBehavior(
        duration: Duration,
        detector: YOLOv11nHumanDetector
    ): ThermalProfile {
        
        val temperatureReadings = mutableListOf<Float>()
        val throttlingEvents = mutableListOf<Long>()
        var maxTemperature = Float.MIN_VALUE
        var minTemperature = Float.MAX_VALUE
        
        val startTime = System.currentTimeMillis()
        val endTime = startTime + duration.toMillis()
        
        while (System.currentTimeMillis() < endTime) {
            val temperature = readDeviceTemperature()
            temperatureReadings.add(temperature)
            
            maxTemperature = maxOf(maxTemperature, temperature)
            minTemperature = minOf(minTemperature, temperature)
            
            // Detect throttling
            if (temperature > THERMAL_THRESHOLD) {
                throttlingEvents.add(System.currentTimeMillis())
            }
            
            // Run inference to generate heat
            val testImage = generateHeatTestImage()
            detector.detectHumans(testImage, 0)
            
            delay(1000) // Sample every second
        }
        
        return ThermalProfile(
            ambientTemperature = temperatureReadings.firstOrNull() ?: 0f,
            deviceTemperature = maxTemperature,
            thermalState = determineThermalState(maxTemperature),
            throttlingEvents = throttlingEvents.size,
            performanceDegradation = calculatePerformanceDegradation(throttlingEvents)
        )
    }
    
    private fun readDeviceTemperature(): Float {
        return try {
            val thermalZones = listOf(
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp",
                "/sys/class/thermal/thermal_zone2/temp"
            )
            
            for (zone in thermalZones) {
                if (File(zone).exists()) {
                    return File(zone).readText().trim().toFloat() / 1000.0f
                }
            }
            
            0f
        } catch (e: Exception) {
            0f
        }
    }
    
    private fun determineThermalState(maxTemp: Float): ThermalState {
        return when {
            maxTemp > 80f -> ThermalState.THROTTLING
            maxTemp > 70f -> ThermalState.HOT
            maxTemp > 60f -> ThermalState.WARM
            else -> ThermalState.NORMAL
        }
    }
    
    companion object {
        private const val THERMAL_THRESHOLD = 70f // Celsius
    }
}
```

### Power Consumption Analysis

```kotlin
class PowerAnalyzer {
    
    data class PowerProfile(
        val averagePowerConsumption: Float, // Watts
        val powerPerInference: Float,       // Joules
        val batteryDrainRate: Float,        // % per hour
        val energyEfficiency: Float         // Inferences per joule
    )
    
    suspend fun analyzePowerConsumption(
        detector: YOLOv11nHumanDetector,
        duration: Duration
    ): PowerProfile {
        
        val initialBatteryLevel = getBatteryLevel()
        val powerMeasurements = mutableListOf<Float>()
        val inferenceCount = AtomicInteger(0)
        
        val startTime = System.currentTimeMillis()
        val endTime = startTime + duration.toMillis()
        
        // Start power monitoring in background
        val powerMonitorJob = CoroutineScope(Dispatchers.Default).launch {
            while (System.currentTimeMillis() < endTime) {
                val power = measureCurrentPower()
                powerMeasurements.add(power)
                delay(100) // Sample every 100ms
            }
        }
        
        // Run inference workload
        val inferenceJob = CoroutineScope(Dispatchers.Default).launch {
            while (System.currentTimeMillis() < endTime) {
                val testImage = generateTestImage()
                detector.detectHumans(testImage, 0)
                inferenceCount.incrementAndGet()
            }
        }
        
        // Wait for completion
        powerMonitorJob.join()
        inferenceJob.join()
        
        val finalBatteryLevel = getBatteryLevel()
        val batteryDrain = initialBatteryLevel - finalBatteryLevel
        
        val totalTime = duration.toMillis() / 1000.0f // seconds
        val totalEnergy = powerMeasurements.average() * totalTime // Joules
        
        return PowerProfile(
            averagePowerConsumption = powerMeasurements.average(),
            powerPerInference = totalEnergy / inferenceCount.get(),
            batteryDrainRate = (batteryDrain / totalTime) * 3600, // % per hour
            energyEfficiency = inferenceCount.get().toFloat() / totalEnergy
        )
    }
    
    private fun measureCurrentPower(): Float {
        // Implementation depends on available power monitoring tools
        // This is a placeholder - actual implementation would depend on
        // available power measurement interfaces on the device
        return try {
            // Example: Read from power supply interface
            // val powerPath = "/sys/class/power_supply/battery/power_now"
            // File(powerPath).readText().trim().toFloat()
            Random.nextFloat() * 5.0f // Placeholder value
        } catch (e: Exception) {
            0f
        }
    }
}
```

## Memory and Resource Analysis

### Memory Profiling

```kotlin
class MemoryProfiler {
    
    data class MemoryProfile(
        val peakMemoryUsage: Long,
        val averageMemoryUsage: Long,
        val memoryGrowth: Long,
        val gcEvents: Int,
        val allocationRate: Float,
        val memoryEfficiency: Float
    )
    
    fun profileMemoryUsage(
        detector: YOLOv11nHumanDetector,
        duration: Duration,
        testImages: List<Bitmap>
    ): MemoryProfile {
        
        val memoryReadings = mutableListOf<Long>()
        var peakMemory = Long.MIN_VALUE
        var minMemory = Long.MAX_VALUE
        val gcEvents = AtomicInteger(0)
        val allocations = AtomicInteger(0)
        
        // Setup GC monitoring
        val gcListener = { event: GCEvent ->
            gcEvents.incrementAndGet()
        }
        
        val startMemory = getCurrentMemory()
        
        val startTime = System.currentTimeMillis()
        val endTime = startTime + duration.toMillis()
        var imageIndex = 0
        
        // Memory monitoring loop
        CoroutineScope(Dispatchers.IO).launch {
            while (System.currentTimeMillis() < endTime) {
                val currentMemory = getCurrentMemory()
                memoryReadings.add(currentMemory)
                peakMemory = maxOf(peakMemory, currentMemory)
                minMemory = minOf(minMemory, currentMemory)
                delay(50)
            }
        }
        
        // Inference workload
        while (System.currentTimeMillis() < endTime) {
            val image = testImages[imageIndex % testImages.size]
            imageIndex++
            
            // Count allocations before inference
            val allocationsBefore = getAllocationCount()
            
            val result = detector.detectHumans(image, 0)
            
            // Count allocations after inference
            val allocationsAfter = getAllocationCount()
            allocations.addAndGet(allocationsAfter - allocationsBefore)
        }
        
        return MemoryProfile(
            peakMemoryUsage = peakMemory,
            averageMemoryUsage = memoryReadings.average().toLong(),
            memoryGrowth = peakMemory - startMemory,
            gcEvents = gcEvents.get(),
            allocationRate = allocations.get().toFloat() / (duration.toMillis() / 1000),
            memoryEfficiency = calculateMemoryEfficiency(peakMemory, testImages.size)
        )
    }
    
    private fun getCurrentMemory(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }
    
    private fun getAllocationCount(): Int {
        // This would use Android's allocation tracking APIs
        // For now, returning a placeholder
        return 0
    }
    
    private fun calculateMemoryEfficiency(peakMemory: Long, inferenceCount: Int): Float {
        return (peakMemory / 1024 / 1024).toFloat() / inferenceCount // MB per inference
    }
}
```

## Automated Reporting

### Comprehensive Report Generation

```kotlin
class BenchmarkReportGenerator {
    
    fun generateComprehensiveReport(benchmarkResults: List<BenchmarkResult>): String {
        val report = StringBuilder()
        
        report.append("# YOLOv11n Android Performance Benchmark Report\n\n")
        report.append("Generated: ${Date()}\n\n")
        
        // Executive Summary
        report.append(generateExecutiveSummary(benchmarkResults))
        
        // Detailed Results
        report.append(generateDetailedResults(benchmarkResults))
        
        // Device Comparison
        report.append(generateDeviceComparison(benchmarkResults))
        
        // Model Comparison
        report.append(generateModelComparison(benchmarkResults))
        
        // Performance Analysis
        report.append(generatePerformanceAnalysis(benchmarkResults))
        
        // Recommendations
        report.append(generateRecommendations(benchmarkResults))
        
        return report.toString()
    }
    
    private fun generateExecutiveSummary(results: List<BenchmarkResult>): String {
        val bestPerforming = results.minByOrNull { it.performanceSummary.avgLatencyMs }
        val mostEfficient = results.minByOrNull { it.performanceSummary.energyConsumption }
        val highestAccuracy = results.maxByOrNull { it.performanceSummary.accuracyMap }
        
        return """
## Executive Summary

### Key Findings
- **Best Overall Performance**: ${bestPerforming?.model?.name} on ${bestPerforming?.device?.name}
  - Average Latency: ${bestPerforming?.performanceSummary?.avgLatencyMs}ms
  - FPS: ${bestPerforming?.performanceSummary?.fps}
  
- **Most Energy Efficient**: ${mostEfficient?.model?.name} on ${mostEfficient?.device?.name}
  - Power Consumption: ${mostEfficient?.performanceSummary?.energyConsumption}W
  - Battery Impact: ${mostEfficient?.performanceSummary?.batteryDrainRate}%/hr

- **Highest Accuracy**: ${highestAccuracy?.model?.name} on ${highestAccuracy?.device?.name}
  - mAP: ${highestAccuracy?.performanceSummary?.accuracyMap}

### Device Performance Ranking
${generateDeviceRanking(results)}

### Model Optimization Impact
${generateOptimizationImpact(results)}

"""
    }
    
    private fun generateDeviceComparison(results: List<BenchmarkResult>): String {
        val deviceGroups = results.groupBy { it.device.name }
        
        var report = "\n## Device Performance Comparison\n\n"
        report += "| Device | Model | Latency (ms) | FPS | Memory (MB) | Accuracy (mAP) |\n"
        report += "|--------|-------|--------------|-----|-------------|----------------|\n"
        
        for ((deviceName, deviceResults) in deviceGroups) {
            for (result in deviceResults) {
                report += "| ${deviceName} | ${result.model.name} | "
                report += "${result.performanceSummary.avgLatencyMs.toFixed(1)} | "
                report += "${result.performanceSummary.fps.toFixed(1)} | "
                report += "${result.performanceSummary.memoryUsageMb.toFixed(1)} | "
                report += "${result.performanceSummary.accuracyMap.toFixed(2)} |\n"
            }
        }
        
        return report
    }
    
    private fun generateOptimizationImpact(results: List<BenchmarkResult>): String {
        val fp32Results = results.filter { it.model.quantization == QuantizationType.FP32 }
        val fp16Results = results.filter { it.model.quantization == QuantizationType.FP16 }
        val int8Results = results.filter { it.model.quantization == QuantizationType.INT8 }
        
        var report = "\n### Quantization Impact Analysis\n\n"
        
        if (fp32Results.isNotEmpty() && fp16Results.isNotEmpty()) {
            val avgFp32Latency = fp32Results.map { it.performanceSummary.avgLatencyMs }.average()
            val avgFp16Latency = fp16Results.map { it.performanceSummary.avgLatencyMs }.average()
            val fp16Speedup = ((avgFp32Latency - avgFp16Latency) / avgFp32Latency * 100)
            
            report += "- **FP16 vs FP32**: ${fp16Speedup.toFixed(1)}% speed improvement\n"
        }
        
        if (fp32Results.isNotEmpty() && int8Results.isNotEmpty()) {
            val avgFp32Latency = fp32Results.map { it.performanceSummary.avgLatencyMs }.average()
            val avgInt8Latency = int8Results.map { it.performanceSummary.avgLatencyMs }.average()
            val int8Speedup = ((avgFp32Latency - avgInt8Latency) / avgFp32Latency * 100)
            
            report += "- **INT8 vs FP32**: ${int8Speedup.toFixed(1)}% speed improvement\n"
        }
        
        return report
    }
    
    private fun generateRecommendations(results: List<BenchmarkResult>): String {
        val bestConfigs = results.sortedWith(
            compareBy<BenchmarkResult> { it.performanceSummary.avgLatencyMs }
                .thenBy { it.performanceSummary.memoryUsageMb }
                .thenByDescending { it.performanceSummary.accuracyMap }
        ).take(5)
        
        var report = "\n## Recommendations\n\n"
        report += "### Top 5 Configurations\n\n"
        
        bestConfigs.forEachIndexed { index, config ->
            report += "${index + 1}. **${config.model.name}** on **${config.device.name}**\n"
            report += "   - Latency: ${config.performanceSummary.avgLatencyMs}ms\n"
            report += "   - FPS: ${config.performanceSummary.fps}\n"
            report += "   - Memory: ${config.performanceSummary.memoryUsageMb}MB\n"
            report += "   - Accuracy: ${config.performanceSummary.accuracyMap}\n\n"
        }
        
        report += "### Deployment Guidelines\n\n"
        report += "Based on the benchmark results:\n\n"
        
        // Add specific recommendations based on results
        val hasGpuResults = results.any { 
            it.device.supportedDelegates.contains(DelegateType.GPU) 
        }
        
        if (hasGpuResults) {
            report += "- **GPU acceleration provides significant performance gains** on devices that support it\n"
        }
        
        val hasInt8Results = results.any { it.model.quantization == QuantizationType.INT8 }
        if (hasInt8Results) {
            val int8Latency = results.filter { it.model.quantization == QuantizationType.INT8 }
                .map { it.performanceSummary.avgLatencyMs }.average()
            val fp16Latency = results.filter { it.model.quantization == QuantizationType.FP16 }
                .map { it.performanceSummary.avgLatencyMs }.average()
            
            if (int8Latency < fp16Latency) {
                report += "- **INT8 quantization is recommended** for devices with strong NPU support\n"
            }
        }
        
        return report
    }
}
```

## Continuous Performance Monitoring

### Production Performance Monitoring

```kotlin
class ProductionPerformanceMonitor {
    
    private val performanceMetrics = mutableListOf<PerformanceMetric>()
    private val anomalyDetector = AnomalyDetector()
    
    data class PerformanceMetric(
        val timestamp: Long,
        val modelVersion: String,
        val deviceId: String,
        val latency: Double,
        val fps: Double,
        val memoryUsage: Long,
        val accuracy: Double?
    )
    
    fun recordInference(
        modelVersion: String,
        deviceId: String,
        latency: Double,
        fps: Double,
        memoryUsage: Long,
        accuracy: Double? = null
    ) {
        val metric = PerformanceMetric(
            timestamp = System.currentTimeMillis(),
            modelVersion = modelVersion,
            deviceId = deviceId,
            latency = latency,
            fps = fps,
            memoryUsage = memoryUsage,
            accuracy = accuracy
        )
        
        performanceMetrics.add(metric)
        
        // Check for anomalies
        val anomalies = anomalyDetector.detectAnomalies(metric, getHistoricalMetrics(modelVersion, deviceId))
        if (anomalies.isNotEmpty()) {
            handleAnomalies(anomalies)
        }
        
        // Keep only last 10000 metrics to prevent memory issues
        if (performanceMetrics.size > 10000) {
            performanceMetrics.removeFirst()
        }
    }
    
    private fun getHistoricalMetrics(
        modelVersion: String,
        deviceId: String,
        timeWindow: Duration = Duration.ofHours(1)
    ): List<PerformanceMetric> {
        val cutoffTime = System.currentTimeMillis() - timeWindow.toMillis()
        return performanceMetrics.filter {
            it.modelVersion == modelVersion && 
            it.deviceId == deviceId && 
            it.timestamp > cutoffTime
        }
    }
    
    private fun handleAnomalies(anomalies: List<Anomaly>) {
        for (anomaly in anomalies) {
            when (anomaly.type) {
                AnomalyType.LATENCY_SPIKE -> handleLatencySpike(anomaly)
                AnomalyType.MEMORY_LEAK -> handleMemoryLeak(anomaly)
                AnomalyType.ACCURACY_DROP -> handleAccuracyDrop(anomaly)
                AnomalyType.FPS_DEGRADATION -> handleFpsDegradation(anomaly)
            }
        }
    }
}

class AnomalyDetector {
    
    data class Anomaly(
        val type: AnomalyType,
        val severity: Severity,
        val details: String,
        val value: Double,
        val threshold: Double
    )
    
    enum class AnomalyType { LATENCY_SPIKE, MEMORY_LEAK, ACCURACY_DROP, FPS_DEGRADATION }
    enum class Severity { LOW, MEDIUM, HIGH, CRITICAL }
    
    fun detectAnomalies(
        currentMetric: ProductionPerformanceMonitor.PerformanceMetric,
        historicalMetrics: List<ProductionPerformanceMonitor.PerformanceMetric>
    ): List<Anomaly> {
        val anomalies = mutableListOf<Anomaly>()
        
        if (historicalMetrics.isEmpty()) return anomalies
        
        // Detect latency spikes
        val avgLatency = historicalMetrics.map { it.latency }.average()
        val latencyStdDev = calculateStandardDeviation(historicalMetrics.map { it.latency })
        val latencyThreshold = avgLatency + (2 * latencyStdDev)
        
        if (currentMetric.latency > latencyThreshold) {
            anomalies.add(Anomaly(
                type = AnomalyType.LATENCY_SPIKE,
                severity = if (currentMetric.latency > latencyThreshold * 1.5) Severity.HIGH else Severity.MEDIUM,
                details = "Latency spike detected: ${currentMetric.latency}ms (threshold: ${latencyThreshold}ms)",
                value = currentMetric.latency,
                threshold = latencyThreshold
            ))
        }
        
        // Detect memory leaks
        val recentMemory = historicalMetrics.takeLast(100).map { it.memoryUsage }.average()
        val memoryGrowth = currentMetric.memoryUsage - recentMemory
        if (memoryGrowth > 10 * 1024 * 1024) { // 10MB growth
            anomalies.add(Anomaly(
                type = AnomalyType.MEMORY_LEAK,
                severity = if (memoryGrowth > 50 * 1024 * 1024) Severity.HIGH else Severity.MEDIUM,
                details = "Potential memory leak: ${memoryGrowth / 1024 / 1024}MB growth",
                value = memoryGrowth.toDouble(),
                threshold = 10 * 1024 * 1024.0
            ))
        }
        
        return anomalies
    }
    
    private fun calculateStandardDeviation(values: List<Double>): Double {
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return sqrt(variance)
    }
}
```

### Performance Monitoring Dashboard

```kotlin
class PerformanceDashboard {
    
    fun getPerformanceSummary(
        timeRange: Duration = Duration.ofHours(24)
    ): DashboardSummary {
        
        val cutoffTime = System.currentTimeMillis() - timeRange.toMillis()
        val recentMetrics = performanceMetrics.filter { it.timestamp > cutoffTime }
        
        return DashboardSummary(
            totalInferences = recentMetrics.size,
            averageLatency = recentMetrics.map { it.latency }.average(),
            averageFps = recentMetrics.map { it.fps }.average(),
            memoryEfficiency = calculateMemoryEfficiency(recentMetrics),
            deviceBreakdown = generateDeviceBreakdown(recentMetrics),
            modelBreakdown = generateModelBreakdown(recentMetrics),
            anomalySummary = generateAnomalySummary(recentMetrics)
        )
    }
    
    private fun generateDeviceBreakdown(metrics: List<ProductionPerformanceMonitor.PerformanceMetric>): Map<String, DeviceStats> {
        return metrics.groupBy { it.deviceId }.mapValues { (deviceId, deviceMetrics) ->
            DeviceStats(
                deviceId = deviceId,
                totalInferences = deviceMetrics.size,
                averageLatency = deviceMetrics.map { it.latency }.average(),
                averageFps = deviceMetrics.map { it.fps }.average(),
                errorRate = calculateErrorRate(deviceMetrics)
            )
        }
    }
}
```

## Best Practices for Performance Benchmarking

### Test Consistency
1. **Use the same test data**: Ensure reproducible results with consistent test images
2. **Control environment variables**: Temperature, battery level, background apps
3. **Multiple test runs**: Average results over multiple runs to account for variability
4. **Warm-up runs**: Include warm-up inference to account for initialization overhead

### Measurement Accuracy
1. **High-resolution timing**: Use nanosecond precision for latency measurements
2. **Resource isolation**: Minimize background activity during testing
3. **Statistical significance**: Use sufficient sample sizes for meaningful conclusions
4. **Outlier handling**: Filter or analyze outliers appropriately

### Real-world Simulation
1. **Typical usage patterns**: Simulate realistic user behavior and workload patterns
2. **Memory constraints**: Test with varying available memory conditions
3. **Network conditions**: Consider any cloud-dependent features
4. **Device state variations**: Test with different battery levels, thermal states

### Documentation and Reporting
1. **Detailed logs**: Record all configuration details and environment information
2. **Version control**: Track model versions and configuration changes
3. **Reproducible reports**: Ensure benchmarks can be reproduced by others
4. **Historical tracking**: Maintain performance baselines for regression detection

---

## Next Steps

- See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues and solutions
- Review the [Model Optimization Guide](model_optimization_guide.md) for optimization techniques
- See [Model Integration Guide](MODEL_INTEGRATION.md) for implementation details