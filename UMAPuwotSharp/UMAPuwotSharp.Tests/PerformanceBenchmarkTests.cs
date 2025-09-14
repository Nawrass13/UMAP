using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Diagnostics;
using UMAPuwotSharp;

namespace UMAPuwotSharp.Tests
{
    [TestClass]
    public class PerformanceBenchmarkTests
    {
        private const int LargeSampleCount = 5000;   // Large enough to show HNSW benefit
        private const int LargeFeatureCount = 50;    // High-dimensional
        private const int BenchmarkEpochs = 50;      // Reasonable for timing

        /// <summary>
        /// Benchmark HNSW vs Exact performance comparison
        /// </summary>
        [TestMethod]
        public void Benchmark_HNSW_vs_Exact_Performance()
        {
            Console.WriteLine("=== PERFORMANCE BENCHMARK: HNSW vs Exact ===");
            Console.WriteLine($"Dataset: {LargeSampleCount} samples × {LargeFeatureCount} features");
            Console.WriteLine($"Epochs: {BenchmarkEpochs}");
            Console.WriteLine();

            var testData = GenerateLargeTestData(LargeSampleCount, LargeFeatureCount, seed: 42);

            // Benchmark HNSW Approximate Mode
            Console.WriteLine("🚀 HNSW Approximate Mode Benchmark");
            var (hnswTime, hnswMemory) = BenchmarkFitOperation(() =>
            {
                using var model = new UMapModel();
                return model.Fit(testData,
                    embeddingDimension: 2,
                    nEpochs: BenchmarkEpochs,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false); // HNSW mode
            }, "HNSW");

            // Benchmark Exact Mode
            Console.WriteLine("⚡ Exact Brute-Force Mode Benchmark");
            var (exactTime, exactMemory) = BenchmarkFitOperation(() =>
            {
                using var model = new UMapModel();
                return model.Fit(testData,
                    embeddingDimension: 2,
                    nEpochs: BenchmarkEpochs,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: true); // Exact mode
            }, "Exact");

            // Performance Analysis
            double speedup = exactTime / hnswTime;
            double memoryReduction = ((exactMemory - hnswMemory) / (double)exactMemory) * 100.0;

            Console.WriteLine();
            Console.WriteLine("📊 PERFORMANCE ANALYSIS");
            Console.WriteLine("========================");
            Console.WriteLine($"HNSW Time:     {hnswTime:F2}s");
            Console.WriteLine($"Exact Time:    {exactTime:F2}s");
            Console.WriteLine($"Speedup:       {speedup:F2}x");
            Console.WriteLine();
            Console.WriteLine($"HNSW Memory:   {hnswMemory / 1024 / 1024:F1} MB");
            Console.WriteLine($"Exact Memory:  {exactMemory / 1024 / 1024:F1} MB");
            Console.WriteLine($"Memory Reduction: {memoryReduction:F1}%");
            Console.WriteLine();

            // Validation Assertions
            Assert.IsTrue(speedup >= 1.0, $"HNSW should be at least as fast as exact (speedup: {speedup:F2}x)");

            // For large datasets, expect significant speedup
            if (LargeSampleCount >= 2000)
            {
                Assert.IsTrue(speedup >= 2.0, $"HNSW should be significantly faster for large datasets (speedup: {speedup:F2}x)");
            }

            // Memory usage should be better with HNSW (though measurement may vary)
            Console.WriteLine($"✅ Performance benchmark completed - {speedup:F2}x speedup achieved");
        }

        /// <summary>
        /// Benchmark different metrics performance with HNSW
        /// </summary>
        [TestMethod]
        public void Benchmark_Multi_Metric_Performance()
        {
            Console.WriteLine("=== MULTI-METRIC PERFORMANCE BENCHMARK ===");

            var testData = GenerateLargeTestData(2000, 30, seed: 123);

            var metrics = new[]
            {
                DistanceMetric.Euclidean,
                DistanceMetric.Cosine,
                DistanceMetric.Manhattan
            };

            Console.WriteLine($"Dataset: {testData.GetLength(0)} samples × {testData.GetLength(1)} features");
            Console.WriteLine();

            foreach (var metric in metrics)
            {
                Console.WriteLine($"🎯 Benchmarking {metric} Distance");

                var (time, memory) = BenchmarkFitOperation(() =>
                {
                    using var model = new UMapModel();
                    return model.Fit(testData,
                        embeddingDimension: 2,
                        nEpochs: 30,
                        metric: metric,
                        forceExactKnn: false); // HNSW mode
                }, metric.ToString());

                Console.WriteLine($"   Time: {time:F2}s, Memory Peak: {memory / 1024 / 1024:F1} MB");
                Console.WriteLine();

                // Basic validation
                Assert.IsTrue(time > 0, $"{metric} benchmark should complete");
                Assert.IsTrue(time < 300, $"{metric} should complete in reasonable time"); // 5 minute timeout
            }

            Console.WriteLine("✅ Multi-metric benchmark completed");
        }

        /// <summary>
        /// Benchmark transform operation performance
        /// </summary>
        [TestMethod]
        public void Benchmark_Transform_Performance()
        {
            Console.WriteLine("=== TRANSFORM PERFORMANCE BENCHMARK ===");

            var trainData = GenerateLargeTestData(3000, 40, seed: 456);

            // Train model first
            using var model = new UMapModel();
            var trainingStopwatch = Stopwatch.StartNew();

            var embedding = model.Fit(trainData,
                embeddingDimension: 2,
                nEpochs: 40,
                forceExactKnn: false);

            trainingStopwatch.Stop();
            Console.WriteLine($"Training completed in {trainingStopwatch.Elapsed.TotalSeconds:F2}s");

            // Generate test samples for transform
            const int transformSamples = 1000;
            var transformData = GenerateLargeTestData(transformSamples, 40, seed: 789);

            // Benchmark transform operations
            var transformStopwatch = Stopwatch.StartNew();
            long memoryBefore = GC.GetTotalMemory(true);

            for (int i = 0; i < transformSamples; i++)
            {
                var singleSample = ExtractRow(transformData, i);
                var result = model.Transform(singleSample);

                // Basic validation
                Assert.IsNotNull(result);
                Assert.AreEqual(2, result.GetLength(1));
            }

            transformStopwatch.Stop();
            long memoryAfter = GC.GetTotalMemory(false);
            long memoryUsed = Math.Max(0, memoryAfter - memoryBefore);

            double avgTransformTime = transformStopwatch.Elapsed.TotalMilliseconds / transformSamples;
            double transformsPerSecond = transformSamples / transformStopwatch.Elapsed.TotalSeconds;

            Console.WriteLine($"Transform Performance:");
            Console.WriteLine($"  Total time: {transformStopwatch.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"  Average per transform: {avgTransformTime:F3}ms");
            Console.WriteLine($"  Transforms/second: {transformsPerSecond:F1}");
            Console.WriteLine($"  Memory overhead: {memoryUsed / 1024:F1} KB");

            // Performance assertions
            Assert.IsTrue(avgTransformTime < 50, "Transform should be fast (<50ms average)");
            Assert.IsTrue(transformsPerSecond > 20, "Should handle >20 transforms per second");

            Console.WriteLine("✅ Transform benchmark completed");
        }

        /// <summary>
        /// Memory usage analysis for different dataset sizes
        /// </summary>
        [TestMethod]
        public void Benchmark_Memory_Scaling()
        {
            Console.WriteLine("=== MEMORY SCALING BENCHMARK ===");

            var dataSizes = new[] { 500, 1000, 2000, 3000 };
            const int features = 25;

            Console.WriteLine($"Testing memory scaling with {features} features");
            Console.WriteLine();

            foreach (var size in dataSizes)
            {
                var testData = GenerateLargeTestData(size, features, seed: size);

                long memoryBefore = GC.GetTotalMemory(true);
                var stopwatch = Stopwatch.StartNew();

                using (var model = new UMapModel())
                {
                    var embedding = model.Fit(testData,
                        embeddingDimension: 2,
                        nEpochs: 20,
                        forceExactKnn: false);

                    stopwatch.Stop();
                    long memoryPeak = GC.GetTotalMemory(false);
                    long memoryUsed = memoryPeak - memoryBefore;

                    Console.WriteLine($"{size:D4} samples: {stopwatch.Elapsed.TotalSeconds:F1}s, " +
                                    $"{memoryUsed / 1024 / 1024:F1} MB");

                    // Validation
                    Assert.IsNotNull(embedding);
                    Assert.AreEqual(size, embedding.GetLength(0));
                }

                // Force cleanup
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            Console.WriteLine();
            Console.WriteLine("✅ Memory scaling benchmark completed");
        }

        #region Helper Methods

        private static (double timeSeconds, long memoryBytes) BenchmarkFitOperation(
            Func<float[,]> operation, string operationName)
        {
            // Force GC before benchmark
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            long memoryBefore = GC.GetTotalMemory(false);
            var stopwatch = Stopwatch.StartNew();

            var result = operation();

            stopwatch.Stop();
            long memoryAfter = GC.GetTotalMemory(false);

            // Validate result
            Assert.IsNotNull(result, $"{operationName} should return valid embedding");

            double timeSeconds = stopwatch.Elapsed.TotalSeconds;
            long memoryUsed = Math.Max(memoryAfter - memoryBefore, 0);

            Console.WriteLine($"   {operationName}: {timeSeconds:F2}s, Peak memory: {memoryUsed / 1024 / 1024:F1} MB");

            return (timeSeconds, memoryUsed);
        }

        private static float[,] GenerateLargeTestData(int nSamples, int nFeatures, int seed = 42)
        {
            var data = new float[nSamples, nFeatures];
            var random = new Random(seed);

            // Generate more structured data with multiple clusters for realistic benchmark
            int nClusters = 5;

            for (int i = 0; i < nSamples; i++)
            {
                int cluster = i % nClusters;

                for (int j = 0; j < nFeatures; j++)
                {
                    // Box-Muller transform for normal distribution
                    double u1 = 1.0 - random.NextDouble();
                    double u2 = 1.0 - random.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                         Math.Sin(2.0 * Math.PI * u2);

                    // Add cluster structure
                    data[i, j] = (float)(cluster * 2.0 + randStdNormal);
                }
            }

            return data;
        }

        private static float[,] ExtractRow(float[,] data, int rowIndex)
        {
            int cols = data.GetLength(1);
            var row = new float[1, cols];

            for (int j = 0; j < cols; j++)
            {
                row[0, j] = data[rowIndex, j];
            }

            return row;
        }

        #endregion
    }
}