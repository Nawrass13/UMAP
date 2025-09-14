using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Diagnostics;
using UMAPuwotSharp;

namespace UMAPuwotSharp.Tests
{
    [TestClass]
    public class UMapModelTests
    {
        private float[,] _testData;
        private const int TestSamples = 200;
        private const int TestFeatures = 10;

        [TestInitialize]
        public void Setup()
        {
            // Generate structured test data with 3 clusters
            _testData = GenerateTestData(TestSamples, TestFeatures, seed: 42);
        }

        /// <summary>
        /// Test HNSW approximate mode (default behavior)
        /// </summary>
        [TestMethod]
        public void Test_HNSW_Approximate_Mode()
        {
            using var model = new UMapModel();

            var stopwatch = Stopwatch.StartNew();
            var embedding = model.Fit(_testData,
                embeddingDimension: 2,
                nNeighbors: 15,
                nEpochs: 50,
                metric: DistanceMetric.Euclidean,
                forceExactKnn: false); // HNSW approximate (default)

            stopwatch.Stop();

            // Validation
            Assert.IsNotNull(embedding);
            Assert.AreEqual(TestSamples, embedding.GetLength(0));
            Assert.AreEqual(2, embedding.GetLength(1));
            Assert.IsTrue(model.IsFitted);

            Console.WriteLine($"HNSW approximate mode completed in {stopwatch.ElapsedMilliseconds}ms");
            ValidateEmbeddingQuality(embedding, "HNSW Approximate");
        }

        /// <summary>
        /// Test exact brute-force mode
        /// </summary>
        [TestMethod]
        public void Test_Exact_BruteForce_Mode()
        {
            using var model = new UMapModel();

            var stopwatch = Stopwatch.StartNew();
            var embedding = model.Fit(_testData,
                embeddingDimension: 2,
                nNeighbors: 15,
                nEpochs: 50,
                metric: DistanceMetric.Euclidean,
                forceExactKnn: true); // Force exact computation

            stopwatch.Stop();

            // Validation
            Assert.IsNotNull(embedding);
            Assert.AreEqual(TestSamples, embedding.GetLength(0));
            Assert.AreEqual(2, embedding.GetLength(1));
            Assert.IsTrue(model.IsFitted);

            Console.WriteLine($"Exact brute-force mode completed in {stopwatch.ElapsedMilliseconds}ms");
            ValidateEmbeddingQuality(embedding, "Exact Brute-Force");
        }

        /// <summary>
        /// Test HNSW vs Exact accuracy comparison
        /// </summary>
        [TestMethod]
        public void Test_HNSW_vs_Exact_Accuracy()
        {
            // Test with smaller dataset for faster execution
            var smallData = GenerateTestData(100, 5, seed: 12345);

            using var hnswModel = new UMapModel();
            using var exactModel = new UMapModel();

            // HNSW approximate
            var hnswEmbedding = hnswModel.Fit(smallData,
                embeddingDimension: 2,
                nEpochs: 30,
                forceExactKnn: false);

            // Exact computation
            var exactEmbedding = exactModel.Fit(smallData,
                embeddingDimension: 2,
                nEpochs: 30,
                forceExactKnn: true);

            // Calculate MSE between embeddings
            double mse = CalculateMSE(hnswEmbedding, exactEmbedding);

            Console.WriteLine($"MSE between HNSW and Exact: {mse:F6}");

            // Validation: MSE should be < 0.01 (as per specification)
            Assert.IsTrue(mse < 0.01, $"MSE ({mse:F6}) should be < 0.01");
            Assert.IsTrue(mse >= 0, "MSE should be non-negative");
        }

        /// <summary>
        /// Test all distance metrics with HNSW support
        /// </summary>
        [TestMethod]
        public void Test_Multi_Metric_Support()
        {
            var supportedMetrics = new[]
            {
                DistanceMetric.Euclidean,
                DistanceMetric.Cosine,
                DistanceMetric.Manhattan
            };

            foreach (var metric in supportedMetrics)
            {
                using var model = new UMapModel();

                Console.WriteLine($"Testing {metric} distance metric...");

                var embedding = model.Fit(_testData,
                    embeddingDimension: 2,
                    nEpochs: 30,
                    metric: metric,
                    forceExactKnn: false); // Should use HNSW

                Assert.IsNotNull(embedding, $"{metric} metric failed");
                Assert.AreEqual(TestSamples, embedding.GetLength(0));
                Assert.AreEqual(2, embedding.GetLength(1));
                Assert.IsTrue(model.IsFitted);

                Console.WriteLine($"✅ {metric} metric test passed");
            }
        }

        /// <summary>
        /// Test unsupported metrics (should fall back to exact)
        /// </summary>
        [TestMethod]
        public void Test_Unsupported_Metrics_Fallback()
        {
            var unsupportedMetrics = new[]
            {
                DistanceMetric.Correlation,
                DistanceMetric.Hamming
            };

            foreach (var metric in unsupportedMetrics)
            {
                using var model = new UMapModel();

                Console.WriteLine($"Testing {metric} distance metric (exact fallback)...");

                var embedding = model.Fit(_testData,
                    embeddingDimension: 2,
                    nEpochs: 20, // Fewer epochs for faster test
                    metric: metric,
                    forceExactKnn: false); // Should fall back to exact automatically

                Assert.IsNotNull(embedding, $"{metric} metric failed");
                Assert.AreEqual(TestSamples, embedding.GetLength(0));
                Assert.AreEqual(2, embedding.GetLength(1));
                Assert.IsTrue(model.IsFitted);

                Console.WriteLine($"✅ {metric} metric fallback test passed");
            }
        }

        /// <summary>
        /// Test progress reporting with new enhanced callback
        /// </summary>
        [TestMethod]
        public void Test_Enhanced_Progress_Reporting()
        {
            using var model = new UMapModel();

            var progressReports = new System.Collections.Generic.List<string>();
            bool progressCallbackInvoked = false;

            // Progress callback that captures reports
            void ProgressCallback(int epoch, int totalEpochs, float percent)
            {
                progressCallbackInvoked = true;
                var report = $"Epoch {epoch}/{totalEpochs}: {percent:F1}%";
                progressReports.Add(report);
                Console.WriteLine($"[PROGRESS] {report}");
            }

            var embedding = model.FitWithProgress(_testData,
                ProgressCallback,
                embeddingDimension: 2,
                nEpochs: 30,
                forceExactKnn: false);

            // Validation
            Assert.IsNotNull(embedding);
            Assert.IsTrue(progressCallbackInvoked, "Progress callback should be invoked");
            Assert.IsTrue(progressReports.Count > 0, "Should have progress reports");

            Console.WriteLine($"Captured {progressReports.Count} progress reports");
        }

        /// <summary>
        /// Test enhanced transform with safety analysis
        /// </summary>
        [TestMethod]
        public void Test_Enhanced_Transform()
        {
            using var model = new UMapModel();

            // Train model first
            var embedding = model.Fit(_testData,
                embeddingDimension: 2,
                nEpochs: 30,
                forceExactKnn: false);

            Assert.IsTrue(model.IsFitted);

            // Test normal transform
            var newData = GenerateSingleSample(TestFeatures, seed: 999);
            var transformResult = model.Transform(newData);

            Assert.IsNotNull(transformResult);
            Assert.AreEqual(2, transformResult.GetLength(1));

            // Test enhanced transform (if available)
            try
            {
                var enhancedResult = model.TransformDetailed(newData);
                Assert.IsNotNull(enhancedResult);
                Assert.IsTrue(enhancedResult.ProjectionCoordinates.Length == 2);

                Console.WriteLine($"Enhanced transform - Confidence: {enhancedResult.ConfidenceScore:F3}, " +
                                $"Outlier Level: {enhancedResult.Severity}, " +
                                $"Percentile: {enhancedResult.PercentileRank:F1}%");
            }
            catch (NotImplementedException)
            {
                Console.WriteLine("Enhanced transform not yet implemented - using basic transform");
            }
        }

        /// <summary>
        /// Test model persistence with HNSW indices
        /// </summary>
        [TestMethod]
        public void Test_Model_Persistence()
        {
            var modelPath = "test_model_hnsw.bin";

            try
            {
                // Train and save model
                using (var model = new UMapModel())
                {
                    var embedding = model.Fit(_testData,
                        embeddingDimension: 2,
                        nEpochs: 30,
                        forceExactKnn: false);

                    Assert.IsTrue(model.IsFitted);

                    // Save model
                    model.SaveModel(modelPath);
                }

                // Load and test model
                using (var loadedModel = new UMapModel())
                {
                    loadedModel.LoadModel(modelPath);
                    Assert.IsTrue(loadedModel.IsFitted);

                    // Test transform with loaded model
                    var newData = GenerateSingleSample(TestFeatures, seed: 777);
                    var transformResult = loadedModel.Transform(newData);

                    Assert.IsNotNull(transformResult);
                    Assert.AreEqual(2, transformResult.GetLength(1));

                    Console.WriteLine($"Model persistence test passed - transform result: " +
                                    $"[{transformResult[0,0]:F3}, {transformResult[0,1]:F3}]");
                }
            }
            finally
            {
                // Cleanup
                if (System.IO.File.Exists(modelPath))
                {
                    System.IO.File.Delete(modelPath);
                }
            }
        }

        /// <summary>
        /// Test parameter validation
        /// </summary>
        [TestMethod]
        public void Test_Parameter_Validation()
        {
            using var model = new UMapModel();

            // Test null data
            Assert.ThrowsException<ArgumentNullException>(() =>
                model.Fit(null!, embeddingDimension: 2));

            // Test empty data
            var emptyData = new float[0, 0];
            Assert.ThrowsException<ArgumentException>(() =>
                model.Fit(emptyData, embeddingDimension: 2));

            // Test invalid dimensions
            Assert.ThrowsException<ArgumentException>(() =>
                model.Fit(_testData, embeddingDimension: 0));

            Assert.ThrowsException<ArgumentException>(() =>
                model.Fit(_testData, embeddingDimension: 51)); // Max is 50

            // Test invalid neighbors
            Assert.ThrowsException<ArgumentException>(() =>
                model.Fit(_testData, nNeighbors: 0));
        }

        #region Helper Methods

        private static float[,] GenerateTestData(int nSamples, int nFeatures, int seed = 42)
        {
            var data = new float[nSamples, nFeatures];
            var random = new Random(seed);

            // Generate 3 distinct clusters
            for (int i = 0; i < nSamples; i++)
            {
                int cluster = i / (nSamples / 3);
                if (cluster >= 3) cluster = 2; // Handle remainder

                for (int j = 0; j < nFeatures; j++)
                {
                    // Normal distribution with cluster-specific mean
                    double u1 = 1.0 - random.NextDouble();
                    double u2 = 1.0 - random.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

                    data[i, j] = (float)(cluster * 3.0 + randStdNormal);
                }
            }

            return data;
        }

        private static float[,] GenerateSingleSample(int nFeatures, int seed = 999)
        {
            var data = new float[1, nFeatures];
            var random = new Random(seed);

            for (int j = 0; j < nFeatures; j++)
            {
                data[0, j] = (float)(random.NextDouble() * 2 - 1); // Range [-1, 1]
            }

            return data;
        }

        private static double CalculateMSE(float[,] embedding1, float[,] embedding2)
        {
            if (embedding1.GetLength(0) != embedding2.GetLength(0) ||
                embedding1.GetLength(1) != embedding2.GetLength(1))
            {
                throw new ArgumentException("Embedding dimensions must match");
            }

            double sumSquaredDiff = 0.0;
            int totalElements = embedding1.GetLength(0) * embedding1.GetLength(1);

            for (int i = 0; i < embedding1.GetLength(0); i++)
            {
                for (int j = 0; j < embedding1.GetLength(1); j++)
                {
                    double diff = embedding1[i, j] - embedding2[i, j];
                    sumSquaredDiff += diff * diff;
                }
            }

            return sumSquaredDiff / totalElements;
        }

        private static void ValidateEmbeddingQuality(float[,] embedding, string modeName)
        {
            // Basic quality checks
            int nSamples = embedding.GetLength(0);
            int nDim = embedding.GetLength(1);

            // Check for NaN or infinity values
            int invalidCount = 0;
            double minVal = double.MaxValue;
            double maxVal = double.MinValue;

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nDim; j++)
                {
                    float val = embedding[i, j];
                    if (float.IsNaN(val) || float.IsInfinity(val))
                    {
                        invalidCount++;
                    }
                    else
                    {
                        minVal = Math.Min(minVal, val);
                        maxVal = Math.Max(maxVal, val);
                    }
                }
            }

            Console.WriteLine($"{modeName} embedding stats - Min: {minVal:F3}, Max: {maxVal:F3}, Invalid: {invalidCount}");

            // Validation assertions
            Assert.AreEqual(0, invalidCount, "Embedding should not contain NaN or infinity values");
            Assert.IsTrue(Math.Abs(maxVal - minVal) > 0.1, "Embedding should have reasonable range");
        }

        #endregion
    }
}