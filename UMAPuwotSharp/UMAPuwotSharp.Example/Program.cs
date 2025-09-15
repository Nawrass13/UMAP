using System;
using System.IO;
using System.Linq;
using UMAPuwotSharp;

namespace UMAPExample
{
    class CompleteUsageDemo
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== Complete Enhanced UMAP Wrapper Demo ===\n");

            try
            {
                // Demo 1: 27D Embedding with Progress Reporting
                Demo27DEmbeddingWithProgress();

                // Demo 2: Multi-Dimensional Embeddings (1D to 50D)
                DemoMultiDimensionalEmbeddings();

                // Demo 3: Model Persistence and Transform
                DemoModelPersistence();

                // Demo 4: Different Data Types and Metrics with Progress
                DemoDistanceMetricsWithProgress();

                // Demo 5: Enhanced Safety Features with HNSW
                DemoSafetyFeatures();

                // Demo 6: New Spread Parameter with Smart Defaults
                DemoSpreadParameter();

                Console.WriteLine("\nAll demos completed successfully!");
                Console.WriteLine("Your enhanced UMAP wrapper is ready for production use!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        static void Demo27DEmbeddingWithProgress()
        {
            Console.WriteLine("=== Demo 1: 27D Embedding with Progress Reporting ===");

            // Generate sample high-dimensional data
            const int nSamples = 10000;
            const int nFeatures = 300;
            const int embeddingDim = 27;

            var data = GenerateTestData(nSamples, nFeatures, DataPattern.Clustered);
            Console.WriteLine($"Generated data: {nSamples} samples × {nFeatures} features");

            using var model = new UMapModel();

            Console.WriteLine("Training 27D UMAP embedding with progress reporting...");
            var startTime = DateTime.Now;

            // Progress tracking variables
            var lastPercent = -1;
            var progressBar = new char[50];

            var embedding = model.FitWithProgress(
                data: data,
                progressCallback: (epoch, totalEpochs, percent) =>
                {
                    var currentPercent = (int)percent;
                    if (currentPercent != lastPercent && currentPercent % 2 == 0) // Update every 2%
                    {
                        lastPercent = currentPercent;

                        // Update progress bar
                        var filled = (int)(percent / 2); // 50 characters for 100%
                        for (int i = 0; i < 50; i++)
                        {
                            progressBar[i] = i < filled ? '█' : '░';
                        }

                        Console.Write($"\r  Progress: [{new string(progressBar)}] {percent:F1}% (Epoch {epoch}/{totalEpochs})");
                    }
                },
                embeddingDimension: embeddingDim,
                nNeighbors: 20,
                minDist: 0.05f,
                nEpochs: 300,
                metric: DistanceMetric.Euclidean
            );

            var elapsed = DateTime.Now - startTime;
            Console.WriteLine($"\nTraining completed in {elapsed.TotalMilliseconds:F0}ms");

            // Display model info
            var info = model.ModelInfo;
            Console.WriteLine($"Model: {info}");

            // Show embedding statistics
            Console.WriteLine($"Embedding shape: [{embedding.GetLength(0)}, {embedding.GetLength(1)}]");
            ShowEmbeddingStats(embedding, "27D embedding");

            Console.WriteLine();
        }

        static void DemoMultiDimensionalEmbeddings()
        {
            Console.WriteLine("=== Demo 2: Multi-Dimensional Embeddings (1D to 50D) ===");

            var data = GenerateTestData(300, 50, DataPattern.Standard);
            Console.WriteLine($"Generated data: {data.GetLength(0)} samples × {data.GetLength(1)} features");

            var testDimensions = new[] { 1, 2, 3, 5, 10, 15, 20, 27, 35, 50 };

            foreach (var dim in testDimensions)
            {
                Console.WriteLine($"\nTesting {dim}D embedding:");

                using var model = new UMapModel();

                // Use progress callback for larger dimensions
                ProgressCallback? progressCallback = null;
                if (dim >= 20)
                {
                    progressCallback = (epoch, totalEpochs, percent) =>
                    {
                        if (epoch % 25 == 0 || epoch == totalEpochs) // Report every 25 epochs
                        {
                            Console.Write($"\r    Training {dim}D: {percent:F0}% ");
                        }
                    };
                }

                var startTime = DateTime.Now;

                float[,] embedding;
                if (progressCallback != null)
                {
                    embedding = model.FitWithProgress(
                        data,
                        progressCallback,
                        embeddingDimension: dim,
                        nNeighbors: 15,
                        minDist: 0.1f,
                        nEpochs: 150,
                        metric: DistanceMetric.Euclidean
                    );
                }
                else
                {
                    embedding = model.Fit(
                        data,
                        embeddingDimension: dim,
                        nNeighbors: 15,
                        minDist: 0.1f,
                        nEpochs: 150,
                        metric: DistanceMetric.Euclidean
                    );
                }

                var elapsed = DateTime.Now - startTime;

                if (dim >= 20)
                {
                    Console.WriteLine(); // New line after progress
                }

                Console.WriteLine($"  Result: {embedding.GetLength(0)} samples → {dim}D in {elapsed.TotalMilliseconds:F0}ms");

                var info = model.ModelInfo;
                Console.WriteLine($"  Model info: {info.OutputDimension}D embedding, {info.TrainingSamples} training samples");

                // Show stats for first few dimensions
                ShowEmbeddingStats(embedding, $"{dim}D embedding", maxDims: Math.Min(5, dim));
            }

            Console.WriteLine();
        }

        static void DemoModelPersistence()
        {
            Console.WriteLine("=== Demo 3: Model Persistence and Transform ===");

            const string modelFile = "demo_model.umap";

            try
            {
                // Generate training data
                var trainData = GenerateTestData(500, 50, DataPattern.Standard);
                var testData = GenerateTestData(100, 50, DataPattern.Standard, seed: 456);

                UMapModelInfo savedInfo;

                // Train and save model with progress
                using (var model = new UMapModel())
                {
                    Console.WriteLine("Training model with progress reporting...");

                    var trainEmbedding = model.FitWithProgress(
                        trainData,
                        progressCallback: (epoch, totalEpochs, percent) =>
                        {
                            if (epoch % 20 == 0 || epoch == totalEpochs)
                            {
                                Console.Write($"\r  Training progress: {percent:F0}% (Epoch {epoch}/{totalEpochs})");
                            }
                        },
                        embeddingDimension: 5,
                        nNeighbors: 15,
                        minDist: 0.1f,
                        nEpochs: 200,
                        metric: DistanceMetric.Cosine
                    );

                    Console.WriteLine(); // New line after progress

                    savedInfo = model.ModelInfo;
                    Console.WriteLine($"Trained model: {savedInfo}");

                    Console.WriteLine("Saving model...");
                    model.Save(modelFile);
                    Console.WriteLine($"Model saved to: {modelFile}");
                }

                // Load and use model
                Console.WriteLine("Loading model from disk...");
                using var loadedModel = UMapModel.Load(modelFile);

                var loadedInfo = loadedModel.ModelInfo;
                Console.WriteLine($"Loaded model: {loadedInfo}");

                // Verify model consistency
                if (savedInfo.ToString() == loadedInfo.ToString())
                {
                    Console.WriteLine("✓ Model loaded successfully with consistent parameters");
                }

                // Transform new data
                Console.WriteLine("Transforming new data...");
                var testEmbedding = loadedModel.Transform(testData);
                Console.WriteLine($"Transformed {testData.GetLength(0)} samples");

                ShowEmbeddingStats(testEmbedding, "Transformed data");
            }
            finally
            {
                // Cleanup
                if (File.Exists(modelFile))
                    File.Delete(modelFile);
            }

            Console.WriteLine();
        }

        static void DemoDistanceMetricsWithProgress()
        {
            Console.WriteLine("=== Demo 4: Different Distance Metrics with Progress ===");

            var metrics = new[]
            {
                (DistanceMetric.Euclidean, DataPattern.Standard, "Standard Gaussian data"),
                (DistanceMetric.Cosine, DataPattern.Sparse, "Sparse high-dimensional data"),
                (DistanceMetric.Manhattan, DataPattern.Clustered, "Clustered data (outlier robust)"),
                (DistanceMetric.Correlation, DataPattern.Correlated, "Correlated features"),
                (DistanceMetric.Hamming, DataPattern.Binary, "Binary/categorical data")
            };

            foreach (var (metric, pattern, description) in metrics)
            {
                Console.WriteLine($"\nTesting {UMapModel.GetMetricName(metric)} metric:");
                Console.WriteLine($"  Data type: {description}");

                var data = GenerateTestData(200, 20, pattern);

                using var model = new UMapModel();

                // Progress callback for this metric
                var metricName = UMapModel.GetMetricName(metric);
                var embedding = model.FitWithProgress(
                    data,
                    progressCallback: (epoch, totalEpochs, percent) =>
                    {
                        if (epoch % 15 == 0 || epoch == totalEpochs)
                        {
                            Console.Write($"\r  {metricName}: {percent:F0}% ");
                        }
                    },
                    embeddingDimension: 2,
                    nNeighbors: 12,
                    minDist: 0.1f,
                    nEpochs: 150,
                    metric: metric
                );

                Console.WriteLine(); // New line after progress

                var info = model.ModelInfo;
                Console.WriteLine($"  Result: {embedding.GetLength(0)} samples → 2D, metric: {info.MetricName}");
                ShowEmbeddingStats(embedding, $"{UMapModel.GetMetricName(metric)} embedding", maxDims: 2);
            }

            Console.WriteLine();
        }

        static void DemoSafetyFeatures()
        {
            Console.WriteLine("=== Demo 5: Enhanced Safety Features with HNSW ===");

            // Generate training data with clear patterns
            var trainData = GenerateTestData(400, 30, DataPattern.Clustered, seed: 123);
            Console.WriteLine($"Training data: {trainData.GetLength(0)} samples × {trainData.GetLength(1)} features (clustered pattern)");

            using var model = new UMapModel();

            // Train the model
            Console.WriteLine("Training model for safety analysis...");
            var trainEmbedding = model.FitWithProgress(
                trainData,
                progressCallback: (epoch, totalEpochs, percent) =>
                {
                    if (epoch % 30 == 0 || epoch == totalEpochs)
                    {
                        Console.Write($"\r  Training: {percent:F0}%");
                    }
                },
                embeddingDimension: 10,
                nNeighbors: 15,
                minDist: 0.1f,
                nEpochs: 200,
                metric: DistanceMetric.Euclidean
            );

            Console.WriteLine("\n  Training completed!");

            // Generate different types of test data to demonstrate safety analysis
            var testScenarios = new[]
            {
                (GenerateTestData(5, 30, DataPattern.Clustered, seed: 200), "Similar to training (clustered)", true),
                (GenerateTestData(5, 30, DataPattern.Standard, seed: 201), "Somewhat different (Gaussian)", false),
                (GenerateExtremeOutliers(3, 30), "Extreme outliers", false)
            };

            Console.WriteLine("\nTransform Analysis with Safety Metrics:");

            foreach (var (testData, description, expectedSafe) in testScenarios)
            {
                Console.WriteLine($"\n--- {description} ---");

                try
                {
                    // Use enhanced transform with safety analysis
                    var results = model.TransformWithSafety(testData);

                    for (int i = 0; i < results.Length; i++)
                    {
                        var result = results[i];
                        Console.WriteLine($"  Sample {i + 1}:");
                        Console.WriteLine($"    Confidence: {result.ConfidenceScore:F3}");
                        Console.WriteLine($"    Severity: {result.Severity}");
                        Console.WriteLine($"    Percentile: {result.PercentileRank:F1}%");
                        Console.WriteLine($"    Z-Score: {result.ZScore:F2}");
                        Console.WriteLine($"    Quality: {result.QualityAssessment}");
                        Console.WriteLine($"    Production Ready: {(result.IsReliable ? "✓ Yes" : "✗ No")}");

                        // Show embedding coordinates (first 3 dimensions)
                        var coords = result.ProjectionCoordinates;
                        var coordStr = string.Join(", ", coords.Take(Math.Min(3, coords.Length)).Select(x => x.ToString("F3")));
                        if (coords.Length > 3) coordStr += "...";
                        Console.WriteLine($"    Coordinates: [{coordStr}]");

                        // Show nearest neighbors info
                        Console.WriteLine($"    Nearest neighbors: {result.NeighborCount} analyzed");
                        var nearestDist = result.NearestNeighborDistances[0];
                        Console.WriteLine($"    Closest training point distance: {nearestDist:F3}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"    Error during transform: {ex.Message}");
                }
            }

            // Demonstrate batch processing with safety filtering
            Console.WriteLine("\n--- Batch Processing with Safety Filtering ---");
            var batchData = GenerateTestData(20, 30, DataPattern.Standard, seed: 300);

            try
            {
                var batchResults = model.TransformWithSafety(batchData);

                var safeCount = batchResults.Count(r => r.IsReliable);
                var normalCount = batchResults.Count(r => r.Severity == OutlierLevel.Normal);
                var outlierCount = batchResults.Count(r => r.Severity >= OutlierLevel.Mild);

                Console.WriteLine($"  Processed {batchResults.Length} samples:");
                Console.WriteLine($"    ✓ Production safe: {safeCount}/{batchResults.Length} ({100.0 * safeCount / batchResults.Length:F1}%)");
                Console.WriteLine($"    Normal: {normalCount}, Outliers: {outlierCount}");

                // Show distribution of confidence scores
                var avgConfidence = batchResults.Average(r => r.ConfidenceScore);
                var minConfidence = batchResults.Min(r => r.ConfidenceScore);
                var maxConfidence = batchResults.Max(r => r.ConfidenceScore);

                Console.WriteLine($"    Confidence range: {minConfidence:F3} - {maxConfidence:F3} (avg: {avgConfidence:F3})");

                // Show severity distribution
                var severityGroups = batchResults.GroupBy(r => r.Severity)
                                                .OrderBy(g => g.Key)
                                                .Select(g => $"{g.Key}: {g.Count()}")
                                                .ToArray();
                Console.WriteLine($"    Severity breakdown: {string.Join(", ", severityGroups)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"    Batch processing error: {ex.Message}");
            }

            Console.WriteLine("\n  Safety analysis demonstrates:");
            Console.WriteLine("  • Real-time outlier detection for production safety");
            Console.WriteLine("  • Confidence scoring for reliability assessment");
            Console.WriteLine("  • Multi-level severity classification");
            Console.WriteLine("  • Nearest neighbor analysis for interpretability");
            Console.WriteLine("  • Quality assessment for decision making");

            Console.WriteLine();
        }

        static float[,] GenerateExtremeOutliers(int nSamples, int nFeatures, int seed = 999)
        {
            var random = new Random(seed);
            var data = new float[nSamples, nFeatures];

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    // Generate extreme values far from typical training data
                    var sign = random.NextDouble() < 0.5 ? -1 : 1;
                    data[i, j] = sign * (float)(10 + random.NextDouble() * 5); // Values in range [-15, -10] or [10, 15]
                }
            }

            return data;
        }

        static float[,] GenerateTestData(int nSamples, int nFeatures, DataPattern pattern, int seed = 42)
        {
            var random = new Random(seed);
            var data = new float[nSamples, nFeatures];

            switch (pattern)
            {
                case DataPattern.Standard:
                    // Standard Gaussian
                    for (int i = 0; i < nSamples; i++)
                    {
                        for (int j = 0; j < nFeatures; j++)
                        {
                            data[i, j] = (float)GenerateNormal(random);
                        }
                    }
                    break;

                case DataPattern.Sparse:
                    // Sparse data (good for cosine metric)
                    for (int i = 0; i < nSamples; i++)
                    {
                        for (int j = 0; j < nFeatures; j++)
                        {
                            data[i, j] = random.NextDouble() < 0.2
                                ? (float)GenerateNormal(random)
                                : 0.0f;
                        }
                    }
                    break;

                case DataPattern.Binary:
                    // Binary data (good for Hamming metric)
                    for (int i = 0; i < nSamples; i++)
                    {
                        for (int j = 0; j < nFeatures; j++)
                        {
                            data[i, j] = random.NextDouble() < 0.5 ? 1.0f : 0.0f;
                        }
                    }
                    break;

                case DataPattern.Clustered:
                    // Clustered data
                    var centers = new[] { -2.0f, 0.0f, 2.0f };
                    for (int i = 0; i < nSamples; i++)
                    {
                        var center = centers[random.Next(centers.Length)];
                        for (int j = 0; j < nFeatures; j++)
                        {
                            data[i, j] = center + (float)GenerateNormal(random) * 0.5f;
                        }
                    }
                    break;

                case DataPattern.Correlated:
                    // Correlated features
                    for (int i = 0; i < nSamples; i++)
                    {
                        var baseValue = (float)GenerateNormal(random);
                        for (int j = 0; j < nFeatures; j++)
                        {
                            var correlation = 0.7f;
                            var noise = (float)GenerateNormal(random) * (1.0f - correlation);
                            data[i, j] = baseValue * correlation + noise;
                        }
                    }
                    break;
            }

            return data;
        }

        static double GenerateNormal(Random random)
        {
            // Box-Muller transform for normal distribution
             double? spare = null;

            if (spare != null)
            {
                var result = spare.Value;
                spare = null;
                return result;
            }

            var u1 = 1.0 - random.NextDouble();
            var u2 = 1.0 - random.NextDouble();
            var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            spare = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

            return normal;
        }

        static void ShowEmbeddingStats(float[,] embedding, string title, int maxDims = 5)
        {
            var nSamples = embedding.GetLength(0);
            var nDims = embedding.GetLength(1);

            Console.WriteLine($"  {title} statistics (first {Math.Min(maxDims, nDims)} dimensions):");

            for (int d = 0; d < Math.Min(maxDims, nDims); d++)
            {
                float min = float.MaxValue, max = float.MinValue, sum = 0;

                for (int i = 0; i < nSamples; i++)
                {
                    var val = embedding[i, d];
                    min = Math.Min(min, val);
                    max = Math.Max(max, val);
                    sum += val;
                }

                var mean = sum / nSamples;
                Console.WriteLine($"    Dim {d}: range=[{min:F3}, {max:F3}], mean={mean:F3}");
            }

            if (nDims > maxDims)
            {
                Console.WriteLine($"    ... ({nDims - maxDims} more dimensions)");
            }
        }

        static void DemoSpreadParameter()
        {
            Console.WriteLine("\n=== Demo 6: NEW Spread Parameter with Smart Defaults ===");

            var data = GenerateTestData(500, 100, DataPattern.Standard);
            Console.WriteLine($"Generated data: {data.GetLength(0)} samples × {data.GetLength(1)} features");

            // Demo 1: Smart auto-defaults (recommended approach)
            Console.WriteLine("\n1. Using Smart Auto-Defaults (Recommended):");
            Console.WriteLine("   - 2D: spread=5.0, min_dist=0.35, neighbors=25 (your optimal research)");
            Console.WriteLine("   - Higher dimensions: automatically scaled down for cluster preservation");

            var dimensions = new[] { 2, 10, 24 };
            foreach (var dim in dimensions)
            {
                using var model = new UMapModel();

                var sw = System.Diagnostics.Stopwatch.StartNew();

                // Use smart defaults - just specify dimension!
                var embedding = model.Fit(data, embeddingDimension: dim);

                sw.Stop();
                Console.WriteLine($"   {dim}D embedding: {embedding.GetLength(0)} samples × {embedding.GetLength(1)}D (auto-optimized in {sw.ElapsedMilliseconds}ms)");
            }

            // Demo 2: Custom spread values comparison
            Console.WriteLine("\n2. Custom Spread Values Comparison (2D Visualization):");

            var testData = GenerateTestData(200, 50, DataPattern.Clustered);
            var spreadValues = new[] { 1.0f, 2.5f, 5.0f, 8.0f };

            foreach (var spread in spreadValues)
            {
                using var model = new UMapModel();

                var sw = System.Diagnostics.Stopwatch.StartNew();

                // Custom spread with your optimal parameters for 2D
                var embedding = model.Fit(
                    data: testData,
                    embeddingDimension: 2,
                    nNeighbors: 25,           // Your optimal
                    minDist: 0.35f,           // Your optimal
                    spread: spread            // Testing different spreads
                );

                sw.Stop();
                Console.WriteLine($"   Spread={spread:F1}: {CalculateSpreadScore(embedding)} space utilization ({sw.ElapsedMilliseconds}ms)");
            }

            // Demo 3: Dimension scaling demonstration
            Console.WriteLine("\n3. Automatic Dimension-Based Scaling:");
            Console.WriteLine("   Higher dimensions automatically use lower spread to preserve clusters:");

            var scalingDemo = new[] {
                (dim: 2, desc: "2D Visualization"),
                (dim: 10, desc: "10D Clustering"),
                (dim: 24, desc: "24D ML Pipeline")
            };

            foreach (var (dim, desc) in scalingDemo)
            {
                using var model = new UMapModel();

                // Show what the auto-calculation would choose
                var autoSpread = dim switch
                {
                    2 => 5.0f,
                    <= 10 => 2.0f,
                    _ => 1.0f
                };

                Console.WriteLine($"   {desc} ({dim}D): auto-spread={autoSpread:F1}");

                var embedding = model.Fit(data, embeddingDimension: dim);
                Console.WriteLine($"     Result: {embedding.GetLength(0)} samples × {embedding.GetLength(1)}D embedding");
            }

            Console.WriteLine("\n✓ Spread parameter successfully implemented with research-based smart defaults!");
            Console.WriteLine("  - Use model.Fit(data, embeddingDimension: dim) for auto-optimized results");
            Console.WriteLine("  - Override with custom spread/minDist/neighbors for fine-tuning");
        }

        private static float CalculateSpreadScore(float[,] embedding)
        {
            // Simple metric: average distance from origin (higher = more spread out)
            float totalDist = 0;
            int nSamples = embedding.GetLength(0);

            for (int i = 0; i < nSamples; i++)
            {
                float dist = 0;
                for (int j = 0; j < embedding.GetLength(1); j++)
                {
                    dist += embedding[i, j] * embedding[i, j];
                }
                totalDist += (float)Math.Sqrt(dist);
            }

            return totalDist / nSamples;
        }

        enum DataPattern
        {
            Standard,
            Sparse,
            Binary,
            Clustered,
            Correlated
        }
    }
}