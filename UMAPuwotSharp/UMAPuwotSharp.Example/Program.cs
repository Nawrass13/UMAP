using System;
using System.IO;
using UMAPuwotSharp;

namespace UMAPuwotSharp.Example
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("=== UMAPuwotSharp C# Wrapper Example ===");
                Console.WriteLine();

                // Generate sample data
                var trainData = GenerateTestData(100, 10, seed: 42);
                var newData = GenerateTestData(20, 10, seed: 123);

                Console.WriteLine($"Generated training data: {trainData.GetLength(0)} samples, {trainData.GetLength(1)} features");
                Console.WriteLine($"Generated new data: {newData.GetLength(0)} samples, {newData.GetLength(1)} features");
                Console.WriteLine();

                // === TRAINING AND EMBEDDING ===
                Console.WriteLine("Step 1: Training UMAP model...");
                
                using var model = new UMapModel();
                var embedding = model.Fit(trainData, nNeighbors: 15, minDist: 0.1f, nEpochs: 200);
                
                Console.WriteLine($"✓ Training completed! Embedding shape: [{embedding.GetLength(0)}, {embedding.GetLength(1)}]");
                Console.WriteLine($"Model info: {model.ModelInfo}");
                Console.WriteLine();

                // Print first few embedding coordinates
                Console.WriteLine("First 5 training embeddings:");
                for (int i = 0; i < Math.Min(5, embedding.GetLength(0)); i++)
                {
                    Console.WriteLine($"  Sample {i}: ({embedding[i, 0]:F3}, {embedding[i, 1]:F3})");
                }
                Console.WriteLine();

                // === SAVE MODEL ===
                Console.WriteLine("Step 2: Saving model...");
                
                const string modelPath = "umap_model_csharp.bin";
                model.Save(modelPath);
                
                Console.WriteLine($"✓ Model saved to: {modelPath}");
                Console.WriteLine($"File size: {new FileInfo(modelPath).Length} bytes");
                Console.WriteLine();

                // === LOAD MODEL ===
                Console.WriteLine("Step 3: Loading model...");
                
                using var loadedModel = UMapModel.Load(modelPath);
                Console.WriteLine($"✓ Model loaded successfully!");
                Console.WriteLine($"Loaded model info: {loadedModel.ModelInfo}");
                Console.WriteLine();

                // === TRANSFORM NEW DATA ===
                Console.WriteLine("Step 4: Transforming new data...");
                
                var newEmbedding = loadedModel.Transform(newData);
                
                Console.WriteLine($"✓ Transform completed! New embedding shape: [{newEmbedding.GetLength(0)}, {newEmbedding.GetLength(1)}]");
                Console.WriteLine();

                // Print first few new embeddings
                Console.WriteLine("First 5 transformed embeddings:");
                for (int i = 0; i < Math.Min(5, newEmbedding.GetLength(0)); i++)
                {
                    Console.WriteLine($"  New sample {i}: ({newEmbedding[i, 0]:F3}, {newEmbedding[i, 1]:F3})");
                }
                Console.WriteLine();

                // === VALIDATION ===
                Console.WriteLine("Step 5: Validation...");
                
                ValidateEmbeddings(embedding, "Training");
                ValidateEmbeddings(newEmbedding, "Transformed");
                
                // === CLEANUP ===
                Console.WriteLine("Step 6: Cleanup...");
                
                if (File.Exists(modelPath))
                {
                    File.Delete(modelPath);
                    Console.WriteLine($"✓ Cleaned up model file: {modelPath}");
                }

                Console.WriteLine();
                Console.WriteLine("=== All tests completed successfully! ===");
                Console.WriteLine();
                Console.WriteLine("UMAPuwotSharp C# wrapper features demonstrated:");
                Console.WriteLine("✓ Cross-platform support (Windows/Linux)");
                Console.WriteLine("✓ Model training and embedding");
                Console.WriteLine("✓ Model persistence (save/load)");
                Console.WriteLine("✓ Transform new data");
                Console.WriteLine("✓ Proper error handling and memory management");
                Console.WriteLine("✓ Clean, type-safe C# API");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Type: {ex.GetType().Name}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner: {ex.InnerException.Message}");
                }
                Environment.Exit(1);
            }
        }

        /// <summary>
        /// Generates synthetic test data for demonstration
        /// </summary>
        static float[,] GenerateTestData(int samples, int features, int seed = 42)
        {
            var random = new Random(seed);
            var data = new float[samples, features];
            
            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    // Generate normally distributed data
                    data[i, j] = (float)(random.NextGaussian() * 1.0 + 0.0);
                }
            }
            
            return data;
        }

        /// <summary>
        /// Validates embedding results
        /// </summary>
        static void ValidateEmbeddings(float[,] embedding, string name)
        {
            var nSamples = embedding.GetLength(0);
            
            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;
            
            for (int i = 0; i < nSamples; i++)
            {
                var x = embedding[i, 0];
                var y = embedding[i, 1];
                
                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
            }
            
            Console.WriteLine($"{name} embedding bounds: X=[{minX:F3}, {maxX:F3}], Y=[{minY:F3}, {maxY:F3}]");
            
            var rangeX = maxX - minX;
            var rangeY = maxY - minY;
            
            if (rangeX > 0 && rangeY > 0)
            {
                Console.WriteLine($"✓ {name} embeddings have valid ranges");
            }
            else
            {
                Console.WriteLine($"⚠ {name} embeddings may have issues (zero range)");
            }
        }
    }

    /// <summary>
    /// Extension methods for generating random numbers
    /// </summary>
    public static class RandomExtensions
    {
        /// <summary>
        /// Generates a normally distributed random number using Box-Muller transform
        /// </summary>
        public static double NextGaussian(this Random random, double mean = 0.0, double stdDev = 1.0)
        {
            // Box-Muller transform
            double u1 = 1.0 - random.NextDouble(); // uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
    }
}