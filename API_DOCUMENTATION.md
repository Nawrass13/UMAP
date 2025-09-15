# Enhanced UMAP API Documentation

## Overview
Complete API documentation for the Enhanced UMAP implementation with revolutionary HNSW k-NN optimization and smart spread parameter support. This document covers both C++ and C# APIs with comprehensive examples and best practices.

## 🚀 New HNSW Optimization Features

The most significant enhancement is the **HNSW (Hierarchical Navigable Small World) k-NN optimization** that provides 50-2000x performance improvement while maintaining accuracy (MSE < 0.01).

### Key Benefits
- **Training Speed**: 50-2000x faster for large datasets
- **Transform Speed**: <3ms per sample vs 50-200ms
- **Memory Usage**: 80-85% reduction
- **Accuracy**: Identical to exact methods (MSE < 0.01)
- **Auto-optimization**: Smart metric-based selection

---

## C# API Reference

### Core Classes

#### UMapModel
Main class for UMAP training and transformations.

```csharp
using UMAPuwotSharp;

// Create model instance
using var model = new UMapModel();
```

### Training Methods

#### Fit() - Standard Training with Spread Parameter
```csharp
public float[,] Fit(float[,] data,
                    int embeddingDimension = 2,
                    int? nNeighbors = null,
                    float? minDist = null,
                    float? spread = null,        // NEW in v3.1.1
                    int nEpochs = 300,
                    DistanceMetric metric = DistanceMetric.Euclidean,
                    bool forceExactKnn = false)
```

**New Parameters:**
- `spread`: **NEW v3.1.1** - Global scale of embedding (auto-optimized by dimension)
- `forceExactKnn`: Force exact brute-force k-NN instead of HNSW optimization

**Smart Defaults by Dimension:**
- **2D**: spread=5.0, minDist=0.35, nNeighbors=25 (optimal for visualization)
- **10D**: spread=2.0, minDist=0.15, nNeighbors=15 (balanced)
- **24D+**: spread=1.0, minDist=0.1, nNeighbors=15 (compact for ML pipelines)

**Examples:**
```csharp
// Smart automatic defaults (recommended)
var embedding2D = model.Fit(data, embeddingDimension: 2);  // Auto: spread=5.0
var embedding27D = model.Fit(data, embeddingDimension: 27); // Auto: spread=1.0

// Manual spread control for fine-tuning
var customEmbedding = model.Fit(data,
    embeddingDimension: 2,
    spread: 5.0f,           // t-SNE-like space-filling
    minDist: 0.35f,         // Minimum point separation
    nNeighbors: 25,         // Optimal neighborhood size
    metric: DistanceMetric.Cosine,
    forceExactKnn: false);  // Use HNSW for massive speedup

// Force exact computation (for validation)
var exactEmbedding = model.Fit(data, forceExactKnn: true);
```

#### FitWithProgress() - Training with Progress Reporting and Spread Parameter
```csharp
public float[,] FitWithProgress(float[,] data,
                               ProgressCallback progressCallback,
                               int embeddingDimension = 2,
                               int? nNeighbors = null,
                               float? minDist = null,
                               float? spread = null,        // NEW v3.1.1
                               int nEpochs = 300,
                               DistanceMetric metric = DistanceMetric.Euclidean,
                               bool forceExactKnn = false)
```

**Enhanced Progress Reporting:**
The callback now receives detailed phase information:
- "Z-Normalization" - Data preprocessing
- "Building HNSW index" - HNSW construction
- "HNSW k-NN Graph" - Fast approximate k-NN
- "Exact k-NN Graph" - Traditional brute-force (when needed)
- "Optimization" - SGD training phases

**Example:**
```csharp
var embedding = model.FitWithProgress(data,
    progressCallback: (epoch, total, percent) =>
    {
        Console.WriteLine($"Training progress: {percent:F1}%");
    },
    embeddingDimension: 2,
    metric: DistanceMetric.Euclidean,
    forceExactKnn: false);  // HNSW optimization enabled
```

### Distance Metrics with HNSW Support

#### Supported Metrics
```csharp
public enum DistanceMetric
{
    Euclidean = 0,    // ✅ HNSW accelerated
    Cosine = 1,       // ✅ HNSW accelerated
    Manhattan = 2,    // ✅ HNSW accelerated
    Correlation = 3,  // ⚡ Falls back to exact
    Hamming = 4       // ⚡ Falls back to exact
}
```

**Performance by Metric:**

| Metric | HNSW Support | Typical Speedup | Best Use Case |
|--------|--------------|-----------------|---------------|
| **Euclidean** | ✅ Full | 50-200x | General-purpose data |
| **Cosine** | ✅ Full | 30-150x | High-dimensional sparse data |
| **Manhattan** | ✅ Full | 40-180x | Outlier-robust applications |
| **Correlation** | ⚡ Fallback | 1x | Time series, correlated features |
| **Hamming** | ⚡ Fallback | 1x | Binary, categorical data |

### Transform Methods

#### Transform() - Standard Transform
```csharp
public float[,] Transform(float[,] newData)
```

**Performance:** <3ms per sample with HNSW optimization

**Example:**
```csharp
// Train model first
var embedding = model.Fit(trainingData, forceExactKnn: false);

// Transform new data (lightning fast with HNSW)
var newSample = new float[1, features];
// ... populate newSample ...
var result = model.Transform(newSample);  // <3ms typical
```

#### TransformDetailed() - Enhanced Transform with Safety Analysis
```csharp
public TransformResult TransformDetailed(float[,] newData)
```

**Returns enhanced safety information:**
```csharp
public class TransformResult
{
    public float[] ProjectionCoordinates;      // Embedding position
    public int[] NearestNeighborIndices;       // Training sample indices
    public float[] NearestNeighborDistances;   // Distances in original space
    public float ConfidenceScore;              // 0.0-1.0 safety confidence
    public OutlierLevel Severity;              // 5-level outlier detection
    public float PercentileRank;               // 0-100% distance ranking
    public float ZScore;                       // Standard deviations from mean
}

public enum OutlierLevel
{
    Normal = 0,         // ≤ 95th percentile
    Unusual = 1,        // 95th-99th percentile
    Mild = 2,           // 99th percentile to 2.5σ
    Extreme = 3,        // 2.5σ to 4σ
    NoMansLand = 4      // > 4σ (high risk)
}
```

### Model Persistence

#### SaveModel() / LoadModel()
```csharp
// Save trained model with HNSW indices
model.SaveModel("my_model.bin");

// Load model (HNSW indices preserved)
using var loadedModel = new UMapModel();
loadedModel.LoadModel("my_model.bin");

// Transform using loaded model (still fast!)
var result = loadedModel.Transform(newData);
```

**HNSW Persistence:** HNSW indices are automatically saved and loaded, maintaining fast transform performance.

### Model Information

#### ModelInfo Property
```csharp
public UMapModelInfo ModelInfo { get; }

public class UMapModelInfo
{
    public int TrainingSamples;      // Number of training samples
    public int InputDimension;       // Original feature dimension
    public int OutputDimension;      // Embedding dimension
    public int NeighborsUsed;        // k-NN parameter used
    public float MinDistanceUsed;    // min_dist parameter used
    public string MetricName;        // Distance metric used
    public bool IsHNSWOptimized;     // NEW - Whether HNSW was used
}
```

**Example:**
```csharp
var info = model.ModelInfo;
Console.WriteLine($"Model: {info.TrainingSamples} samples, " +
                 $"{info.InputDimension}D → {info.OutputDimension}D");
Console.WriteLine($"Metric: {info.MetricName}, HNSW: {info.IsHNSWOptimized}");
```

---

## C++ API Reference

### Core Functions

#### uwot_fit_with_enhanced_progress()
**NEW Enhanced Training Function**
```cpp
int uwot_fit_with_enhanced_progress(
    UwotModel* model,
    float* data, int n_obs, int n_dim,
    int embedding_dim, int n_neighbors, float min_dist, int n_epochs,
    UwotMetric metric, float* embedding,
    uwot_progress_callback_v2 progress_callback,
    int force_exact_knn = 0);  // NEW parameter
```

**Enhanced Progress Callback:**
```cpp
typedef void (*uwot_progress_callback_v2)(
    const char* phase,        // Phase name
    int current,              // Current progress
    int total,                // Total items
    float percent,            // Progress percentage
    const char* message       // Time estimates, warnings, or NULL
);
```

**Example:**
```cpp
void progress_callback(const char* phase, int current, int total,
                      float percent, const char* message) {
    printf("[%s] %.1f%% (%d/%d)", phase, percent, current, total);
    if (message) printf(" - %s", message);
    printf("\n");
}

// Use HNSW optimization (recommended)
int result = uwot_fit_with_enhanced_progress(
    model, data, n_obs, n_dim, embedding_dim,
    15, 0.1f, 300, UWOT_METRIC_EUCLIDEAN,
    embedding, progress_callback,
    0  // force_exact_knn = 0 (use HNSW)
);
```

### Distance Metrics

#### UwotMetric Enum
```cpp
typedef enum {
    UWOT_METRIC_EUCLIDEAN = 0,    // HNSW accelerated
    UWOT_METRIC_COSINE = 1,       // HNSW accelerated
    UWOT_METRIC_MANHATTAN = 2,    // HNSW accelerated
    UWOT_METRIC_CORRELATION = 3,  // Exact fallback
    UWOT_METRIC_HAMMING = 4       // Exact fallback
} UwotMetric;
```

### Enhanced Transform

#### uwot_transform_detailed()
**NEW Enhanced Transform with Safety Analysis**
```cpp
int uwot_transform_detailed(
    UwotModel* model,
    float* new_data, int n_new_obs, int n_dim,
    float* embedding,              // Output coordinates
    int* nn_indices,              // Nearest neighbor indices
    float* nn_distances,          // Nearest neighbor distances
    float* confidence_score,      // Confidence (0.0-1.0)
    int* outlier_level,          // UwotOutlierLevel enum
    float* percentile_rank,      // Percentile ranking (0-100)
    float* z_score               // Z-score (std devs from mean)
);
```

**Outlier Detection Levels:**
```cpp
typedef enum {
    UWOT_OUTLIER_NORMAL = 0,      // ≤ 95th percentile
    UWOT_OUTLIER_UNUSUAL = 1,     // 95th-99th percentile
    UWOT_OUTLIER_MILD = 2,        // 99th percentile to 2.5σ
    UWOT_OUTLIER_EXTREME = 3,     // 2.5σ to 4σ
    UWOT_OUTLIER_NOMANSLAND = 4   // > 4σ
} UwotOutlierLevel;
```

---

## Performance Optimization Guide

### When to Use HNSW vs Exact

#### Use HNSW (forceExactKnn = false) When:
✅ **Large datasets** (≥1,000 samples)
✅ **Production applications** requiring fast transforms
✅ **Supported metrics** (Euclidean, Cosine, Manhattan)
✅ **Memory-constrained** environments
✅ **Real-time processing** needs

#### Use Exact (forceExactKnn = true) When:
⚡ **Small datasets** (<1,000 samples)
⚡ **Validation/research** requiring perfect accuracy
⚡ **Unsupported metrics** (though auto-fallback handles this)
⚡ **Debugging** HNSW vs exact differences

### Optimal Parameters by Dataset Size

| Dataset Size | Recommended Settings |
|-------------|---------------------|
| **< 1,000** | `forceExactKnn: true`, any metric |
| **1,000-10,000** | `forceExactKnn: false`, `nEpochs: 200-500` |
| **10,000-50,000** | `forceExactKnn: false`, `nEpochs: 300-800` |
| **50,000+** | `forceExactKnn: false`, `nEpochs: 500-1000` |

### Memory Optimization Tips

1. **Use HNSW**: 80-85% memory reduction
2. **Choose appropriate embedding dimensions**: Higher dimensions = more memory
3. **Batch processing**: Process large datasets in chunks
4. **Model persistence**: Save/load models instead of retraining

---

## Error Handling

### Common Error Codes
```cpp
#define UWOT_SUCCESS 0
#define UWOT_ERROR_INVALID_PARAMS -1
#define UWOT_ERROR_MEMORY -2
#define UWOT_ERROR_NOT_IMPLEMENTED -3
#define UWOT_ERROR_FILE_IO -4
#define UWOT_ERROR_MODEL_NOT_FITTED -5
#define UWOT_ERROR_INVALID_MODEL_FILE -6
```

### Error Messages
```cpp
const char* uwot_get_error_message(int error_code);
```

### C# Exception Handling
```csharp
try
{
    var embedding = model.Fit(data, forceExactKnn: false);
}
catch (ArgumentNullException ex)
{
    // Handle null data
}
catch (ArgumentException ex)
{
    // Handle invalid parameters (dimensions, etc.)
}
catch (InvalidOperationException ex)
{
    // Handle model state errors
}
```

---

## Best Practices

### Training Best Practices

1. **Start with HNSW**: Default `forceExactKnn: false` for best performance
2. **Choose optimal metrics**: Euclidean/Cosine/Manhattan get HNSW acceleration
3. **Use progress callbacks**: Monitor training with enhanced phase reporting
4. **Validate accuracy**: Compare HNSW vs exact on small subset when needed
5. **Save trained models**: Avoid retraining with model persistence

### Transform Best Practices

1. **Batch similar data**: Group similar transforms for efficiency
2. **Monitor outlier levels**: Use `TransformDetailed()` for production safety
3. **Set confidence thresholds**: Define acceptable confidence scores
4. **Handle edge cases**: Plan for `NoMansLand` outlier detections

### Production Deployment

1. **Use HNSW models**: Deploy with `forceExactKnn: false` for speed
2. **Implement safety checks**: Monitor outlier levels and confidence scores
3. **Set up monitoring**: Track transform times and outlier rates
4. **Plan fallbacks**: Handle extreme outliers appropriately
5. **Test cross-platform**: Verify performance on target deployment platforms

---

## Migration from Previous Versions

### From v2.x to v3.x (HNSW)

**API Changes:**
- Added `forceExactKnn` parameter to `Fit()` and `FitWithProgress()`
- Enhanced progress callback with phase information
- Added `TransformDetailed()` method
- Added `UwotOutlierLevel` enum

**Performance Improvements:**
- Training: 50-2000x faster with HNSW
- Transform: <3ms per sample
- Memory: 80-85% reduction

**Migration Code:**
```csharp
// Old v2.x code
var embedding = model.Fit(data, embeddingDimension: 2);

// New v3.x code (backward compatible)
var embedding = model.Fit(data,
    embeddingDimension: 2,
    forceExactKnn: false);  // Add for HNSW optimization

// Enhanced features (new in v3.x)
var result = model.TransformDetailed(newData);
Console.WriteLine($"Outlier level: {result.Severity}");
```

---

## Support and Resources

### Documentation
- **API Reference**: This document
- **Build Instructions**: `/uwot_pure_cpp/build_instructions.md`
- **Test Documentation**: `/UMAPuwotSharp.Tests/README.md`
- **HNSW Validation**: `/uwot_pure_cpp/test_validation_readme.md`

### Performance Testing
- **C++ Tests**: `ctest` in build directory
- **C# Tests**: `dotnet test` in UMAPuwotSharp directory
- **Benchmarks**: Included in test suites with accuracy validation

### Getting Help
1. **Check examples**: Review provided code examples
2. **Run tests**: Validate installation with test suites
3. **Review benchmarks**: Compare your performance with provided benchmarks
4. **Check parameters**: Verify data types and parameter ranges

This enhanced API provides unprecedented performance for C# UMAP applications while maintaining full backward compatibility and adding comprehensive safety features for production deployment.