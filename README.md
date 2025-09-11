# Enhanced High-Performance UMAP C++ Implementation with C# Wrapper

## Project Motivation

This project was created specifically because existing NuGet packages and open-source C# implementations for UMAP lack critical functionality required for production machine learning applications:

- **No model persistence**: Cannot save trained UMAP models for reuse
- **No true transform capability**: Cannot project new data points using existing trained models
- **Limited dimensionality support**: Restricted to 2D or 3D embeddings
- **Missing distance metrics**: Only basic Euclidean distance support
- **No progress reporting**: No feedback during long training processes
- **Limited production readiness**: Missing essential features for real-world deployment

This implementation addresses these fundamental gaps by providing complete model persistence, authentic transform functionality, arbitrary embedding dimensions (1D-50D), multiple distance metrics, and progress reporting based on the proven uwot algorithm.

## Overview

A complete, production-ready UMAP (Uniform Manifold Approximation and Projection) implementation based on the high-performance [uwot R package](https://github.com/jlmelville/uwot), providing both standalone C++ libraries and cross-platform C# integration with **enhanced features not available in other C# UMAP libraries**.

## Enhanced Features (NEW)

### 🚀 **Arbitrary Embedding Dimensions (1D to 50D)**
```csharp
// Standard 2D visualization
var embedding2D = model.Fit(data, embeddingDimension: 2);

// High-dimensional embeddings for feature extraction
var embedding27D = model.Fit(data, embeddingDimension: 27);  // Perfect for specialized ML pipelines
var embedding50D = model.Fit(data, embeddingDimension: 50);  // Maximum supported

// Even 1D embeddings for specialized use cases
var embedding1D = model.Fit(data, embeddingDimension: 1);
```

### 📊 **Multiple Distance Metrics**
Choose the optimal distance metric for your data type:

```csharp
// Euclidean (default) - general-purpose data
var euclidean = model.Fit(data, metric: DistanceMetric.Euclidean);

// Cosine - excellent for high-dimensional sparse data (text, images)
var cosine = model.Fit(data, metric: DistanceMetric.Cosine);

// Manhattan - robust to outliers
var manhattan = model.Fit(data, metric: DistanceMetric.Manhattan);

// Correlation - measures linear relationships, good for time series
var correlation = model.Fit(data, metric: DistanceMetric.Correlation);

// Hamming - for binary or categorical data
var hamming = model.Fit(data, metric: DistanceMetric.Hamming);
```

### ⏱️ **Real-Time Progress Reporting**
Get live feedback during training with customizable progress callbacks:

```csharp
var embedding = model.FitWithProgress(
    data,
    progressCallback: (epoch, totalEpochs, percent) =>
    {
        Console.WriteLine($"Training: {percent:F1}% (Epoch {epoch}/{totalEpochs})");
        // Update UI progress bar, log to file, etc.
    },
    embeddingDimension: 27,
    nEpochs: 500
);
```

### 🔧 **Complete API Example**
```csharp
using UMAPuwotSharp;

// Create model with enhanced features
using var model = new UMapModel();

// Train with progress reporting and custom dimensions/metrics
var embedding = model.FitWithProgress(
    data: trainingData,
    progressCallback: (epoch, total, percent) => 
    {
        if (epoch % 50 == 0) 
            Console.WriteLine($"Progress: {percent:F0}%");
    },
    embeddingDimension: 27,        // Any dimension 1-50
    nNeighbors: 20,
    minDist: 0.05f,
    nEpochs: 300,
    metric: DistanceMetric.Cosine  // Optimal for your data type
);

// Access comprehensive model information
var info = model.ModelInfo;
Console.WriteLine($"Model: {info.TrainingSamples} samples, " +
                 $"{info.InputDimension}D → {info.OutputDimension}D, " +
                 $"metric: {info.MetricName}");

// Save and load models (unique to this implementation)
model.Save("enhanced_model.umap");
using var loadedModel = UMapModel.Load("enhanced_model.umap");

// Transform new data using saved model
var newEmbedding = loadedModel.Transform(newData);
```

## Prebuilt Binaries Available

**Ready-to-use enhanced native libraries are included for immediate deployment:**

- **Windows x64**: `uwot.dll` - Enhanced version with multi-metric support and progress reporting
- **Linux x64**: `libuwot.so` - Complete feature parity with Windows version

These prebuilt binaries provide:
- **All enhanced features**: Multi-dimensional support, multiple metrics, progress reporting
- **Production stability**: Thoroughly tested across multiple environments
- **Optimized performance**: Compiled with release optimizations and OpenMP support
- **Immediate deployment**: No compilation required - works out of the box
- **Cross-platform compatibility**: Automatic runtime detection selects the correct native library

## What is UMAP?

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that can be used for visualization, feature extraction, and preprocessing of high-dimensional data. Unlike many other dimensionality reduction algorithms, UMAP excels at preserving both local and global structure in the data.

![UMAP 3D Visualization](Other/rot3DUMAP_alltp_360.gif)
*Example: 3D UMAP embedding rotation showing preserved data structure and clustering*

**For an excellent interactive explanation of UMAP, see: [Understanding UMAP](https://pair-code.github.io/understanding-umap/)**

### UMAP Advantages

- **Preserves local structure**: Keeps similar points close together
- **Maintains global structure**: Preserves overall data topology effectively
- **Scalable**: Handles large datasets efficiently
- **Fast**: High-performance implementation optimized for speed
- **Versatile**: Works well for visualization, clustering, and as preprocessing
- **Deterministic**: Consistent results across runs (with fixed random seed)
- **Flexible**: Supports various distance metrics and custom parameters
- **Multi-dimensional**: Supports any embedding dimension from 1D to 50D

### UMAP Limitations

- **Parameter sensitivity**: Results can vary significantly with parameter changes
- **Interpretation challenges**: Distances in embedding space don't always correspond to original space
- **Memory usage**: Can be memory-intensive for very large datasets (e.g., 100k samples × 300 features typically requires ~4-8GB RAM during processing, depending on n_neighbors parameter)
- **Mathematical complexity**: The underlying theory is more complex than simpler methods like PCA

## Why This Enhanced Implementation?

### Critical Gap in Existing C# Libraries

Currently available UMAP libraries for C# (including popular NuGet packages) have significant limitations:

- **No model persistence**: Cannot save trained models for later use
- **No true transform capability**: Cannot embed new data points using pre-trained models
- **Limited dimensionality**: Usually restricted to 2D or 3D embeddings only
- **Single distance metric**: Only Euclidean distance supported
- **No progress feedback**: No way to monitor training progress
- **Performance issues**: Often slower implementations without the optimizations of uwot
- **Limited parameter support**: Missing important UMAP parameters and customization options

This enhanced implementation addresses ALL these gaps by providing:

- **True model persistence**: Save and load trained UMAP models in efficient binary format
- **Authentic transform functionality**: Embed new data using existing models (essential for production ML pipelines)
- **Arbitrary dimensions**: Support for 1D to 50D embeddings including specialized dimensions like 27D
- **Multiple distance metrics**: Five different metrics optimized for different data types
- **Real-time progress reporting**: Live feedback during training with customizable callbacks
- **High performance**: Based on the optimized uwot implementation used in production R environments
- **Complete parameter support**: Full access to UMAP's hyperparameters and options

## Enhanced Use Cases

### 27D Embeddings for Specialized Applications
```csharp
// Feature extraction for downstream ML models
var features27D = model.Fit(highDimData, embeddingDimension: 27, metric: DistanceMetric.Cosine);
// Use as input to neural networks, clustering algorithms, etc.
```

### Multi-Metric Analysis
```csharp
// Compare different distance metrics for the same data
var metrics = new[] { 
    DistanceMetric.Euclidean, 
    DistanceMetric.Cosine, 
    DistanceMetric.Manhattan 
};

foreach (var metric in metrics)
{
    var embedding = model.Fit(data, metric: metric, embeddingDimension: 2);
    // Analyze which metric produces the best clustering/visualization
}
```

### Production ML Pipelines with Progress Monitoring
```csharp
// Long-running training with progress tracking
var embedding = model.FitWithProgress(
    largeDataset,
    progressCallback: (epoch, total, percent) =>
    {
        // Log to monitoring system
        logger.LogInformation($"UMAP Training: {percent:F1}% complete");
        
        // Update database/UI
        await UpdateTrainingProgress(percent);
    },
    embeddingDimension: 10,
    nEpochs: 1000,
    metric: DistanceMetric.Correlation
);
```

## Projects Structure

### uwot_pure_cpp
Enhanced standalone C++ UMAP library extracted and adapted from the uwot R package:

- **Model Training**: Complete UMAP algorithm with customizable parameters
- **Multiple Distance Metrics**: Euclidean, Cosine, Manhattan, Correlation, Hamming
- **Arbitrary Dimensions**: Support for 1D to 50D embeddings
- **Progress Reporting**: Real-time training feedback with callback support
- **Model Persistence**: Save/load functionality using efficient binary format
- **Transform Support**: Embed new data points using pre-trained models
- **Cross-Platform**: Builds on Windows (Visual Studio) and Linux (GCC/Docker)
- **Memory Safe**: Proper resource management and error handling
- **OpenMP Support**: Parallel processing for improved performance

### UMAPuwotSharp
Enhanced production-ready C# wrapper providing .NET integration:

- **Enhanced Type-Safe API**: Clean C# interface with progress reporting support
- **Multi-Dimensional Support**: Full API for 1D-50D embeddings
- **Distance Metric Selection**: Complete enum and validation for all metrics
- **Progress Callbacks**: .NET delegate integration for real-time feedback
- **Cross-Platform**: Automatic Windows/Linux runtime detection
- **NuGet Ready**: Complete package with embedded enhanced native libraries
- **Memory Management**: Proper IDisposable implementation
- **Error Handling**: Comprehensive exception mapping from native errors
- **Model Information**: Rich metadata about fitted models

## Quick Start

### Using Prebuilt Enhanced Binaries (Recommended)

The fastest way to get started with all enhanced features:

```cmd
# Install via NuGet (when published)
dotnet add package UMAPuwotSharp

# Or clone and build the enhanced C# wrapper
git clone https://github.com/78Spinoza/UMAP.git
cd UMAP/UMAPuwotSharp
dotnet build
dotnet run --project UMAPuwotSharp.Example
```

### Complete Enhanced API Example

```csharp
using UMAPuwotSharp;

Console.WriteLine("=== Enhanced UMAP Demo ===");

// Generate sample data
var data = GenerateTestData(1000, 100);

using var model = new UMapModel();

// Train with progress reporting and custom settings
Console.WriteLine("Training 27D embedding with Cosine metric...");

var embedding = model.FitWithProgress(
    data: data,
    progressCallback: (epoch, totalEpochs, percent) =>
    {
        if (epoch % 25 == 0)
            Console.WriteLine($"  Progress: {percent:F0}% (Epoch {epoch}/{totalEpochs})");
    },
    embeddingDimension: 27,           // High-dimensional embedding
    nNeighbors: 20,
    minDist: 0.05f,
    nEpochs: 300,
    metric: DistanceMetric.Cosine     // Optimal for high-dim sparse data
);

// Display comprehensive model information
var info = model.ModelInfo;
Console.WriteLine($"\nModel Info: {info}");
Console.WriteLine($"  Training samples: {info.TrainingSamples}");
Console.WriteLine($"  Input → Output: {info.InputDimension}D → {info.OutputDimension}D");
Console.WriteLine($"  Distance metric: {info.MetricName}");
Console.WriteLine($"  Neighbors: {info.Neighbors}, Min distance: {info.MinimumDistance}");

// Save enhanced model
model.Save("enhanced_model.umap");
Console.WriteLine("Model saved with all enhanced features!");

// Load and transform new data
using var loadedModel = UMapModel.Load("enhanced_model.umap");
var newData = GenerateTestData(100, 100);
var transformedData = loadedModel.Transform(newData);

Console.WriteLine($"Transformed {newData.GetLength(0)} new samples to {transformedData.GetLength(1)}D");
```

### Building Enhanced Version from Source

If you want to build the enhanced native libraries yourself:

**Cross-platform enhanced build (production-ready):**
```cmd
cd uwot_pure_cpp
BuildDockerLinuxWindows.bat
```

This builds the enhanced version with all new features:
- Multi-dimensional support (1D-50D)
- Multiple distance metrics
- Progress reporting infrastructure
- Enhanced model persistence format

## Performance and Compatibility

- **Enhanced algorithms**: All new features optimized for performance
- **Cross-platform**: Windows and Linux support with automatic runtime detection
- **Memory efficient**: Careful resource management even with high-dimensional embeddings
- **Production tested**: Comprehensive test suite validating all enhanced functionality
- **64-bit optimized**: Native libraries compiled for x64 architecture with enhanced feature support
- **Backward compatible**: Models saved with basic features can be loaded by enhanced version

## Enhanced Technical Implementation

This implementation extends the core C++ algorithms from uwot with:

- **Multi-metric distance computation**: Optimized implementations for all five distance metrics
- **Arbitrary dimension support**: Memory-efficient handling of 1D-50D embeddings
- **Progress callback infrastructure**: Thread-safe progress reporting from C++ to C#
- **Enhanced binary model format**: Extended serialization supporting all new features
- **Cross-platform enhanced build system**: CMake with Docker support ensuring feature parity

## Version Information

- **Enhanced Native Libraries**: Based on uwot algorithms with extensive enhancements
- **C# Wrapper**: Version 2.0.0 (UMAPuwotSharp Enhanced)
- **Target Framework**: .NET 8.0
- **Supported Platforms**: Windows x64, Linux x64
- **New Features**: Multi-dimensional (1D-50D), Multi-metric, Progress reporting

## References

1. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
2. **Interactive UMAP Guide**: https://pair-code.github.io/understanding-umap/
3. **uwot R package**: https://github.com/jlmelville/uwot
4. **Original Python UMAP**: https://github.com/lmcinnes/umap

## License

Maintains compatibility with the GPL-3 license of the original uwot package.

---

This enhanced implementation represents the most complete and feature-rich UMAP library available for C#/.NET, providing capabilities that surpass even many Python implementations. The combination of arbitrary embedding dimensions, multiple distance metrics, progress reporting, and complete model persistence makes it ideal for both research and production machine learning applications.