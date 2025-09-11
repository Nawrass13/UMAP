# High-Performance UMAP C++ Implementation with C# Wrapper

## Project Motivation

This project was created specifically because existing NuGet packages and open-source C# implementations for UMAP lack critical functionality required for production machine learning applications:

- **No model persistence**: Cannot save trained UMAP models for reuse
- **No true transform capability**: Cannot project new data points using existing trained models
- **Limited production readiness**: Missing essential features for real-world deployment

This implementation addresses these fundamental gaps by providing complete model persistence and authentic transform functionality based on the proven uwot algorithm.

## Overview

A complete, production-ready UMAP (Uniform Manifold Approximation and Projection) implementation based on the high-performance [uwot R package](https://github.com/jlmelville/uwot), providing both standalone C++ libraries and cross-platform C# integration.

## Prebuilt Binaries Available

**Ready-to-use native libraries are included for immediate deployment:**

- **Windows x64**: `uwot.dll` - Based on uwot CRAN release 0.2.3, optimized for 64-bit Windows systems
- **Linux x64**: `libuwot.so` - Based on uwot CRAN release 0.2.3, compiled for 64-bit Linux distributions

These prebuilt binaries are automatically included in the NuGet package and provide:
- **Production stability**: Based on the stable CRAN release 0.2.3 of uwot
- **Optimized performance**: Compiled with release optimizations and OpenMP support
- **Immediate deployment**: No compilation required - works out of the box
- **Cross-platform compatibility**: Automatic runtime detection selects the correct native library

The binaries include all uwot 0.2.3 features and have been thoroughly tested across multiple Windows and Linux environments. For custom builds or other platforms, full source code and build instructions are provided below.

## What is UMAP?

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that can be used for visualization, feature extraction, and preprocessing of high-dimensional data. Unlike many other dimensionality reduction algorithms, UMAP excels at preserving both local and global structure in the data.

**For an excellent interactive explanation of UMAP, see: [Understanding UMAP](https://pair-code.github.io/understanding-umap/)**

### UMAP Advantages

- **Preserves local structure**: Keeps similar points close together
- **Maintains global structure**: Preserves overall data topology effectively
- **Scalable**: Handles large datasets efficiently
- **Fast**: High-performance implementation optimized for speed
- **Versatile**: Works well for visualization, clustering, and as preprocessing
- **Deterministic**: Consistent results across runs (with fixed random seed)
- **Flexible**: Supports various distance metrics and custom parameters

### UMAP Limitations

- **Parameter sensitivity**: Results can vary significantly with parameter changes
- **Interpretation challenges**: Distances in embedding space don't always correspond to original space
- **Memory usage**: Can be memory-intensive for very large datasets (e.g., 100k samples × 300 features typically requires ~4-8GB RAM during processing, depending on n_neighbors parameter)
- **Mathematical complexity**: The underlying theory is more complex than simpler methods like PCA

## Why This Implementation?

### Critical Gap in Existing C# Libraries

Currently available UMAP libraries for C# (including popular NuGet packages) have significant limitations:

- **No model persistence**: Cannot save trained models for later use
- **No true transform capability**: Cannot embed new data points using pre-trained models
- **Performance issues**: Often slower implementations without the optimizations of uwot
- **Limited parameter support**: Missing important UMAP parameters and customization options

This implementation addresses these gaps by providing:

- **True model persistence**: Save and load trained UMAP models in efficient binary format
- **Authentic transform functionality**: Embed new data using existing models (essential for production ML pipelines)
- **High performance**: Based on the optimized uwot implementation used in production R environments
- **Complete parameter support**: Full access to UMAP's hyperparameters and options

## Projects Structure

### uwot_pure_cpp
Standalone C++ UMAP library extracted and adapted from the uwot R package:

- **Model Training**: Complete UMAP algorithm with customizable parameters
- **Model Persistence**: Save/load functionality using efficient binary format
- **Transform Support**: Embed new data points using pre-trained models
- **Cross-Platform**: Builds on Windows (Visual Studio) and Linux (GCC/Docker)
- **Memory Safe**: Proper resource management and error handling
- **OpenMP Support**: Parallel processing for improved performance

### UMAPuwotSharp
Production-ready C# wrapper providing .NET integration:

- **Type-Safe API**: Clean C# interface following .NET conventions
- **Cross-Platform**: Automatic Windows/Linux runtime detection
- **NuGet Ready**: Complete package with embedded native libraries
- **Memory Management**: Proper IDisposable implementation
- **Error Handling**: Comprehensive exception mapping from native errors

## Key Features

### Model Persistence (Unique to This Implementation)
```csharp
// Train once, save for production use
using var model = new UMapModel();
var embedding = model.Fit(trainingData, nNeighbors: 15, minDist: 0.1f, nEpochs: 200);
model.Save("production_model.bin");

// Later: Fast loading and immediate use
using var productionModel = UMapModel.Load("production_model.bin");
var newEmbedding = productionModel.Transform(newData);
```

### True Transform Capability
Unlike other C# UMAP libraries that require retraining, this implementation provides authentic transform functionality:

```csharp
// Embed new data using existing model structure
var newPoints = loadedModel.Transform(unseenData);
```

This is essential for:
- **Production ML pipelines**: Consistent embedding of new data
- **Real-time applications**: No retraining latency
- **Batch processing**: Process new data using established models
- **A/B testing**: Consistent embeddings across experiments

## Quick Start

### Using Prebuilt Binaries (Recommended)

The fastest way to get started is using the included prebuilt binaries:

```cmd
# Install via NuGet (when published)
dotnet add package UMAPuwotSharp

# Or clone and build the C# wrapper
git clone https://github.com/78Spinoza/UMAP.git
cd UMAP/UMAPuwotSharp
dotnet build
dotnet run --project UMAPuwotSharp.Example
```

The prebuilt binaries (based on uwot 0.2.3) are automatically deployed and require no additional setup.

### Building from Source (Advanced Users)

If you want to build the native libraries yourself, use the provided batch files for the easiest experience, or use CMake directly for custom configurations:

**Easy Build with Batch Files (Recommended):**

**Windows-only build (fastest, no Docker):**
```cmd
cd uwot_pure_cpp
BuildWindows.bat
```

**Cross-platform build (production-ready):**
```cmd
cd uwot_pure_cpp
BuildDockerLinuxWindows.bat
```

The cross-platform build script automatically:
- Builds Windows libraries using Visual Studio
- Builds Linux libraries using Docker (Ubuntu 22.04)
- Creates the UMAPuwotSharp project structure if needed
- Copies native libraries to the correct runtime folders
- Handles the Linux library naming (libuwot.so.1.0 → libuwot.so)
- Runs comprehensive tests on both platforms
- Provides detailed build summary and verification

**Manual CMake (for custom configurations):**
```cmd
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

All dependencies are included in the source code, making this a straightforward build with no external library management required. The project is self-contained and includes:

- **Embedded algorithm headers**: All uwot mathematical implementations
- **Standalone implementations**: Custom smooth k-NN and other algorithms  
- **No external dependencies**: Everything needed is in the source tree
- **Cross-platform compatibility**: Works with Visual Studio, GCC, Clang out of the box

### Using the C# Library

```csharp
using UMAPuwotSharp;

// Create and configure model
using var model = new UMapModel();

// Train on your data
var embedding = model.Fit(data, 
    nNeighbors: 15,      // Number of nearest neighbors
    minDist: 0.1f,       // Minimum distance in embedding
    nEpochs: 200);       // Training iterations

// Save for production use
model.Save("my_umap_model.bin");

// Later: Load and transform new data
using var loadedModel = UMapModel.Load("my_umap_model.bin");
var newEmbedding = loadedModel.Transform(newData);
```

## Performance and Compatibility

- **Based on uwot 0.2.3**: Uses the same optimized algorithms as the stable CRAN release
- **Cross-platform**: Windows and Linux support with automatic runtime detection
- **Memory efficient**: Careful resource management and cleanup
- **Production tested**: Comprehensive test suite validating all functionality
- **64-bit optimized**: Native libraries compiled for x64 architecture

## Technical Implementation

This implementation extracts the core C++ algorithms from the uwot R package (version 0.2.3), removes R-specific dependencies, and provides:

- **Standalone smooth k-NN**: Independent implementation without Rcpp dependencies
- **Binary model format**: Efficient serialization optimized for C# interop
- **Memory-safe C interface**: Proper error handling and resource management
- **Cross-platform build system**: CMake with Docker support for Linux compilation
- **CRAN-compatible**: Maintains algorithm parity with uwot 0.2.3 release

## Version Information

- **Native Libraries**: Based on uwot CRAN release 0.2.3
- **C# Wrapper**: Version 1.0.0 (UMAPuwotSharp)
- **Target Framework**: .NET 8.0
- **Supported Platforms**: Windows x64, Linux x64

## References

1. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
2. **Interactive UMAP Guide**: https://pair-code.github.io/understanding-umap/
3. **uwot R package**: https://github.com/jlmelville/uwot
4. **uwot CRAN 0.2.3**: https://cran.r-project.org/package=uwot
5. **Original Python UMAP**: https://github.com/lmcinnes/umap

## License

Maintains compatibility with the GPL-3 license of the original uwot package.

---

This implementation fills a critical gap in the .NET ecosystem by providing the first C# UMAP library with complete model persistence and transform capabilities, essential for production machine learning applications. The included prebuilt binaries based on uwot 0.2.3 ensure immediate usability and production stability.