# Enhanced UMAP C++ Wrapper with C# Integration

A complete, production-ready UMAP (Uniform Manifold Approximation and Projection) implementation based on the high-performance [uwot R package](https://github.com/jlmelville/uwot), providing both standalone C++ libraries and cross-platform C# integration with enhanced features.

## What is UMAP?

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that can be used for visualization, feature extraction, and preprocessing of high-dimensional data. Unlike many other dimensionality reduction algorithms, UMAP excels at preserving both local and global structure in the data.

**Key advantages of UMAP:**
- Preserves both local and global data structure
- Fast performance on large datasets
- Theoretically founded (topological data analysis)
- Excellent for visualization and preprocessing
- Supports out-of-sample projection (transform new data)

**Learn more:** [Understanding UMAP](https://pair-code.github.io/understanding-umap/)

## Why This Implementation?

**Problem with Existing C# Libraries:**
- Popular NuGet packages like `umap-sharp` **DO NOT** support:
  - ❌ Save/load trained models
  - ❌ Transform new data points (out-of-sample projection)
  - ❌ True UMAP algorithm implementation
  - ❌ Multiple distance metrics
  - ❌ Arbitrary embedding dimensions

**Our Solution Provides:**
- ✅ **Complete Model Persistence** - Save and load trained models
- ✅ **True Out-of-Sample Projection** - Transform new data using fitted models
- ✅ **Authentic UMAP Algorithm** - Based on proven uwot implementation
- ✅ **Multiple Distance Metrics** - Euclidean, Cosine, Manhattan, Correlation, Hamming
- ✅ **Arbitrary Dimensions** - 1D to 50D embeddings (including 27D)
- ✅ **Cross-Platform** - Windows, Linux, macOS support
- ✅ **Production Ready** - Memory-safe, thread-safe, comprehensive error handling

## Enhanced Features

### 🎯 Arbitrary Embedding Dimensions
Unlike other implementations limited to 2D/3D, this supports **1D to 50D embeddings**:
```csharp
// Create 27D embedding
var embedding = model.Fit(data, embeddingDimension: 27);

// Or any dimension you need
var embedding5D = model.Fit(data, embeddingDimension: 5);
var embedding1D = model.Fit(data, embeddingDimension: 1);
```

### 📏 Multiple Distance Metrics
Choose the right metric for your data:
```csharp
// For sparse, high-dimensional data
model.Fit(data, metric: DistanceMetric.Cosine);

// For outlier-robust analysis  
model.Fit(data, metric: DistanceMetric.Manhattan);

// For correlation-based relationships
model.Fit(data, metric: DistanceMetric.Correlation);

// For binary/categorical data
model.Fit(data, metric: DistanceMetric.Hamming);
```

### 💾 Complete Model Persistence
```csharp
// Train and save
using var model = new UMapModel();
var embedding = model.Fit(trainData, embeddingDimension: 27);
model.Save("my_model.umap");

// Load and use later
using var loadedModel = UMapModel.Load("my_model.umap");
var newEmbedding = loadedModel.Transform(newData);
```

### 🔄 True Out-of-Sample Projection
```csharp
// Transform new data points using the fitted model
var newEmbedding = model.Transform(newData);  // Same dimensionality as training
```

## Quick Start

### C# Usage
```csharp
using UMAPuwotSharp;

// Generate or load your data
float[,] data = GetYourData(); // [samples, features]

// Create and configure model
using var model = new UMapModel();

// Train with custom parameters
var embedding = model.Fit(
    data: data,
    embeddingDimension: 3,           // Target dimension
    nNeighbors: 15,                  // Number of neighbors
    minDist: 0.1f,                   // Minimum distance in embedding
    nEpochs: 300,                    // Training epochs
    metric: DistanceMetric.Euclidean // Distance metric
);

// embedding is now a [samples, 3] array

// Save the model
model.Save("my_model.umap");

// Transform new data
var newEmbedding = model.Transform(newData);

// Get model information
var info = model.ModelInfo;
Console.WriteLine(info); // Displays all parameters
```

### C++ Usage
```cpp
#include "uwot_simple_wrapper.h"

// Create model
UwotModel* model = uwot_create();

// Prepare data
float data[1000 * 50];  // 1000 samples, 50 features
// ... fill data ...

// Train model
float embedding[1000 * 27];  // 27D embedding
int result = uwot_fit(model, data, 1000, 50, 27, 15, 0.1f, 300, 
                      UWOT_METRIC_EUCLIDEAN, embedding);

if (result == UWOT_SUCCESS) {
    // Save model
    uwot_save_model(model, "model.umap");
    
    // Transform new data
    float new_data[100 * 50];  // 100 new samples
    float new_embedding[100 * 27];
    uwot_transform(model, new_data, 100, 50, new_embedding);
}

// Cleanup
uwot_destroy(model);
```

## Installation

### Prerequisites
- CMake 3.12+
- C++17 compatible compiler
- .NET 6.0+ (for C# wrapper)

### Build from Source
```bash
git clone https://github.com/yourusername/enhanced-umap-wrapper.git
cd enhanced-umap-wrapper

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)

# Install
sudo make install
```

### Windows Build
```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

## Project Structure
```
enhanced-umap-wrapper/
├── include/
│   └── uwot_simple_wrapper.h          # C API header
├── src/
│   └── uwot_enhanced_wrapper.cpp      # Main implementation
├── uwot_core_extracted/               # uwot library headers
│   ├── smooth_knn.h
│   ├── gradient.h
│   ├── optimize.h
│   └── ... (all uwot headers)
├── UMAPuwotSharp/                     # C# wrapper
│   ├── UMapModel.cs                   # Main C# interface
│   └── UMAPuwotSharp.csproj          # NuGet package project
├── tests/
│   ├── test_enhanced_wrapper.cpp      # C++ tests
│   └── UMAPTests.cs                   # C# tests
├── examples/
│   ├── cpp_example.cpp                # C++ examples
│   └── csharp_example.cs              # C# examples
└── CMakeLists.txt                     # Build configuration
```

## API Reference

### C# API

#### UMapModel Class
- `Fit(data, embeddingDimension, nNeighbors, minDist, nEpochs, metric)` - Train model
- `Transform(newData)` - Transform new data  
- `Save(filename)` - Save model to file
- `Load(filename)` - Load model from file (static)
- `ModelInfo` - Get model parameters
- `IsFitted` - Check if model is trained

#### DistanceMetric Enum
- `Euclidean` - Standard L2 distance
- `Cosine` - Good for sparse/high-dimensional data
- `Manhattan` - L1 distance, robust to outliers
- `Correlation` - Linear relationship based
- `Hamming` - For binary/categorical data

### C API

#### Core Functions
- `uwot_create()` - Create model instance
- `uwot_fit()` - Train model with enhanced parameters
- `uwot_transform()` - Transform new data
- `uwot_save_model()` / `uwot_load_model()` - Persistence
- `uwot_destroy()` - Clean up resources

#### Utility Functions
- `uwot_get_model_info()` - Get model parameters
- `uwot_get_error_message()` - Human-readable errors
- `uwot_get_metric_name()` - Distance metric names

## Performance

### Benchmarks (approximate)
- **Training**: ~2-5 seconds for 10K samples, 100 features → 2D on modern CPU
- **Transform**: ~50-100ms for 1K new samples  
- **Memory**: ~100MB for 10K samples with 27D embedding
- **Scaling**: Linear with sample count, sub-linear with dimensions

### Optimization Tips
- Use more neighbors (20-50) for higher-dimensional embeddings
- Increase epochs (400-600) for better convergence in high dimensions
- Choose appropriate distance metric for your data type
- Use batch processing for large transform operations

## Distance Metric Guide

| Metric | Best For | Notes |
|--------|----------|-------|
| **Euclidean** | General purpose, continuous data | Default choice, works well for most cases |
| **Cosine** | Sparse, high-dimensional data (text, images) | Focuses on direction, not magnitude |
| **Manhattan** | Outlier-robust analysis | Less sensitive to extreme values |
| **Correlation** | Time series, correlated features | Captures linear relationships |
| **Hamming** | Binary, categorical data | Counts differences in discrete features |

## Technical Details

### Based on uwot Library
This implementation uses the core algorithms from the [uwot R package](https://github.com/jlmelville/uwot):
- Same gradient calculations and optimization
- Identical smooth k-NN weight computation  
- Proven negative sampling strategy
- Consistent results with R implementation

### Enhancements Added
- Multiple distance metrics for k-NN graph construction
- Arbitrary embedding dimensions (1-50D)
- Complete model serialization/deserialization
- Cross-platform C# wrapper with proper memory management
- Enhanced error handling and validation

### Memory Management
- RAII patterns in C++
- Automatic resource cleanup in C#
- No memory leaks in long-running applications
- Efficient memory usage for large datasets

## Examples and Use Cases

### 27D Time Series Embedding
```csharp
// Multi-variate time series with 9 sensors over 3 time windows
float[,] timeSeriesData = LoadSensorData(); // [samples, 27 features]

using var model = new UMapModel();
var embedding = model.Fit(
    timeSeriesData,
    embeddingDimension: 5,  // Reduce to 5D for downstream ML
    metric: DistanceMetric.Correlation
);

// Save for later use
model.Save("timeseries_model.umap");
```

### High-Dimensional Sparse Data
```csharp
// Text embeddings, sparse features
float[,] textFeatures = LoadTextData(); 

var embedding = model.Fit(
    textFeatures,
    embeddingDimension: 10,
    metric: DistanceMetric.Cosine,  // Good for sparse data
    nNeighbors: 30                   // More neighbors for sparse data
);
```

### Batch Processing Pipeline
```csharp
// Train once
var model = UMapModel.Load("production_model.umap");

// Process data in batches
foreach (var batch in GetDataBatches()) {
    var batchEmbedding = model.Transform(batch);
    ProcessBatchResults(batchEmbedding);
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
1. Clone the repository
2. Install dependencies
3. Build in debug mode: `cmake -DCMAKE_BUILD_TYPE=Debug ..`
4. Run tests: `make test`

## License

This project is licensed under the BSD 2-Clause License - see [LICENSE](LICENSE) file for details.

The uwot library components are also licensed under BSD 2-Clause License.
Copyright 2020 James Melville.

## Acknowledgments

- [James Melville](https://github.com/jlmelville) for the excellent uwot R package
- [Leland McInnes](https://github.com/lmcinnes) for the original UMAP algorithm
- The UMAP paper: [McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018](https://arxiv.org/abs/1802.03426)

## Support

- 📖 Documentation: [Link to docs]
- 🐛 Issues: [GitHub Issues]
- 💬 Discussions: [GitHub Discussions]
- 📧 Contact: [Your email]

---

**Ready for production use with your 27D embeddings and any distance metric you need!**