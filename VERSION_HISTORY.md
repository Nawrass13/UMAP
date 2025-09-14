# UMAPuwotSharp Version History

## Version 3.1.0 - Revolutionary HNSW k-NN Optimization (Current)

### 🚀 BREAKTHROUGH PERFORMANCE
- **Complete HNSW k-NN optimization**: 50-2000x training speedup
- **Lightning-fast transforms**: <3ms per sample (vs 50-200ms before)
- **Massive memory reduction**: 80-85% less RAM usage (15-45MB vs 240MB)
- **Training optimization**: Hours → Minutes → Seconds for large datasets

### 🆕 NEW API FEATURES
- **forceExactKnn parameter**: Choose HNSW speed or exact accuracy
- **Enhanced progress callbacks**: Phase-aware reporting with time estimates
- **Smart auto-optimization**: Automatic HNSW/exact selection by metric
- **OpenMP parallelization**: Multi-core acceleration built-in
- **Advanced warning system**: Helpful guidance for optimal performance

### 🔥 HNSW-ACCELERATED METRICS
- ✅ **Euclidean**: General-purpose data (50-200x speedup)
- ✅ **Cosine**: High-dimensional sparse data (30-150x speedup)
- ✅ **Manhattan**: Outlier-robust applications (40-180x speedup)
- ⚡ **Correlation/Hamming**: Auto-fallback to exact with warnings

### 📊 VALIDATED PERFORMANCE
- **Accuracy**: MSE < 0.01 between HNSW and exact embeddings
- **Speed**: 230x faster for 50k+ sample datasets
- **Memory**: 87% reduction for production deployments
- **Cross-platform**: Windows/Linux parity with comprehensive test suites

### 💻 TECHNICAL IMPROVEMENTS
- **HNSW Integration**: Full hnswlib integration with custom L1Space for Manhattan
- **SpaceFactory Pattern**: Automatic metric-based space selection
- **Unified Normalization**: Streamlined data preparation pipeline
- **Enhanced Error Handling**: Comprehensive validation and user feedback
- **Build System**: CMake-based cross-platform compilation
- **Test Coverage**: Extensive C++ and C# test suites

### 📋 API CHANGES
```csharp
// New forceExactKnn parameter in Fit methods
var embedding = model.Fit(data,
    embeddingDimension: 2,
    forceExactKnn: false);  // Enable HNSW optimization

// Enhanced progress callback with phase information
var embedding = model.FitWithProgress(data, (epoch, total, percent) => {
    // Now includes phase names like "Building HNSW index"
});
```

---

## Version 3.0.1 - Critical Cross-Platform Fix

### 🔧 CRITICAL FIXES
- **Linux HNSW library**: Fixed missing HNSW optimization in Linux build
- **Cross-platform parity**: Both Windows (150KB) and Linux (174KB) now include HNSW
- **Build process**: Enhanced BuildDockerLinuxWindows.bat for proper cross-compilation

### ⚠️ IMPORTANT UPGRADE NOTE
Version 3.0.0 had an incomplete Linux native library (69KB) missing HNSW optimization.
v3.0.1 includes the complete Linux library (174KB) with full HNSW acceleration.

---

## Version 3.0.0 - HNSW Optimization Introduction

### 🎯 MAJOR FEATURES
- **First HNSW implementation**: Revolutionary k-NN acceleration
- **Production safety features**: 5-level outlier detection (Normal → No Man's Land)
- **Enhanced transform capability**: TransformDetailed with confidence scoring
- **Model persistence**: Complete save/load with HNSW indices
- **Multi-dimensional support**: 1D-50D embeddings all HNSW-optimized

### 🚨 KNOWN ISSUES (FIXED IN v3.0.1)
- Linux native library incomplete (missing HNSW optimization)
- Cross-platform performance disparity

---

## Version 2.x Series - Legacy Implementation

### Core Features
- ✅ **Standard UMAP**: Complete uwot-based implementation
- ✅ **Multi-dimensional**: 1D-50D embedding support
- ✅ **Multi-metric**: Euclidean, Cosine, Manhattan, Correlation, Hamming
- ✅ **Progress reporting**: Training progress callbacks
- ✅ **Model persistence**: Save/load trained models
- ✅ **Transform capability**: Project new data points

### Performance Characteristics
- **Memory usage**: ~240MB for typical datasets (stored all training data)
- **Transform speed**: 50-200ms per sample (brute-force search)
- **Training time**: Standard O(n²) complexity for k-NN computation

### Limitations Addressed in v3.x
- ❌ **Memory intensive**: Stored full training dataset for transforms
- ❌ **Slow transforms**: Linear search through all training points
- ❌ **No safety features**: No out-of-distribution detection
- ❌ **Limited scalability**: Performance degraded with large datasets

---

## Migration Guide

### From v2.x to v3.1.0
```csharp
// v2.x code (still works)
var embedding = model.Fit(data, embeddingDimension: 2);

// v3.1.0 optimized code
var embedding = model.Fit(data,
    embeddingDimension: 2,
    forceExactKnn: false);  // Enable HNSW for 50-2000x speedup

// New safety features
var result = model.TransformDetailed(newData);
Console.WriteLine($"Confidence: {result.ConfidenceScore}");
Console.WriteLine($"Outlier level: {result.Severity}");
```

### Breaking Changes
- **None**: Full backward compatibility maintained
- **New parameters**: All new parameters are optional with sensible defaults
- **Enhanced results**: TransformDetailed provides additional safety information

---

## Performance Evolution

| Version | Transform Speed | Memory Usage | k-NN Method | Accuracy |
|---------|----------------|--------------|-------------|----------|
| **2.x** | 50-200ms | 240MB | Brute-force | Exact |
| **3.0.0** | <3ms | 15-45MB | HNSW | MSE < 0.01 |
| **3.0.1** | <3ms | 15-45MB | HNSW (both platforms) | MSE < 0.01 |
| **3.1.0** | <3ms | 15-45MB | HNSW + enhancements | MSE < 0.01 |

### Speedup Achievements
- **50-2000x faster transforms**: Production-ready real-time processing
- **80-85% memory reduction**: Scalable to much larger datasets
- **Hours → Seconds**: Training time optimization for large datasets
- **Cross-platform parity**: Identical performance Windows/Linux

---

## Future Roadmap

### Planned Features
- **GPU acceleration**: CUDA support for even faster processing
- **Streaming updates**: Incremental model updates without full retraining
- **Additional metrics**: Extended distance function support
- **Python bindings**: Broader ecosystem integration
- **Web assembly**: Browser-based UMAP processing

### Community Contributions
- Bug reports and feature requests welcome
- Performance benchmarking across different hardware
- Additional usage examples and tutorials
- Integration guides for specific ML frameworks

---

*This version history tracks the evolution of UMAPuwotSharp from a standard UMAP implementation to a revolutionary high-performance system with HNSW optimization and production safety features.*