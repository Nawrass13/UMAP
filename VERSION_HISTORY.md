# UMAPuwotSharp Version History

## Version 3.3.0 - HNSW Core Optimization (Current)

### 🚀 PERFORMANCE RELEASE - HNSW Acceleration
- **Enhanced HNSW optimization**: Refined k-NN acceleration for all supported metrics
- **Improved memory efficiency**: Further optimization of runtime memory usage
- **Enhanced progress reporting**: Better feedback during training with phase-aware callbacks
- **Cross-platform stability**: Improved build system and runtime compatibility

### 🎯 TECHNICAL IMPROVEMENTS
- **Better k-NN graph construction**: Optimized neighbor search algorithms
- **Enhanced distance metric support**: Improved performance for Euclidean, Cosine, and Manhattan
- **Refined memory management**: Reduced peak memory usage during training
- **Improved error handling**: Better diagnostic messages and recovery

---

## Version 3.1.2 - Spread Parameter Implementation

### 🆕 MAJOR FEATURE RELEASE - Spread Parameter
- **Complete spread parameter implementation**: Based on official UMAP algorithm
- **Smart dimension-based defaults**: 2D=5.0, 10D=2.0, 24D+=1.0 for optimal results
- **Mathematical curve fitting**: Proper a,b calculation from spread and min_dist
- **Enhanced API**: Nullable parameters with intelligent auto-optimization

### 📋 NEW API FEATURES
```csharp
// Smart defaults based on dimensions
var embedding2D = model.Fit(data, embeddingDimension: 2);  // Auto: spread=5.0

// Manual control for fine-tuning
var customEmbedding = model.Fit(data,
    embeddingDimension: 2,
    spread: 5.0f,          // Space-filling visualization
    minDist: 0.35f,        // Minimum point separation
    nNeighbors: 25);       // Optimal for 2D
```

---

## Version 3.1.0 - Revolutionary HNSW k-NN Optimization

### 🚀 BREAKTHROUGH PERFORMANCE
- **Complete HNSW k-NN optimization**: 50-2000x training speedup
- **Lightning-fast transforms**: <3ms per sample (vs 50-200ms before)
- **Massive memory reduction**: 80-85% less RAM usage (15-45MB vs 240MB)
- **Training optimization**: Hours → Minutes → Seconds for large datasets

### 📋 API CHANGES
```csharp
// New forceExactKnn parameter in Fit methods
var embedding = model.Fit(data,
    embeddingDimension: 2,
    forceExactKnn: false);  // Enable HNSW optimization
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

### From v2.x to v3.1.1
```csharp
// v2.x code (still works)
var embedding = model.Fit(data, embeddingDimension: 2);

// v3.1.1 optimized code with spread parameter
var embedding = model.Fit(data,
    embeddingDimension: 2,
    spread: 5.0f,          // NEW: t-SNE-like space-filling
    forceExactKnn: false); // Enable HNSW for 50-2000x speedup

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
| **3.1.0** | <3ms | 15-45MB | HNSW optimized | MSE < 0.01 |
| **3.3.0** | <3ms | 15-45MB | HNSW enhanced | MSE < 0.01 |

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