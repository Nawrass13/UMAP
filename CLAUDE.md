# Enhanced UMAP C++ Implementation with C# Wrapper - Claude Code Guide

## Project Overview
High-performance UMAP implementation with enhanced features not available in other C# libraries:
- **Multi-dimensional embeddings**: 1D to 50D (including specialized 27D)
- **Multiple distance metrics**: Euclidean, Cosine, Manhattan, Correlation, Hamming
- **Model persistence**: Save/load trained models
- **Progress reporting**: Real-time training feedback
- **Transform capability**: Project new data using existing models

## 🚀 CRITICAL OPTIMIZATION: HNSW Integration
**Problem**: Current implementation has scalability crisis:
- 240MB memory usage (stores all training data)
- 50-200ms transform times (linear search)
- Production deployment bottleneck

**Solution**: HNSW (Hierarchical Navigable Small World) indexing integration
- **Memory reduction**: 80-85% savings (15-45MB vs 240MB)
- **Speed improvement**: 50-2000x faster transforms (<1ms vs 50-200ms)
- **Enhanced safety**: Multi-level out-of-distribution detection

## Project Structure
```
UMAP/
├── UMAPuwotSharp/                 # C# wrapper library
│   ├── UMAPuwotSharp/             # Main library project (needs TransformResult class)
│   ├── UMAPuwotSharp.Example/     # Demo application
│   └── UMapSharp.sln              # Visual Studio solution
├── uwot_pure_cpp/                 # Native C++ implementation
│   ├── CMakeLists.txt             # Cross-platform build
│   ├── uwot_simple_wrapper.cpp/.h # Main C++ wrapper (needs HNSW integration)
│   ├── *.h                        # HNSW library headers (7 files from nmslib/hnswlib)
│   ├── build/                     # Windows build outputs
│   └── build-linux/               # Linux build outputs
└── Other/                         # Documentation assets
```

## HNSW Integration Requirements

### Input/Output Specification
**Training Input**:
- Variable sample count × variable feature dimensions (e.g., 200k × 300)
- Raw MyTickDataValue objects from C#
- C++ handles normalization calculation and HNSW index building

**Transform Input/Output**:
- Input: Single MyTickDataValue (matching training dimensions)
- Output: Enhanced TransformResult with safety metrics

```csharp
public class TransformResult {
    public float[] ProjectionCoordinates;      // 1-50D embedding position
    public int[] NearestNeighborIndices;       // Closest training point indices
    public float[] NearestNeighborDistances;   // Distances in original space
    public float ConfidenceScore;              // 0.0-1.0 safety confidence
    public OutlierLevel Severity;              // Normal/Unusual/Mild/Extreme/NoMansLand
    public float PercentileRank;               // 0-100% distance ranking
    public float ZScore;                       // Standard deviations from mean
}
```

## Build Commands

### C# Library (Main Development)
```bash
cd UMAPuwotSharp
dotnet build                       # Build library and example
dotnet run --project UMAPuwotSharp.Example  # Run demo
```

### C++ Native Library (Advanced)
```bash
cd uwot_pure_cpp
BuildDockerLinuxWindows.bat        # Cross-platform build
```

## Current Status
- ✅ Core UMAP functionality working perfectly
- ✅ All enhanced features operational (1D-50D, multi-metrics, progress reporting)
- ✅ Cross-platform binaries included (Windows/Linux)
- ✅ Demo application runs successfully
- ✅ NuGet package (v2.0.0) builds successfully
- ✅ HNSW library headers downloaded (7 files from nmslib/hnswlib)
- ✅ **HNSW INTEGRATION COMPLETED**: Direct filestream operations, no temp file management issues
- ✅ **ENHANCED API COMPLETED**: TransformResult class, OutlierLevel enum, safety metrics
- ✅ **C# INTEGRATION READY**: P/Invoke declarations, enhanced examples, runtime binaries
- ✅ **BUILD SYSTEM ENHANCED**: Automated binary copying to runtime folders
- ✅ **COMPREHENSIVE TESTING**: C++ test suite with HNSW validation, performance testing
- ✅ **Clean compilation**: All nullability warnings fixed, zero build errors

## Known Issues
- ✅ ~~`CS8625` warning in `UMAPuwotSharp/UMAPuwotSharp/UMapModel.cs:247`~~ - **FIXED**: Proper nullable parameter handling
- ✅ ~~`CS8600` warning in `UMAPuwotSharp/UMAPuwotSharp.Example/Program.cs:114`~~ - **FIXED**: Nullable progress callback

## Next Steps

### 🎯 PRIORITY 0: HNSW OPTIMIZATION (CRITICAL)
**Implementation Checklist**:
- [ ] **C++ Structure Updates**
  - [ ] Update UwotModel with HNSW index (`std::unique_ptr<hnswlib::HierarchicalNSW<float>>`)
  - [ ] Add normalization vectors (`feature_means`, `feature_stds`)
  - [ ] Add safety statistics (distance thresholds, percentiles)
  - [ ] Include HNSW headers in uwot_simple_wrapper.cpp
- [ ] **C++ Training Function**
  - [ ] Move normalization calculation from C# to C++
  - [ ] Build HNSW index during training
  - [ ] Compute neighbor distance statistics for outlier detection
  - [ ] Update save/load to exclude training data, include statistics
- [ ] **C++ Transform Function**
  - [ ] Implement internal normalization
  - [ ] HNSW approximate nearest neighbor search
  - [ ] Statistical safety analysis (confidence, outlier level, percentile rank)
  - [ ] Return comprehensive TransformResult data
- [ ] **C# API Updates**
  - [ ] Create TransformResult class with OutlierLevel enum
  - [ ] Remove normalization logic from UMapManager.cs
  - [ ] Update P/Invoke declarations for new C++ functions
  - [ ] Update example code to use enhanced safety features
- [ ] **Build & Test**
  - [ ] Update CMakeLists.txt with HNSW integration
  - [ ] Verify Windows 64-bit and Linux compatibility
  - [ ] Update C++ test file for HNSW validation
  - [ ] Performance testing: memory usage and transform speed

**Expected Results**:
- Memory: 240MB → 15-45MB (80-85% reduction)
- Transform speed: 50-200ms → <1ms (50-2000x improvement)
- Enhanced safety: Multi-level outlier detection

### Priority 1: Code Quality (After HNSW)
- [ ] Fix nullability warnings in UMapModel.cs and Program.cs
- [ ] Add comprehensive error handling validation
- [ ] Review memory management for high-dimensional embeddings

### Priority 2: Testing & Validation
- [ ] Add unit tests for multi-dimensional embeddings (especially 27D)
- [ ] Create performance benchmarks for different dimensions/metrics
- [ ] Test memory usage patterns with large datasets
- [ ] Validate model persistence across different scenarios
- [ ] **HNSW-specific testing**:
  - [ ] Validate transform accuracy vs linear search
  - [ ] Test outlier detection reliability
  - [ ] Benchmark memory usage across dataset sizes

### Priority 3: Documentation & Examples
- [ ] Add more usage examples for different distance metrics
- [ ] Document best practices for choosing embedding dimensions
- [ ] Create performance optimization guide
- [ ] Add troubleshooting section for common issues
- [ ] **HNSW documentation**:
  - [ ] Document TransformResult safety features
  - [ ] Performance characteristics guide
  - [ ] Outlier detection interpretation guide

### Future Enhancements
- [ ] GPU acceleration support investigation
- [ ] Additional distance metrics (if needed)
- [ ] Streaming/incremental learning capabilities
- [ ] Python bindings for broader ecosystem support
- [ ] Web assembly port for browser usage

## Key Features Demo Commands
```bash
# Run full enhanced demo
cd UMAPuwotSharp
dotnet run --project UMAPuwotSharp.Example

# Test specific dimensions
# (modify Program.cs embeddingDimension parameter)

# Build NuGet package
dotnet pack UMAPuwotSharp/UMAPuwotSharp.csproj
```

## Development Notes
- Project uses .NET 8.0 target framework
- Native libraries are pre-built and included in runtimes/ folders
- Cross-platform compatibility handled automatically at runtime
- OpenMP support enabled for parallel processing
- Based on proven uwot R package algorithms with C# enhancements

## Performance Characteristics

### Current Implementation
- **Memory usage**: ~4-8GB RAM for 100k samples × 300 features (depending on n_neighbors)
- **Training time**: Scales with embedding dimension and epoch count
- **Transform time**: 50-200ms (linear search through training data)

### After HNSW Optimization (Target)
- **Memory usage**: 80-85% reduction (15-45MB for 200k × 300 dataset)
- **Transform time**: <1ms (50-2000x improvement via HNSW indexing)
- **Training time**: Minimal increase (HNSW index construction)
- **Additional features**: Out-of-distribution detection, safety metrics

### Best Distance Metrics by Data Type
- **Euclidean**: General-purpose numeric data
- **Cosine**: High-dimensional sparse data (text, images)
- **Manhattan**: Outlier-robust applications
- **Correlation**: Time series or correlated features
- **Hamming**: Binary/categorical data

### HNSW Configuration
- **Build environment**: Windows 64-bit and Linux
- **Index type**: Hierarchical Navigable Small World (approximate NN)
- **Input dimensions**: Variable (e.g., 300 features, configurable)
- **Output dimensions**: 1-50D embeddings (maximum 50D constraint)
- **Safety features**: Multi-level outlier detection using training statistics

## Common Workflows
1. **Standard 2D visualization**: `embeddingDimension: 2, metric: Euclidean`
2. **Feature extraction**: `embeddingDimension: 27, metric: Cosine`
3. **Production ML pipeline**: Save model → Transform new data
4. **Research/experimentation**: Multi-metric comparison with progress monitoring

## HNSW Implementation Files Checklist

### Required C++ Files
- [x] **HNSW Headers**: 7 files from https://github.com/nmslib/hnswlib
  - [x] `bruteforce.h`, `hnswalg.h`, `hnswlib.h`
  - [x] `space_ip.h`, `space_l2.h`, `stop_condition.h`, `visited_list_pool.h`
- [x] **Main wrapper**: `uwot_simple_wrapper.cpp/.h` ✅ **HNSW integration completed with enhanced stream operations**
- [x] **uwot headers**: `smooth_knn.h`, `transform.h`, `gradient.h` (existing)

### Required C# Files
- [ ] **TransformResult class**: Enhanced result object with safety metrics
- [ ] **UMapManager.cs**: Remove normalization, add TransformResult support
- [ ] **OutlierLevel enum**: Normal/Unusual/Mild/Extreme/NoMansLand severity levels
- [ ] **P/Invoke updates**: New C++ function declarations
- [ ] **Example updates**: Demonstrate safety features

### Build Configuration
- [ ] **CMakeLists.txt**: Add HNSW integration flags
- [ ] **Batch files**: Update build scripts for HNSW
- [ ] **C++ test file**: Add HNSW validation tests

## Implementation Phases
1. **Phase 1**: C++ structure updates and HNSW integration
2. **Phase 2**: C++ training/transform function implementation
3. **Phase 3**: C# API updates and TransformResult class
4. **Phase 4**: Build system updates and cross-platform testing
5. **Phase 5**: Performance validation and documentation