# HNSW Validation Test Suite

## Overview
Comprehensive test suite for validating the HNSW k-NN optimization implementation.

## Test Components

### 1. hnsw_validation_test.cpp
**Purpose**: Complete validation of HNSW optimization features

**Tests Included**:
- **Test 1**: HNSW vs Exact k-NN Accuracy Validation
  - Compares HNSW approximate with exact brute-force results
  - **Target**: MSE < 0.01 between embeddings
  - **Performance**: Validates speedup > 1.0x (typically 10-100x)

- **Test 2**: Multi-Metric Support Validation
  - Tests all supported metrics: Euclidean, Cosine, Manhattan
  - Tests unsupported metrics fallback: Correlation, Hamming
  - Validates proper space selection and preparation

- **Test 3**: Memory Usage and Model Persistence
  - Tests model save/load functionality with HNSW indices
  - Validates transform operations work correctly
  - Tests enhanced transform with safety analysis

## Build Instructions

### Using CMake (Recommended)
```bash
cd uwot_pure_cpp
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make
ctest
```

### Using Visual Studio (Windows)
```bash
cd uwot_pure_cpp
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -DBUILD_TESTS=ON
cmake --build . --config Release
ctest -C Release
```

### Manual Build (if needed)
```bash
cd uwot_pure_cpp
g++ -std=c++17 -O3 -fopenmp -I. hnsw_validation_test.cpp uwot_simple_wrapper.cpp -o hnsw_validation_test
./hnsw_validation_test
```

## Expected Results

### Success Criteria
✅ **Test 1**: MSE < 0.01, Speedup > 1.0x
✅ **Test 2**: All metrics complete successfully
✅ **Test 3**: Model persistence and transform work correctly

### Performance Benchmarks
- **Approximate k-NN**: 10-100x faster than exact for 1000+ samples
- **Memory efficiency**: ~80-85% reduction in memory usage
- **Accuracy**: MSE typically < 0.005 for structured data

## Troubleshooting

### Common Issues
1. **Build errors**: Ensure C++17 support and OpenMP availability
2. **HNSW header issues**: Verify all 7 HNSW headers are present
3. **Memory errors**: Check for proper model destruction in tests

### Debug Mode
Add `-DCMAKE_BUILD_TYPE=Debug` for detailed debugging information.

## Integration with CI/CD
This test suite can be integrated into automated testing pipelines:
- **GitHub Actions**: Use `ctest` command in CI
- **Azure DevOps**: Compatible with CMake build tasks
- **Jenkins**: Standard CMake/CTest integration