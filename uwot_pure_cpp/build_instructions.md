# Cross-Platform Build Instructions

## Overview
The UMAP C++ library with HNSW optimization supports building on both Windows and Linux using CMake. This document provides comprehensive build instructions for development and production deployments.

## Prerequisites

### Windows
- **Visual Studio 2022** (with C++ workload)
- **CMake 3.12+**: Download from https://cmake.org/download/
- **Git**: For repository cloning
- **Optional**: MinGW-w64 for GCC builds

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install build-essential cmake git
sudo apt install libomp-dev  # OpenMP support
```

### Linux (CentOS/RHEL/Fedora)
```bash
sudo yum install gcc-c++ cmake git make
sudo yum install libomp-devel  # OpenMP support
```

## Build Process

### Standard Development Build

#### Windows (Visual Studio)
```bash
cd uwot_pure_cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTS=ON
cmake --build . --config Release
ctest -C Release
```

#### Windows (MinGW)
```bash
cd uwot_pure_cpp
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build .
ctest
```

#### Linux
```bash
cd uwot_pure_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
make -j$(nproc)
ctest
```

### Production Build (Optimized)

#### Windows Production
```bash
mkdir build-release && cd build-release
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_SHARED_LIBS=ON ^
    -DBUILD_TESTS=OFF
cmake --build . --config Release
```

#### Linux Production
```bash
mkdir build-release && cd build-release
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -DUSE_AVX -DUSE_SSE" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTS=OFF
make -j$(nproc)
```

## Build Options

### CMake Configuration Options
- **`BUILD_TESTS=ON/OFF`**: Enable/disable test compilation (default: ON)
- **`BUILD_SHARED_LIBS=ON/OFF`**: Build shared/static library (default: ON)
- **`CMAKE_BUILD_TYPE`**: Debug, Release, RelWithDebInfo, MinSizeRel
- **`OpenMP_ROOT`**: Custom OpenMP installation path

### Compiler Optimizations
```bash
# Maximum optimization (Linux)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DUSE_AVX -DUSE_SSE"

# Windows MSVC optimizations
cmake .. -DCMAKE_CXX_FLAGS="/O2 /arch:AVX2 /DUSE_AVX /DUSE_SSE"
```

## Testing

### Run All Tests
```bash
# After building with BUILD_TESTS=ON
ctest                                    # Linux/MinGW
ctest -C Release                        # Visual Studio
```

### Specific Test Execution
```bash
# Run C++ validation tests
./hnsw_validation_test                  # Linux/MinGW
Release\hnsw_validation_test.exe        # Windows

# Run enhanced wrapper tests
./test_enhanced_wrapper                 # Linux/MinGW
Release\test_enhanced_wrapper.exe       # Windows
```

### Verbose Test Output
```bash
ctest --verbose
# or
ctest --output-on-failure
```

## Cross-Platform NuGet Build

### Automated Cross-Platform Build
For official NuGet package releases, use the Docker-based build:

```bash
cd uwot_pure_cpp
BuildDockerLinuxWindows.bat            # Builds both platforms
```

This script:
1. **Windows**: Builds using Visual Studio tools with HNSW optimization
2. **Linux**: Uses Docker container with GCC and proper HNSW integration
3. **Output**: Places binaries in correct NuGet runtime structure

### Manual Cross-Platform Verification

#### 1. Windows Build Verification
```bash
# Check Windows DLL size and dependencies
dir build\Release\uwot.dll              # Should be ~150KB
dumpbin /dependents build\Release\uwot.dll
```

#### 2. Linux Build Verification
```bash
# Check Linux SO size and dependencies
ls -la build/libuwot.so                 # Should be ~174KB
ldd build/libuwot.so
nm -D build/libuwot.so | grep uwot      # Check exported symbols
```

## Troubleshooting

### Common Build Issues

#### 1. CMake Configuration Errors
```bash
# Clear CMake cache
rm -rf build/
mkdir build && cd build

# Regenerate with verbose output
cmake .. --debug-output
```

#### 2. Missing OpenMP
```bash
# Linux: Install OpenMP
sudo apt install libomp-dev

# Windows: Ensure Visual Studio C++ tools include OpenMP
# Or use MinGW: pacman -S mingw-w64-x86_64-openmp
```

#### 3. HNSW Header Issues
```bash
# Verify all 7 HNSW headers are present
ls -la *.h | grep -E "(hnsw|space_|bruteforce|visited)"

# Expected files:
# bruteforce.h, hnswalg.h, hnswlib.h, space_ip.h,
# space_l2.h, stop_condition.h, visited_list_pool.h
```

#### 4. C++17 Compatibility
```bash
# Ensure C++17 support
cmake .. -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
```

### Performance Validation

#### Quick Performance Check
```bash
# Run the comprehensive validation test
./hnsw_validation_test

# Expected output:
# ✅ TEST 1 PASSED: HNSW vs Exact Accuracy (MSE < 0.01)
# ✅ TEST 2 PASSED: Multi-Metric Support
# ✅ TEST 3 PASSED: Memory Usage and Persistence
# 🎉 ALL TESTS PASSED! HNSW optimization ready for deployment.
```

#### Memory and Speed Validation
The test suite automatically validates:
- **Speedup**: ≥2x improvement for datasets >2000 samples
- **Accuracy**: MSE between HNSW and exact <0.01
- **Memory**: Proper index size and persistence

## Integration with C# Project

### Copy Built Libraries
After successful build, copy libraries to C# runtime folders:

```bash
# Windows
cp build/Release/uwot.dll ../UMAPuwotSharp/UMAPuwotSharp/runtimes/win-x64/native/

# Linux
cp build/libuwot.so ../UMAPuwotSharp/UMAPuwotSharp/runtimes/linux-x64/native/
```

### Test C# Integration
```bash
cd ../UMAPuwotSharp
dotnet build
dotnet test                             # Should pass all C# tests
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Build C++ Library
  run: |
    cd uwot_pure_cpp
    mkdir build && cd build
    cmake .. -DBUILD_TESTS=ON
    cmake --build . --config Release
    ctest --output-on-failure
```

### Docker Build Example
```dockerfile
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y build-essential cmake libomp-dev
COPY uwot_pure_cpp /src
WORKDIR /src
RUN mkdir build && cd build && cmake .. && make -j$(nproc) && ctest
```

This comprehensive build system ensures the HNSW optimization works consistently across all target platforms!