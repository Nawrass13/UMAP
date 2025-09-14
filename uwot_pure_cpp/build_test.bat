@echo off
echo Building HNSW Validation Test Suite...

REM Clean previous builds
if exist hnsw_validation_test.exe del hnsw_validation_test.exe

REM Build the test with OpenMP and HNSW support
g++ -std=c++17 -O3 -fopenmp ^
    -DUSE_SSE -DUSE_AVX -msse4.2 -mavx ^
    -I. ^
    hnsw_validation_test.cpp uwot_simple_wrapper.cpp ^
    -o hnsw_validation_test.exe

if exist hnsw_validation_test.exe (
    echo ✅ Build successful! Running validation tests...
    echo.
    hnsw_validation_test.exe
) else (
    echo ❌ Build failed!
    pause
)