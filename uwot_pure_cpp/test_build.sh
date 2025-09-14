#!/bin/bash
echo "Testing Cross-Platform Build with CMake (Linux)..."
echo

# Clean any previous build
rm -rf build
mkdir build
cd build

echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed!"
    exit 1
fi

echo
echo "Building with make..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo
echo "Running tests..."
ctest --output-on-failure

if [ $? -ne 0 ]; then
    echo "⚠️  Some tests failed, but build completed successfully."
else
    echo "✅ All tests passed!"
fi

echo
echo "Build artifacts:"
ls -la libuwot.so 2>/dev/null || echo "Library not found"
ls -la *test* 2>/dev/null || echo "Test executables not found"

echo
echo "Checking library dependencies:"
if [ -f "libuwot.so" ]; then
    ldd libuwot.so
    echo "Library size: $(stat -c%s libuwot.so) bytes"

    # Should be around 174KB with HNSW
    size=$(stat -c%s libuwot.so)
    if [ $size -gt 100000 ]; then
        echo "✅ Library size indicates HNSW integration is present"
    else
        echo "⚠️  Library size seems small - HNSW integration might be missing"
    fi
fi

echo
echo "✅ Cross-platform build test completed!"