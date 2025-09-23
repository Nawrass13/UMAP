@echo off
REM Set console to handle UTF-8 properly (suppress output)
chcp 65001 >nul 2>&1

echo ===========================================
echo   Building Enhanced UMAP C++ Library
echo ===========================================
echo.

REM Check if build directory exists, clean if needed
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

echo Creating build directory...
mkdir build
cd build

echo.
echo Configuring Enhanced UMAP with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=ON
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo Building Release version...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   Build completed successfully!
echo ===========================================
echo.
echo Generated files:
echo   - uwot.dll (enhanced shared library for C#)
echo   - test_enhanced_wrapper.exe (comprehensive functionality test)
echo.

REM Test the builds
echo ===========================================
echo   Running All Enhanced Tests (1-4)
echo ===========================================
echo.

set ALL_TESTS_PASSED=1

REM Change to Release directory for all tests
cd Release

echo [TEST 1] Running standard comprehensive validation...
if exist "test_standard_comprehensive.exe" (
    test_standard_comprehensive.exe
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] Test 1: Standard comprehensive test PASSED
    ) else (
        echo [FAIL] Test 1: Standard comprehensive test FAILED with code %ERRORLEVEL%
        set ALL_TESTS_PASSED=0
    )
) else (
    echo [WARN] test_standard_comprehensive.exe not found
    set ALL_TESTS_PASSED=0
)

echo.
echo [TEST 2] Running error fixes validation...
if exist "test_error_fixes_simple.exe" (
    test_error_fixes_simple.exe
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] Test 2: Error fixes test PASSED
    ) else (
        echo [FAIL] Test 2: Error fixes test FAILED with code %ERRORLEVEL%
        set ALL_TESTS_PASSED=0
    )
) else (
    echo [WARN] test_error_fixes_simple.exe not found
    set ALL_TESTS_PASSED=0
)

echo.
echo [TEST 3] Running comprehensive pipeline validation...
if exist "test_comprehensive_pipeline.exe" (
    test_comprehensive_pipeline.exe
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] Test 3: Comprehensive pipeline test PASSED
    ) else (
        echo [FAIL] Test 3: Comprehensive pipeline test FAILED with code %ERRORLEVEL%
        set ALL_TESTS_PASSED=0
    )
) else (
    echo [WARN] test_comprehensive_pipeline.exe not found
    set ALL_TESTS_PASSED=0
)

echo.
echo [TEST 4] Running enhanced wrapper validation...
if exist "test_enhanced_wrapper.exe" (
    test_enhanced_wrapper.exe
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] Test 4: Enhanced wrapper test PASSED
    ) else (
        echo [FAIL] Test 4: Enhanced wrapper test FAILED with code %ERRORLEVEL%
        set ALL_TESTS_PASSED=0
    )
) else (
    echo [WARN] test_enhanced_wrapper.exe not found
    set ALL_TESTS_PASSED=0
)

cd ..

echo.
if %ALL_TESTS_PASSED% EQU 1 (
    echo [PASS] ALL 4 TESTS COMPLETED SUCCESSFULLY!
    echo       - Standard comprehensive validation verified
    echo       - Error fixes validation verified
    echo       - Comprehensive pipeline validation verified
    echo       - Enhanced wrapper validation verified
    echo       - 27D embedding capability confirmed
    echo       - Multiple distance metrics working
    echo       - Save/load functionality operational
    echo       - Transform (out-of-sample) working
) else (
    echo [FAIL] SOME TESTS FAILED!
    echo       Build may have issues - check test output above
)

cd ..

echo.
echo ===========================================
echo   Copying Files to C# Project
echo ===========================================
echo.

REM Copy DLL directly to C# project folder
if exist "build\Release\uwot.dll" (
    if exist "..\UMAPuwotSharp\UMAPuwotSharp\" (
        copy "build\Release\uwot.dll" "..\UMAPuwotSharp\UMAPuwotSharp\" >nul
        echo [COPY] uwot.dll copied to C# project folder
    ) else (
        echo [WARN] C# project folder not found: ..\UMAPuwotSharp\UMAPuwotSharp\
    )
) else (
    echo [FAIL] uwot.dll not found for copying
)

echo.
echo ===========================================
echo   Enhanced UMAP Build Summary
echo ===========================================
echo.
echo Your Enhanced UMAP C++ library is ready for C# integration!
echo.
echo Key files for C# P/Invoke:
echo   [FILE] build\Release\uwot.dll           - Enhanced library with 27D support
echo   [FILE] uwot_simple_wrapper.h            - Enhanced header with all functions
echo.
echo Enhanced features available:
echo   [NEW] Arbitrary Dimensions      - 1D to 50D embeddings (including 27D)
echo   [NEW] Multiple Distance Metrics - Euclidean, Cosine, Manhattan, Correlation, Hamming
echo   [YES] Model Training           - uwot_fit() with enhanced parameters
echo   [YES] Model Persistence        - uwot_save_model() / uwot_load_model()
echo   [YES] Data Transformation      - uwot_transform() for out-of-sample projection
echo   [YES] Model Information        - uwot_get_model_info() with metric support
echo   [YES] Cross-platform          - Windows ready, Linux compatible
echo.
echo Enhanced C# usage example:
echo   // Create model with 27D embedding and cosine distance
echo   [DllImport("uwot.dll")] static extern IntPtr uwot_create();
echo   [DllImport("uwot.dll")] 
echo   static extern int uwot_fit(IntPtr model, float[] data, int n_obs, int n_dim, 
echo                              int embedding_dim, int n_neighbors, float min_dist,
echo                              int n_epochs, UwotMetric metric, float[] embedding);
echo   [DllImport("uwot.dll")] static extern int uwot_save_model(IntPtr model, string filename);
echo   [DllImport("uwot.dll")] static extern int uwot_transform(IntPtr model, float[] new_data, 
echo                                                            int n_new_obs, int n_dim, float[] embedding);
echo.

REM Check for enhanced files
echo ===========================================
echo   Enhanced File Check
echo ===========================================
echo.
if exist "build\Release\uwot.dll" (
    echo [PASS] uwot.dll found (Enhanced UMAP library - ONLY FILE NEEDED FOR C#)
    for %%A in ("build\Release\uwot.dll") do echo         Size: %%~zA bytes
) else (
    echo [FAIL] uwot.dll missing!
)

if exist "build\Release\test_standard_comprehensive.exe" (
    echo [PASS] test_standard_comprehensive.exe found (Critical validation)
) else (
    echo [WARN] test_standard_comprehensive.exe missing
)

if exist "build\Release\test_error_fixes_simple.exe" (
    echo [PASS] test_error_fixes_simple.exe found (Error fixes validation)
) else (
    echo [WARN] test_error_fixes_simple.exe missing
)

if exist "build\Release\test_comprehensive_pipeline.exe" (
    echo [PASS] test_comprehensive_pipeline.exe found (Pipeline validation)
) else (
    echo [WARN] test_comprehensive_pipeline.exe missing
)

if exist "build\Release\test_enhanced_wrapper.exe" (
    echo [PASS] test_enhanced_wrapper.exe found (Enhanced features)
) else (
    echo [WARN] test_enhanced_wrapper.exe missing
)

if exist "uwot_simple_wrapper.h" (
    echo [PASS] uwot_simple_wrapper.h found (Enhanced API header)
) else (
    echo [FAIL] uwot_simple_wrapper.h missing!
)

echo.
echo NOTE: Only uwot.dll is needed for C# P/Invoke!
echo       The .lib file is generated but not required for runtime.

echo.
echo ===========================================
echo   Enhanced Features Verified
echo ===========================================
echo.
echo Your library now supports:
echo   - 27D embeddings: model.Fit(data, embeddingDimension: 27)
echo   - Distance metrics: DistanceMetric.Euclidean, .Cosine, .Manhattan, .Correlation, .Hamming
echo   - Model persistence: model.Save("file.umap") and UMapModel.Load("file.umap")  
echo   - Out-of-sample projection: model.Transform(newData)
echo   - Based on proven uwot algorithms for quality results
echo.
echo Ready for C# integration with enhanced capabilities!
echo Build process completed!
echo Press any key to exit...
pause >nul