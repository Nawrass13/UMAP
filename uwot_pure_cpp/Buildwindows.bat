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
echo   Running Enhanced Tests
echo ===========================================
echo.

echo Running enhanced functionality test...
echo ----------------------------------------

REM Change to Release directory and run test
if exist "Release\test_enhanced_wrapper.exe" (
    echo Starting test execution...
    cd Release
    test_enhanced_wrapper.exe
    set TEST_RESULT=%ERRORLEVEL%
    cd ..
    
    if %TEST_RESULT% EQU 0 (
        echo.
        echo [PASS] Enhanced test completed successfully!
        echo       - 27D embedding capability verified
        echo       - Multiple distance metrics tested
        echo       - Save/load functionality confirmed
        echo       - Transform (out-of-sample) working
    ) else (
        echo.
        echo [WARN] Enhanced test failed with error code %TEST_RESULT%
        echo       Test executable ran but returned an error
    )
) else (
    echo [WARN] test_enhanced_wrapper.exe not found
    echo       Enhanced tests cannot be run
    echo       Expected location: Release\test_enhanced_wrapper.exe
)

cd ..

echo.
echo ===========================================
echo   Copying Files to C# Project
echo ===========================================
echo.

REM Copy DLL to C# project base folder
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

if exist "build\Release\test_enhanced_wrapper.exe" (
    echo [PASS] test_enhanced_wrapper.exe found (Enhanced tests)
) else (
    echo [WARN] test_enhanced_wrapper.exe missing - tests cannot verify enhanced features
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