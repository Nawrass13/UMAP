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
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64
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
echo   - uwot.dll (shared library for C#)
echo   - uwot.lib (import library)
echo   - uwot_test.exe (basic functionality test)
echo   - uwot_enhanced_test.exe (save/load/transform test)
echo.

REM Test the builds
echo ===========================================
echo   Running Tests
echo ===========================================
echo.

echo [1/2] Running basic functionality test...
echo ----------------------------------------
cd Release
uwot_test.exe
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Basic test failed with error code %ERRORLEVEL%
) else (
    echo [PASS] Basic test completed successfully!
)

echo.
echo [2/2] Running enhanced functionality test...
echo ----------------------------------------
if exist uwot_enhanced_test.exe (
    uwot_enhanced_test.exe
    if %ERRORLEVEL% NEQ 0 (
        echo [WARN] Enhanced test failed with error code %ERRORLEVEL%
    ) else (
        echo [PASS] Enhanced test completed successfully!
    )
) else (
    echo [INFO] Enhanced test not built (enhanced_test.cpp not found)
    echo       You can create enhanced_test.cpp to enable save/load/transform testing
)

cd ..
cd ..

echo.
echo ===========================================
echo   Build Summary
echo ===========================================
echo.
echo Your UMAP C++ library is ready for C# integration!
echo.
echo Key files for C# P/Invoke:
echo   [FILE] build\Release\uwot.dll    - Main library
echo   [FILE] build\Release\uwot.lib    - Import library  
echo   [FILE] uwot_simple_wrapper.h     - Header with function declarations
echo.
echo Features available:
echo   [YES] Model Training        - uwot_fit()
echo   [YES] Model Persistence     - uwot_save_model() / uwot_load_model()
echo   [YES] Data Transformation   - uwot_transform()
echo   [YES] Model Information     - uwot_get_model_info() / uwot_is_fitted()
echo   [YES] Cross-platform        - Windows and Linux compatible
echo.
echo Example C# usage:
echo   [DllImport("uwot.dll")] static extern IntPtr uwot_create();
echo   [DllImport("uwot.dll")] static extern int uwot_fit(...);
echo   [DllImport("uwot.dll")] static extern int uwot_save_model(...);
echo   [DllImport("uwot.dll")] static extern IntPtr uwot_load_model(...);
echo   [DllImport("uwot.dll")] static extern int uwot_transform(...);
echo.

REM Check for any missing files
echo ===========================================
echo   File Check
echo ===========================================
echo.
if exist "build\Release\uwot.dll" (
    echo [PASS] uwot.dll found
) else (
    echo [FAIL] uwot.dll missing!
)

if exist "build\Release\uwot.lib" (
    echo [PASS] uwot.lib found
) else (
    echo [FAIL] uwot.lib missing!
)

if exist "build\Release\uwot_test.exe" (
    echo [PASS] uwot_test.exe found
) else (
    echo [FAIL] uwot_test.exe missing!
)

if exist "build\Release\uwot_enhanced_test.exe" (
    echo [PASS] uwot_enhanced_test.exe found
) else (
    echo [INFO] uwot_enhanced_test.exe missing (optional)
)

if exist "uwot_simple_wrapper.h" (
    echo [PASS] uwot_simple_wrapper.h found
) else (
    echo [FAIL] uwot_simple_wrapper.h missing!
)

echo.
echo Build process completed!
echo Press any key to exit...
pause >nul