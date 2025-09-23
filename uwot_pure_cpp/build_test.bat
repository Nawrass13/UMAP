@echo off
echo ===========================================
echo   Building and Testing All 4 Test Suites
echo ===========================================
echo.

REM Check if build directory exists, create if needed
if not exist build (
    echo Creating build directory...
    mkdir build
)

cd build

echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=ON
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

echo Building Release version...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    cd ..
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   Running All 4 Tests
echo ===========================================
echo.

set ALL_TESTS_PASSED=1

REM Change to Release directory
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
cd ..

echo.
echo ===========================================
echo   Test Results Summary
echo ===========================================
echo.
if %ALL_TESTS_PASSED% EQU 1 (
    echo [SUCCESS] ALL 4 TESTS PASSED!
    echo ✅ Standard comprehensive validation
    echo ✅ Error fixes validation
    echo ✅ Comprehensive pipeline validation
    echo ✅ Enhanced wrapper validation
    echo.
    echo UMAP modular architecture fully validated!
) else (
    echo [FAILURE] SOME TESTS FAILED!
    echo ❌ Check test output above for details
    echo.
    echo Build may have issues!
)

echo.
echo Press any key to exit...
pause >nul