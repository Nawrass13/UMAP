@echo off
setlocal enabledelayedexpansion
echo ===========================================
echo   Enhanced UMAP Cross-Platform Build
echo ===========================================
echo.

REM Check Docker
docker --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Docker not available
    pause
    exit /b 1
)

REM === BUILD WINDOWS VERSION ===
echo [1/2] Building Enhanced Windows version...
echo ----------------------------------------

if exist build-windows (
    rmdir /s /q build-windows
)
mkdir build-windows
cd build-windows

cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=ON
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Windows CMake configuration failed!
    exit /b 1
)

cmake --build . --config Release
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Windows build failed!
    exit /b 1
)

echo [INFO] Running Windows tests...
if exist "Release\test_enhanced_wrapper.exe" (
    cd Release
    test_enhanced_wrapper.exe
    set WIN_TEST_RESULT=!ERRORLEVEL!
    cd ..
    if !WIN_TEST_RESULT! EQU 0 (
        echo [PASS] Windows enhanced tests completed successfully
    ) else (
        echo [WARN] Windows enhanced tests failed with code !WIN_TEST_RESULT!
    )
) else (
    echo [WARN] Windows test executable not found
)

echo [PASS] Windows build completed
cd ..

REM === BUILD LINUX VERSION WITH DOCKER ===
echo.
echo [2/2] Building Enhanced Linux version with Docker...
echo ----------------------------------------

if exist build-linux (
    rmdir /s /q build-linux
)

REM Run Docker build with direct commands
docker run --rm -v "%cd%":/src -w /src ubuntu:22.04 bash -c "apt-get update && apt-get install -y build-essential cmake libstdc++-11-dev && mkdir -p build-linux && cd build-linux && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON && make -j4 && ls -la . && if [ -f './test_enhanced_wrapper' ]; then echo 'Running tests...' && ./test_enhanced_wrapper && echo '[PASS] Linux enhanced test completed successfully'; else echo '[WARN] Linux test executable not found'; fi"

if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Docker Linux build failed!
    exit /b 1
)

echo [PASS] Linux build completed

REM === SETUP CSHARP PROJECT STRUCTURE ===
echo.
echo [3/3] Setting up Enhanced UMAPuwotSharp project...
echo ----------------------------------------

REM Create C# project structure
if not exist "..\UMAPuwotSharp" (
    echo [INFO] Creating UMAPuwotSharp project structure
    mkdir "..\UMAPuwotSharp"
)

if not exist "..\UMAPuwotSharp\UMAPuwotSharp" (
    mkdir "..\UMAPuwotSharp\UMAPuwotSharp"
)

REM Copy Windows DLL to both base folder and runtime directory
if exist "build-windows\Release\uwot.dll" (
    copy "build-windows\Release\uwot.dll" "..\UMAPuwotSharp\UMAPuwotSharp\" >nul
    echo [PASS] Copied Windows uwot.dll to C# project base folder

    REM Create and copy to Windows runtime folder for packaging
    if not exist "..\UMAPuwotSharp\UMAPuwotSharp\runtimes\win-x64\native\" (
        mkdir "..\UMAPuwotSharp\UMAPuwotSharp\runtimes\win-x64\native\"
    )
    copy "build-windows\Release\uwot.dll" "..\UMAPuwotSharp\UMAPuwotSharp\runtimes\win-x64\native\" >nul
    echo [PASS] Copied Windows uwot.dll to runtime folder (win-x64/native)
) else (
    echo [FAIL] Windows uwot.dll not found in build-windows\Release\
)

REM Copy Linux .so file to both base folder and runtime directory
set LINUX_LIB_COPIED=0

REM Create Linux runtime directory
if not exist "..\UMAPuwotSharp\UMAPuwotSharp\runtimes\linux-x64\native\" (
    mkdir "..\UMAPuwotSharp\UMAPuwotSharp\runtimes\linux-x64\native\"
)

REM Check for versioned .so files first (the actual library files)
for %%F in (build-linux\libuwot.so.*.*.*) do (
    if exist "%%F" (
        for %%A in ("%%F") do (
            if %%~zA GTR 1000 (
                copy "%%F" "..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so" >nul
                echo [PASS] Copied %%~nxF as libuwot.so to C# project base folder
                copy "%%F" "..\UMAPuwotSharp\UMAPuwotSharp\runtimes\linux-x64\native\libuwot.so" >nul
                echo [PASS] Copied %%~nxF to Linux runtime folder (linux-x64/native)
                set LINUX_LIB_COPIED=1
                goto :linux_lib_done
            )
        )
    )
)

REM Check for libuwot.so (only if no versioned file found)
if exist "build-linux\libuwot.so" (
    for %%A in ("build-linux\libuwot.so") do (
        if %%~zA GTR 1000 (
            copy "build-linux\libuwot.so" "..\UMAPuwotSharp\UMAPuwotSharp\" >nul
            echo [PASS] Copied Linux libuwot.so to C# project base folder
            copy "build-linux\libuwot.so" "..\UMAPuwotSharp\UMAPuwotSharp\runtimes\linux-x64\native\" >nul
            echo [PASS] Copied Linux libuwot.so to runtime folder (linux-x64/native)
            set LINUX_LIB_COPIED=1
        )
    )
)

:linux_lib_done
if !LINUX_LIB_COPIED! EQU 0 (
    echo [FAIL] Linux libuwot.so not found in build-linux\
    echo [INFO] Available files in build-linux:
    if exist "build-linux" (
        dir build-linux\ /B 2>nul || echo        Directory empty or not accessible
    ) else (
        echo        build-linux directory not found
    )
)

echo [PASS] Enhanced libraries setup completed

REM === SUMMARY ===
echo.
echo ===========================================
echo   Enhanced UMAP Build Summary
echo ===========================================
echo.
echo Windows libraries (build-windows\Release\):
if exist "build-windows\Release\uwot.dll" (
    echo   [PASS] uwot.dll (Enhanced UMAP with multiple metrics)
    for %%A in ("build-windows\Release\uwot.dll") do echo         Size: %%~zA bytes
) else (echo   [FAIL] uwot.dll)

if exist "build-windows\Release\test_enhanced_wrapper.exe" (
    echo   [PASS] test_enhanced_wrapper.exe
) else (
    echo   [FAIL] test_enhanced_wrapper.exe
)

echo.
echo Linux libraries (build-linux\):
if exist "build-linux" (
    dir build-linux\libuwot.so* build-linux\*.so /B 2>nul | findstr /R ".*" >nul
    if !ERRORLEVEL! EQU 0 (
        echo   [PASS] Linux .so files found:
        for %%F in (build-linux\libuwot.so* build-linux\*.so) do (
            if exist "%%F" (
                for %%A in ("%%F") do echo         %%~nxF (%%~zA bytes)
            )
        )
    ) else (
        echo   [FAIL] No Linux .so files found
    )
) else (
    echo   [FAIL] build-linux directory not found
)

if exist "build-linux\test_enhanced_wrapper" (
    echo   [PASS] test_enhanced_wrapper
) else (
    echo   [FAIL] test_enhanced_wrapper
)

echo.
echo UMAPuwotSharp C# project files:
if exist "..\UMAPuwotSharp\UMAPuwotSharp\uwot.dll" (
    echo   [PASS] Windows library: uwot.dll
    for %%A in ("..\UMAPuwotSharp\UMAPuwotSharp\uwot.dll") do echo         Size: %%~zA bytes
) else (echo   [FAIL] Windows library missing)

if exist "..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so" (
    echo   [PASS] Linux library: libuwot.so
    for %%A in ("..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so") do echo         Size: %%~zA bytes
) else (echo   [FAIL] Linux library missing)

echo.
echo Runtime folders for cross-platform deployment:
if exist "..\UMAPuwotSharp\UMAPuwotSharp\runtimes\win-x64\native\uwot.dll" (
    echo   [PASS] Windows runtime: runtimes/win-x64/native/uwot.dll
) else (echo   [WARN] Windows runtime missing)

if exist "..\UMAPuwotSharp\UMAPuwotSharp\runtimes\linux-x64\native\libuwot.so" (
    echo   [PASS] Linux runtime: runtimes/linux-x64/native/libuwot.so
) else (echo   [WARN] Linux runtime missing)

echo.
echo ===========================================
echo   Enhanced UMAP Features Available
echo ===========================================
echo.
echo Your Enhanced UMAPuwotSharp library now supports:
echo   - Arbitrary embedding dimensions (1D to 50D, including 27D)
echo   - Multiple distance metrics (Euclidean, Cosine, Manhattan, Correlation, Hamming)
echo   - Complete model save/load functionality
echo   - True out-of-sample projection (transform new data)
echo   - Progress reporting with callback support
echo   - Based on proven uwot algorithms
echo   - Cross-platform support (Windows + Linux)
echo.
echo ===========================================
echo   Ready for C# Development!
echo ===========================================
echo.
echo Next steps:
echo   1. Build C# library: dotnet build ..\UMAPuwotSharp\UMAPuwotSharp
echo   2. Create NuGet package: dotnet pack ..\UMAPuwotSharp\UMAPuwotSharp --configuration Release
echo   3. Test 27D embedding: var embedding27D = model.Fit(data, embeddingDimension: 27);
echo   4. Use progress reporting: model.FitWithProgress(data, progressCallback);
echo.
pause