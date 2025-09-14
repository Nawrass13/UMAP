@echo off
echo Testing Cross-Platform Build with CMake (Windows)...
echo.

REM Clean any previous build
if exist build rmdir /s /q build
mkdir build
cd build

echo Configuring with CMake (Visual Studio 2022)...
cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTS=ON

if errorlevel 1 (
    echo ❌ CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo Building Release configuration...
cmake --build . --config Release

if errorlevel 1 (
    echo ❌ Build failed!
    pause
    exit /b 1
)

echo.
echo Running tests...
ctest -C Release --output-on-failure

if errorlevel 1 (
    echo ⚠️  Some tests failed, but build completed successfully.
) else (
    echo ✅ All tests passed!
)

echo.
echo Build artifacts:
dir Release\*.dll 2>nul
dir Release\*.exe 2>nul

echo.
echo ✅ Cross-platform build test completed!
pause