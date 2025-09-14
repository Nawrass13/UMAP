@echo off
echo Building NuGet Package with HNSW Optimization...
echo.

REM First, ensure native libraries are built with HNSW
echo Step 1: Building cross-platform native libraries...
cd ..\uwot_pure_cpp

REM Use the Docker build script for cross-platform binaries
echo Running cross-platform build (this builds both Windows and Linux with HNSW)...
call BuildDockerLinuxWindows.bat
if errorlevel 1 (
    echo ❌ Cross-platform native library build failed!
    pause
    exit /b 1
)

echo ✅ Cross-platform native libraries built successfully!
echo.

REM Return to C# project and build NuGet package
cd ..\UMAPuwotSharp

echo Step 2: Building C# library and NuGet package...
dotnet clean UMAPuwotSharp
if errorlevel 1 (
    echo ⚠️  Clean failed, but continuing...
)

echo Building in Release configuration...
dotnet build UMAPuwotSharp --configuration Release
if errorlevel 1 (
    echo ❌ C# library build failed!
    pause
    exit /b 1
)

echo Running comprehensive test suite...
dotnet test --configuration Release --verbosity minimal
if errorlevel 1 (
    echo ⚠️  Some tests failed, but package can still be built.
    echo Check test results before publishing!
)

echo Creating NuGet package...
dotnet pack UMAPuwotSharp --configuration Release --no-build
if errorlevel 1 (
    echo ❌ NuGet package creation failed!
    pause
    exit /b 1
)

echo.
echo ✅ NuGet package build completed!

REM Show package information
echo Package details:
dir UMAPuwotSharp\bin\Release\*.nupkg
echo.

REM Verify native library sizes (should be ~150KB Windows, ~174KB Linux with HNSW)
echo Native library verification:
dir UMAPuwotSharp\runtimes\win-x64\native\uwot.dll 2>nul
dir UMAPuwotSharp\runtimes\linux-x64\native\libuwot.so 2>nul
echo.

echo 🎉 NuGet package ready for publishing!
echo.
echo Next steps:
echo 1. Test the package: Install locally and run tests
echo 2. Publish: dotnet nuget push --source https://api.nuget.org/v3/index.json
echo.

pause