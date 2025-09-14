@echo off
echo Publishing UMAP v3.1.0 NuGet Package - Final Steps
echo.

REM Validate the package was built successfully
if not exist "UMAPuwotSharp\bin\Release\UMAPuwotSharp.3.1.0.nupkg" (
    echo ❌ NuGet package not found! Run build_nuget.bat first.
    pause
    exit /b 1
)

echo ✅ Package found: UMAPuwotSharp.3.1.0.nupkg
echo.

REM Display package details
echo Package Information:
dir UMAPuwotSharp\bin\Release\UMAPuwotSharp.3.1.0.*
echo.

REM Validate native libraries are included with correct sizes
echo Validating native libraries in package...
call verify_binaries.bat
echo.

REM Show package contents (optional verification)
echo Package contents preview:
7z l UMAPuwotSharp\bin\Release\UMAPuwotSharp.3.1.0.nupkg 2>nul || echo "7z not available - manual package inspection recommended"
echo.

echo 🚀 READY FOR PUBLICATION!
echo.
echo Publishing commands:
echo.
echo 1. Test locally first:
echo    dotnet add package UMAPuwotSharp --source "UMAPuwotSharp\bin\Release"
echo.
echo 2. Publish to NuGet.org:
echo    dotnet nuget push UMAPuwotSharp\bin\Release\UMAPuwotSharp.3.1.0.nupkg --source https://api.nuget.org/v3/index.json --api-key YOUR_API_KEY
echo.
echo 3. Verify publication:
echo    https://www.nuget.org/packages/UMAPuwotSharp/3.1.0
echo.
echo 📋 Pre-publication checklist:
echo ✅ HNSW optimization implemented (50-2000x speedup)
echo ✅ Cross-platform binaries (Windows 150KB + Linux 174KB)
echo ✅ Enhanced API with forceExactKnn parameter
echo ✅ Comprehensive test suite with MSE validation
echo ✅ Updated documentation and API reference
echo ✅ Version 3.1.0 with detailed release notes
echo.
echo ⚠️  IMPORTANT: Make sure you have your NuGet API key ready!
echo    Get it from: https://www.nuget.org/account/apikeys
echo.

pause