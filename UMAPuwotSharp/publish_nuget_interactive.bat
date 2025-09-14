@echo off
setlocal enabledelayedexpansion
echo Publishing UMAP NuGet Package to NuGet.org
echo.

REM Find the latest UMAPuwotSharp package
for /f %%i in ('dir /b /o-n "UMAPuwotSharp\bin\Release\UMAPuwotSharp.*.nupkg" 2^>nul') do (
    set "LATEST_PACKAGE=%%i"
    goto :found_package
)

:found_package
if "!LATEST_PACKAGE!"=="" (
    echo ❌ No NuGet package found! Run build_nuget.bat first.
    pause
    exit /b 1
)

REM Extract version from filename
for /f "tokens=2 delims=." %%a in ("!LATEST_PACKAGE!") do (
    for /f "tokens=1,2,3 delims=." %%b in ("%%a") do (
        set "VERSION=%%b.%%c.%%d"
    )
)

echo ✅ Package found: !LATEST_PACKAGE! (v!VERSION!)
echo.

REM Show package details
echo Package Information:
dir "UMAPuwotSharp\bin\Release\!LATEST_PACKAGE!"
dir "UMAPuwotSharp\bin\Release\UMAPuwotSharp.!VERSION!.snupkg" 2>nul
echo.

echo 🚀 READY TO PUBLISH TO NUGET.ORG!
echo.
echo ⚠️  This will publish your package to the public NuGet repository.
echo    Make sure you're ready for this!
echo.

REM Ask for confirmation
set /p "confirm=Do you want to proceed with publishing? (y/N): "
if /i not "!confirm!"=="y" (
    echo Publishing cancelled.
    pause
    exit /b 0
)

echo.
echo 🔐 Please enter your NuGet API key:
echo    (Get it from: https://www.nuget.org/account/apikeys)
echo.
set /p "apikey=API Key: "

if "!apikey!"=="" (
    echo ❌ No API key provided. Publishing cancelled.
    pause
    exit /b 1
)

echo.
echo 📦 Publishing package to NuGet.org...
echo.

REM Change to the package directory
cd UMAPuwotSharp\bin\Release

REM Execute the publish command
dotnet nuget push "!LATEST_PACKAGE!" --source https://api.nuget.org/v3/index.json --api-key "!apikey!"

if !ERRORLEVEL! EQU 0 (
    echo.
    echo 🎉 SUCCESS! Package published successfully!
    echo.
    echo 📍 Your package is now available at:
    echo    https://www.nuget.org/packages/UMAPuwotSharp/!VERSION!
    echo.
    echo ⏰ Note: It may take a few minutes to appear in search results.
    echo.
    echo 🚀 Your revolutionary UMAP v!VERSION! with HNSW optimization is now live!
) else (
    echo.
    echo ❌ Publishing failed! Error code: !ERRORLEVEL!
    echo.
    echo Common issues:
    echo - Invalid API key
    echo - Package version already exists
    echo - Network connectivity issues
    echo - Package validation errors
    echo.
    echo Please check the error message above and try again.
)

echo.
pause