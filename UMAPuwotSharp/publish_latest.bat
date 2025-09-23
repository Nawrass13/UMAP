@echo off
setlocal enabledelayedexpansion

echo =====================================
echo   UMAP NuGet Publisher v3.12.0
echo =====================================
echo.

cd /d "%~dp0"

REM Use PowerShell to get the latest package reliably
for /f %%i in ('powershell -Command "Get-ChildItem UMAPuwotSharp\bin\Release\UMAPuwotSharp.*.nupkg | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Select-Object -ExpandProperty Name"') do (
    set "LATEST_PACKAGE=%%i"
)

if "!LATEST_PACKAGE!"=="" (
    echo ❌ No NuGet package found!
    echo Run: dotnet pack --configuration Release
    pause
    exit /b 1
)

REM Extract version from filename
for /f "tokens=2 delims=." %%a in ("!LATEST_PACKAGE!") do (
    for /f "tokens=1,2,3 delims=." %%b in ("%%a") do (
        set "VERSION=%%b.%%c.%%d"
    )
)

echo ✅ Latest package found: !LATEST_PACKAGE!
echo 📦 Version: v!VERSION!
echo.

REM Show package details
echo Package Information:
dir "UMAPuwotSharp\bin\Release\!LATEST_PACKAGE!" | find "!LATEST_PACKAGE!"
if exist "UMAPuwotSharp\bin\Release\UMAPuwotSharp.!VERSION!.snupkg" (
    dir "UMAPuwotSharp\bin\Release\UMAPuwotSharp.!VERSION!.snupkg" | find ".snupkg"
    echo ✅ Symbol package (.snupkg) found
) else (
    echo ⚠️  Symbol package (.snupkg) not found
)
echo.

echo 🚀 READY TO PUBLISH TO NUGET.ORG!
echo.
echo ⚠️  This will publish v!VERSION! to the public NuGet repository.
echo    Make sure you're ready for this critical update!
echo.

REM Ask for confirmation
set /p "confirm=Do you want to proceed with publishing v!VERSION!? (y/N): "
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
echo 📦 Publishing v!VERSION! to NuGet.org...
echo.

REM Change to the package directory
cd UMAPuwotSharp\bin\Release

REM Execute the publish command
dotnet nuget push "!LATEST_PACKAGE!" --source https://api.nuget.org/v3/index.json --api-key "!apikey!"

if !ERRORLEVEL! EQU 0 (
    echo.
    echo 🎉 SUCCESS! Package v!VERSION! published successfully!
    echo.
    echo 📍 Your package is now available at:
    echo    https://www.nuget.org/packages/UMAPuwotSharp/!VERSION!
    echo.
    echo ⏰ Note: It may take a few minutes to appear in search results.
    echo.
    echo 🚨 CRITICAL UPDATE v!VERSION! with testing fixes is now live!
) else (
    echo.
    echo ❌ Publishing failed! Error code: !ERRORLEVEL!
    echo.
    echo Common issues:
    echo - Invalid API key
    echo - Package version already exists on NuGet.org
    echo - Network connectivity issues
    echo - Package validation errors
    echo.
    echo Please check the error message above and try again.
)

echo.
pause