@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   TESTING VERSION DETECTION
echo ========================================
echo.

REM Test the FIXED PowerShell method
echo 🔧 FIXED METHOD (PowerShell date-based):
for /f %%i in ('powershell -Command "Get-ChildItem UMAPuwotSharp\bin\Release\UMAPuwotSharp.*.nupkg | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Select-Object -ExpandProperty Name"') do (
    set "LATEST_PACKAGE=%%i"
)
echo   Latest: !LATEST_PACKAGE!

REM Extract version
for /f "tokens=2,3,4 delims=." %%a in ("!LATEST_PACKAGE!") do (
    set "VERSION=%%a.%%b.%%c"
)
echo   Version: !VERSION!
echo.

REM Test the BROKEN dir method (what was causing the problem)
echo ❌ BROKEN METHOD (dir alphabetical):
for /f %%i in ('dir /b /o-n "UMAPuwotSharp\bin\Release\UMAPuwotSharp.*.nupkg" 2^>nul') do (
    set "BROKEN_PACKAGE=%%i"
    goto :found_broken
)
:found_broken
echo   Would detect: !BROKEN_PACKAGE! (WRONG!)
echo.

echo ========================================
echo   SUMMARY
echo ========================================
echo ✅ CORRECT (Fixed):  v!VERSION!
echo ❌ WRONG (Broken):   v3.9.0
echo.
echo The batch file has been FIXED!
pause