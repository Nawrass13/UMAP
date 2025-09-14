@echo off
echo Verifying Cross-Platform Native Libraries for NuGet Package...
echo.

cd UMAPuwotSharp

echo Checking Windows native library...
if exist "runtimes\win-x64\native\uwot.dll" (
    for %%A in ("runtimes\win-x64\native\uwot.dll") do (
        set size=%%~zA
        echo ✅ Windows library: %%~zA bytes
        if %%~zA LSS 100000 (
            echo ⚠️  WARNING: Windows library seems small. Expected ~150KB with HNSW.
        ) else (
            echo ✅ Windows library size looks good for HNSW optimization.
        )
    )
) else (
    echo ❌ Windows native library not found!
)

echo.
echo Checking Linux native library...
if exist "runtimes\linux-x64\native\libuwot.so" (
    for %%A in ("runtimes\linux-x64\native\libuwot.so") do (
        echo ✅ Linux library: %%~zA bytes
        if %%~zA LSS 100000 (
            echo ⚠️  WARNING: Linux library seems small. Expected ~174KB with HNSW.
            echo This was the issue in v3.0.0 - make sure BuildDockerLinuxWindows.bat was run!
        ) else (
            echo ✅ Linux library size looks good for HNSW optimization.
        )
    )
) else (
    echo ❌ Linux native library not found!
)

echo.
echo Library comparison:
echo Expected sizes with HNSW optimization:
echo   Windows (uwot.dll): ~150KB
echo   Linux (libuwot.so): ~174KB
echo.
echo If libraries are smaller than expected, run:
echo   cd ..\uwot_pure_cpp
echo   BuildDockerLinuxWindows.bat
echo.

REM Check if README is included
if exist "README.md" (
    echo ✅ README.md found for NuGet package description
) else (
    echo ⚠️  README.md not found in project root
)

echo.
echo Cross-platform binary verification completed!
pause