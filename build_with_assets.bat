@echo off
REM Asset Pipeline Build Script
REM Usage: build_with_assets.bat [clean] [full-rebuild] [compress]

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "ASSETS_DIR=%SCRIPT_DIR%assets"
set "ASSETS_PROCESSED=%ASSETS_DIR%\.processed"
set "DATABASE_FILE=%ASSETS_DIR%\.database.json"

REM Parse arguments
set "CLEAN_BUILD=0"
set "FULL_REBUILD=0"
set "COMPRESS=0"

for %%A in (%*) do (
    if "%%A"=="clean" set "CLEAN_BUILD=1"
    if "%%A"=="full-rebuild" set "FULL_REBUILD=1"
    if "%%A"=="compress" set "COMPRESS=1"
)

echo.
echo ========================================
echo Asset Pipeline Build System
echo ========================================
echo.

REM Step 1: Validate directories
if not exist "%ASSETS_DIR%" (
    echo Error: Assets directory not found: %ASSETS_DIR%
    exit /b 1
)

if not exist "%BUILD_DIR%" (
    echo Creating build directory...
    mkdir "%BUILD_DIR%"
)

REM Step 2: Clean if requested
if %CLEAN_BUILD%==1 (
    echo Cleaning processed assets...
    if exist "%ASSETS_PROCESSED%" (
        rmdir /s /q "%ASSETS_PROCESSED%"
    )
    if exist "%DATABASE_FILE%" (
        del "%DATABASE_FILE%"
    )
    echo Clean complete.
    echo.
)

REM Step 3: CMake configuration
echo Configuring CMake...
cd /d "%BUILD_DIR%"
cmake .. -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% neq 0 (
    echo Error: CMake configuration failed
    exit /b 1
)

REM Step 4: Build GameEngine
echo.
echo Building game engine...
cmake --build . --config Debug --parallel

if %ERRORLEVEL% neq 0 (
    echo Error: Build failed
    exit /b 1
)

REM Step 5: Run asset pipeline
echo.
echo ========================================
echo Running Asset Pipeline
echo ========================================
echo.

if %COMPRESS%==1 (
    echo Mode: Compressed (optimization)
) else (
    echo Mode: Development (fast iteration)
)

if %FULL_REBUILD%==1 (
    echo Strategy: Full rebuild
) else (
    echo Strategy: Incremental (only changed assets)
)

echo.
echo Asset source directory: %ASSETS_DIR%
echo Output directory: %ASSETS_PROCESSED%
echo Database file: %DATABASE_FILE%
echo.

REM Run the built executable with asset pipeline
cd /d "%SCRIPT_DIR%"

if exist "%BUILD_DIR%\Debug\GameEngine.exe" (
    echo Running asset pipeline...
    "%BUILD_DIR%\Debug\GameEngine.exe" --process-assets
    
    if %ERRORLEVEL% neq 0 (
        echo Warning: Asset processing reported issues
    )
) else (
    echo Error: GameEngine executable not found
    exit /b 1
)

echo.
echo ========================================
echo Build Complete
echo ========================================
echo.
echo Output: %BUILD_DIR%\Debug\GameEngine.exe
echo Assets: %ASSETS_PROCESSED%
echo.

REM Optionally run the game
if not "%1"=="no-run" (
    echo Launching game...
    "%BUILD_DIR%\Debug\GameEngine.exe"
)

endlocal
exit /b 0
