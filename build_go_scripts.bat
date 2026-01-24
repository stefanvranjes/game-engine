@echo off
REM build_go_scripts.bat - Build Go scripts for Windows

setlocal enabledelayedexpansion

REM Set output directory
set OUTPUT_DIR=scripts\build
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Detect Go installation
where go >nul 2>&1
if errorlevel 1 (
    echo Go not found in PATH. Please install Go 1.21+
    exit /b 1
)

echo Building Go scripts for Windows...

REM Build each .go file
for %%F in (scripts\*.go) do (
    set "filename=%%~nF"
    set "name=!filename:.go=!"
    
    if not "!name!"=="example_go_systems" (
        echo Building !name!...
        go build -o "%OUTPUT_DIR%\!name!.dll" -buildmode=c-shared "%%F"
        if errorlevel 1 (
            echo Failed to build !name!
            exit /b 1
        )
    )
)

REM Build example systems
echo Building example_go_systems...
go build -o "%OUTPUT_DIR%\example_go_systems.dll" -buildmode=c-shared scripts\example_go_systems.go
if errorlevel 1 (
    echo Failed to build example_go_systems
    exit /b 1
)

echo.
echo Go scripts built successfully to %OUTPUT_DIR%
echo.
dir "%OUTPUT_DIR%\*.dll"

endlocal
