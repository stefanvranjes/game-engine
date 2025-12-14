@echo off
REM Build and Package Script for Game Engine
REM Creates installable packages for Windows

setlocal enabledelayedexpansion

set VERSION=0.1.0
set PROJECT_ROOT=%~dp0
set BUILD_DIR=%PROJECT_ROOT%build
set PACKAGE_DIR=%BUILD_DIR%\packages
set INSTALL_PREFIX=%BUILD_DIR%\install

if "%1%"=="" (
    echo Usage: package.bat [option]
    echo.
    echo Options:
    echo   nsis       - Create NSIS installer
    echo   zip        - Create ZIP portable bundle
    echo   portable   - Create portable directory structure
    echo   all        - Create all package types
    echo   clean      - Clean package directory
    echo.
    exit /b 0
)

:parse_args
if "%1%"=="" goto build_start

if /i "%1%"=="nsis" (
    set PACKAGE_NSIS=1
) else if /i "%1%"=="zip" (
    set PACKAGE_ZIP=1
) else if /i "%1%"=="portable" (
    set PACKAGE_PORTABLE=1
) else if /i "%1%"=="all" (
    set PACKAGE_NSIS=1
    set PACKAGE_ZIP=1
    set PACKAGE_PORTABLE=1
) else if /i "%1%"=="clean" (
    echo Cleaning package directory...
    if exist "%PACKAGE_DIR%" rmdir /s /q "%PACKAGE_DIR%"
    exit /b 0
)

shift
goto parse_args

:build_start
echo.
echo =====================================
echo Game Engine Packager
echo =====================================
echo.

REM Create package directory
if not exist "%PACKAGE_DIR%" mkdir "%PACKAGE_DIR%"

REM Build Release if not already built
if not exist "%BUILD_DIR%\Release\GameEngine.exe" (
    echo Building Release configuration...
    cd /d "%BUILD_DIR%"
    cmake --build . --config Release
    cd /d "%PROJECT_ROOT%"
)

REM Create portable bundle
if defined PACKAGE_PORTABLE (
    echo.
    echo Creating portable bundle...
    call :create_portable
)

REM Create ZIP archive
if defined PACKAGE_ZIP (
    echo.
    echo Creating ZIP archive...
    call :create_zip
)

REM Create NSIS installer
if defined PACKAGE_NSIS (
    echo.
    echo Creating NSIS installer...
    call :create_nsis
)

echo.
echo =====================================
echo Packaging complete!
echo =====================================
echo Packages location: %PACKAGE_DIR%
echo.

exit /b 0

:create_portable
setlocal
set PORTABLE_DIR=%PACKAGE_DIR%\GameEngine-portable
if exist "%PORTABLE_DIR%" rmdir /s /q "%PORTABLE_DIR%"
mkdir "%PORTABLE_DIR%\bin"

echo Copying executable...
copy "%BUILD_DIR%\Release\GameEngine.exe" "%PORTABLE_DIR%\bin\"
copy /s "%PROJECT_ROOT%assets" "%PORTABLE_DIR%\"
copy /s "%PROJECT_ROOT%shaders" "%PORTABLE_DIR%\"

REM Create launcher batch
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo bin\GameEngine.exe %%*
) > "%PORTABLE_DIR%\GameEngine.bat"

REM Create README
(
    echo Game Engine - Windows Portable
    echo.
    echo Simply extract and run GameEngine.bat
    echo No installation required!
) > "%PORTABLE_DIR%\README.txt"

echo Created portable bundle at: %PORTABLE_DIR%
endlocal
exit /b 0

:create_zip
setlocal
set PORTABLE_DIR=%PACKAGE_DIR%\GameEngine-portable
set ZIP_FILE=%PACKAGE_DIR%\GameEngine-%VERSION%-windows-portable.zip

if not exist "%PORTABLE_DIR%" (
    echo Portable bundle not found. Create it first.
    exit /b 1
)

echo Creating ZIP: %ZIP_FILE%

REM Use PowerShell to create ZIP
powershell -NoProfile -Command ^
    "Add-Type -AssemblyName System.IO.Compression.FileSystem; ^
    [System.IO.Compression.ZipFile]::CreateFromDirectory('%PORTABLE_DIR%', '%ZIP_FILE%')"

if %ERRORLEVEL% equ 0 (
    echo Created: %ZIP_FILE%
) else (
    echo Failed to create ZIP
    exit /b 1
)

endlocal
exit /b 0

:create_nsis
setlocal
set NSIS_SCRIPT=%PACKAGE_DIR%\installer.nsi
set PORTABLE_DIR=%PACKAGE_DIR%\GameEngine-portable

if not exist "%PORTABLE_DIR%" (
    echo Portable bundle not found. Create it first.
    exit /b 1
)

echo Creating NSIS installer script...

(
    echo ; Game Engine NSIS Installer Script
    echo !include "MUI2.nsh"
    echo.
    echo ; Basic settings
    echo Name "Game Engine %VERSION%"
    echo OutFile "%PACKAGE_DIR%\GameEngine-%VERSION%-installer.exe"
    echo InstallDir "$PROGRAMFILES\GameEngine"
    echo.
    echo ; Pages
    echo !insertmacro MUI_PAGE_DIRECTORY
    echo !insertmacro MUI_PAGE_INSTFILES
    echo !insertmacro MUI_LANGUAGE "English"
    echo.
    echo Section "Install"
    echo   SetOutPath "$INSTDIR"
    echo   File /r "%PORTABLE_DIR%\*.*"
    echo   CreateDirectory "$SMPROGRAMS\Game Engine"
    echo   CreateShortCut "$SMPROGRAMS\Game Engine\Game Engine.lnk" "$INSTDIR\bin\GameEngine.exe"
    echo   CreateShortCut "$SMPROGRAMS\Game Engine\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
    echo SectionEnd
    echo.
    echo Section "Uninstall"
    echo   RMDir /r "$INSTDIR"
    echo   RMDir /r "$SMPROGRAMS\Game Engine"
    echo SectionEnd
) > "%NSIS_SCRIPT%"

REM Check if NSIS is installed
where makensis >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Building NSIS installer...
    makensis "%NSIS_SCRIPT%"
    if %ERRORLEVEL% equ 0 (
        echo Created: GameEngine-%VERSION%-installer.exe
    ) else (
        echo NSIS build failed
    )
) else (
    echo NSIS not found. Install from: https://nsis.sourceforge.io/
    echo Script saved at: %NSIS_SCRIPT%
)

endlocal
exit /b 0
