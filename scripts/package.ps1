#!/usr/bin/env pwsh
# PowerShell packaging script for Game Engine

param(
    [Parameter(Position = 0)]
    [ValidateSet('nsis', 'zip', 'portable', 'deb', 'rpm', 'all', 'clean', 'help')]
    [string]$PackageType = 'help'
)

$VERSION = "0.1.0"
$PROJECT_ROOT = Split-Path -Parent $PSCommandPath
$BUILD_DIR = Join-Path $PROJECT_ROOT "build"
$PACKAGE_DIR = Join-Path $BUILD_DIR "packages"
$RELEASE_DIR = Join-Path $BUILD_DIR "Release"

# Utility functions
function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 50) -ForegroundColor Green
    Write-Host $Message -ForegroundColor Green
    Write-Host ("=" * 50) -ForegroundColor Green
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "→ $Message" -ForegroundColor Cyan
}

function Show-Help {
    Write-Header "Game Engine Packaging Tool - Help"
    Write-Host "Usage: .\package.ps1 [PackageType]`n"
    Write-Host "Package Types:" -ForegroundColor Yellow
    Write-Host "  nsis     - Create NSIS Windows installer"
    Write-Host "  zip      - Create ZIP portable bundle"
    Write-Host "  portable - Create portable directory structure"
    Write-Host "  deb      - Create Debian package (Linux)"
    Write-Host "  rpm      - Create RPM package (Linux)"
    Write-Host "  all      - Create all available packages"
    Write-Host "  clean    - Clean package directory"
    Write-Host "  help     - Show this help message`n"
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\package.ps1 portable"
    Write-Host "  .\package.ps1 all"
    Write-Host "  .\package.ps1 clean`n"
}

function Test-BuildExists {
    if (-not (Test-Path $RELEASE_DIR)) {
        Write-Error "Release build not found at $RELEASE_DIR"
        Write-Info "Build the project first: .\build.ps1 release"
        exit 1
    }
    Write-Success "Release build found"
}

function New-PortableBundle {
    Write-Header "Creating Portable Bundle"
    
    $portableDir = Join-Path $PACKAGE_DIR "GameEngine-portable"
    
    if (Test-Path $portableDir) {
        Remove-Item -Path $portableDir -Recurse -Force
    }
    
    New-Item -ItemType Directory -Path "$portableDir\bin" -Force > $null
    
    Write-Info "Copying executable..."
    Copy-Item -Path "$RELEASE_DIR\GameEngine.exe" -Destination "$portableDir\bin\"
    
    Write-Info "Copying assets..."
    if (Test-Path "$PROJECT_ROOT\assets") {
        Copy-Item -Path "$PROJECT_ROOT\assets" -Destination "$portableDir\" -Recurse -Force
    }
    
    Write-Info "Copying shaders..."
    if (Test-Path "$PROJECT_ROOT\shaders") {
        Copy-Item -Path "$PROJECT_ROOT\shaders" -Destination "$portableDir\" -Recurse -Force
    }
    
    # Create launcher batch
    $launcherContent = @'
@echo off
cd /d "%~dp0"
bin\GameEngine.exe %*
pause
'@
    Set-Content -Path "$portableDir\GameEngine.bat" -Value $launcherContent
    
    # Create silent launcher VBScript
    $silentContent = @'
' Silent launcher for Game Engine
Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")
strPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
objShell.Run """" & strPath & "\bin\GameEngine.exe""", 0, False
'@
    Set-Content -Path "$portableDir\GameEngine-silent.vbs" -Value $silentContent
    
    # Create README
    $readmeContent = @"
Game Engine - Windows Portable Edition
=======================================

This is a portable, standalone build of the Game Engine.
No installation required - simply extract and run!

Quick Start:
  1. Double-click GameEngine.bat to run
  2. Or click GameEngine-silent.vbs to run without console window

System Requirements:
  - Windows 7 SP1 or later
  - 4GB RAM minimum
  - OpenGL 3.3+ capable graphics card
  - 500MB free disk space

For more information:
  Homepage: https://github.com/YOUR_USERNAME/game-engine
"@
    Set-Content -Path "$portableDir\README.txt" -Value $readmeContent
    
    Write-Success "Portable bundle created at: $portableDir"
}

function New-ZipArchive {
    Write-Header "Creating ZIP Archive"
    
    $portableDir = Join-Path $PACKAGE_DIR "GameEngine-portable"
    $zipFile = Join-Path $PACKAGE_DIR "GameEngine-${VERSION}-windows-portable.zip"
    
    if (-not (Test-Path $portableDir)) {
        Write-Error "Portable bundle not found. Create it first."
        return
    }
    
    Write-Info "Creating: $(Split-Path -Leaf $zipFile)"
    
    if (Get-Command Compress-Archive -ErrorAction SilentlyContinue) {
        Compress-Archive -Path $portableDir -DestinationPath $zipFile -Force
        Write-Success "Created: $(Split-Path -Leaf $zipFile) ($('{0:F1}' -f ((Get-Item $zipFile).Length / 1MB)) MB)"
    } else {
        Write-Error "Compress-Archive not available"
    }
}

function New-NsisInstaller {
    Write-Header "Creating NSIS Installer"
    
    $nsisScript = Join-Path $PACKAGE_DIR "installer.nsi"
    $portableDir = Join-Path $PACKAGE_DIR "GameEngine-portable"
    
    if (-not (Test-Path $portableDir)) {
        Write-Error "Portable bundle not found. Create it first."
        return
    }
    
    Write-Info "Creating NSIS script..."
    
    $nsisContent = @"
; Game Engine NSIS Installer
!include "MUI2.nsh"

Name "Game Engine $VERSION"
OutFile "$PACKAGE_DIR\GameEngine-${VERSION}-installer.exe"
InstallDir "`$PROGRAMFILES\GameEngine"

!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "Install"
  SetOutPath "`$INSTDIR"
  File /r "$portableDir\*.*"
  CreateDirectory "`$SMPROGRAMS\Game Engine"
  CreateShortCut "`$SMPROGRAMS\Game Engine\Game Engine.lnk" "`$INSTDIR\bin\GameEngine.exe"
SectionEnd

Section "Uninstall"
  RMDir /r "`$INSTDIR"
  RMDir /r "`$SMPROGRAMS\Game Engine"
SectionEnd
"@
    
    Set-Content -Path $nsisScript -Value $nsisContent
    
    $makensis = Get-Command makensis -ErrorAction SilentlyContinue
    
    if ($makensis) {
        Write-Info "Building with NSIS..."
        & $makensis.Source $nsisScript
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Created NSIS installer"
        } else {
            Write-Error "NSIS build failed"
        }
    } else {
        Write-Info "NSIS not installed. Script saved at: $nsisScript"
        Write-Info "Download from: https://nsis.sourceforge.io/"
    }
}

function New-DebPackage {
    Write-Header "Creating Debian Package"
    Write-Info "Use 'cpack -G DEB' in the build directory on Linux"
}

function New-RpmPackage {
    Write-Header "Creating RPM Package"
    Write-Info "Use 'cpack -G RPM' in the build directory on Linux"
}

function Clear-Packages {
    Write-Header "Cleaning Packages"
    
    if (Test-Path $PACKAGE_DIR) {
        Remove-Item -Path $PACKAGE_DIR -Recurse -Force
        Write-Success "Cleaned: $PACKAGE_DIR"
    } else {
        Write-Info "Nothing to clean"
    }
}

# Main execution
function Main {
    if (-not (Test-Path $PACKAGE_DIR)) {
        New-Item -ItemType Directory -Path $PACKAGE_DIR -Force > $null
    }
    
    switch ($PackageType) {
        'portable' {
            Test-BuildExists
            New-PortableBundle
        }
        'zip' {
            Test-BuildExists
            New-PortableBundle
            New-ZipArchive
        }
        'nsis' {
            Test-BuildExists
            New-PortableBundle
            New-NsisInstaller
        }
        'deb' {
            New-DebPackage
        }
        'rpm' {
            New-RpmPackage
        }
        'all' {
            Test-BuildExists
            New-PortableBundle
            New-ZipArchive
            if ($PSVersionTable.OS -match "Windows") {
                New-NsisInstaller
            }
        }
        'clean' {
            Clear-Packages
        }
        'help' {
            Show-Help
        }
        default {
            Write-Error "Unknown package type: $PackageType"
            Show-Help
        }
    }
    
    if ($PackageType -ne 'help') {
        Write-Host ""
        Write-Info "Packages ready at: $PACKAGE_DIR"
    }
}

Main
