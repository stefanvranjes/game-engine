#!/bin/bash
# Windows Portable Bundle Creator
# Creates portable ZIP bundles for Windows distribution

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build/windows-portable"
BUNDLE_DIR="$BUILD_DIR/GameEngine"
VERSION="0.1.0"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# Create windows portable bundle
create_windows_portable() {
    print_header "Creating Windows Portable Bundle"
    
    # Clean previous build
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUNDLE_DIR"
    
    print_info "Copying executable..."
    mkdir -p "$BUNDLE_DIR/bin"
    cp "$PROJECT_ROOT/build/Debug/GameEngine.exe" "$BUNDLE_DIR/bin/" 2>/dev/null || \
    cp "$PROJECT_ROOT/build/Release/GameEngine.exe" "$BUNDLE_DIR/bin/" 2>/dev/null || \
    print_error "Could not find GameEngine.exe. Build the project first."
    
    print_info "Copying assets..."
    cp -r "$PROJECT_ROOT/assets" "$BUNDLE_DIR/" 2>/dev/null || true
    
    print_info "Copying shaders..."
    cp -r "$PROJECT_ROOT/shaders" "$BUNDLE_DIR/" 2>/dev/null || true
    
    print_success "Files copied"
}

# Create launcher batch script
create_launcher() {
    print_header "Creating Launcher Scripts"
    
    # Batch launcher
    cat > "$BUNDLE_DIR/GameEngine.bat" << 'EOF'
@echo off
REM Game Engine Launcher

cd /d "%~dp0"
bin\GameEngine.exe %*
pause
EOF
    
    # VBScript launcher (silent)
    cat > "$BUNDLE_DIR/GameEngine-silent.vbs" << 'EOF'
' Silent launcher for Game Engine
Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")

strPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
objShell.Run """" & strPath & "\bin\GameEngine.exe""", 0, False
EOF
    
    print_success "Launcher scripts created"
}

# Create documentation
create_docs() {
    print_header "Creating Documentation"
    
    cat > "$BUNDLE_DIR/README.txt" << 'EOF'
Game Engine - Windows Portable Edition
=======================================

This is a portable, standalone build of the Game Engine.
No installation required - simply extract and run!

Quick Start:
  1. Double-click GameEngine.bat to run
  2. Or click GameEngine-silent.vbs to run without console window

System Requirements:
  - Windows 7 SP1 or later (Vista, 7, 8, 8.1, 10, 11)
  - 4GB RAM minimum (8GB recommended)
  - OpenGL 3.3+ capable graphics card
  - 500MB free disk space

Features:
  - Deferred Rendering
  - Advanced Post-Processing (SSAO, SSR, TAA, Bloom)
  - Skeletal Animation with IK
  - Particle System with GPU Compute
  - 3D Audio
  - Multiplayer Networking

Portable Notes:
  - You can copy this folder anywhere
  - You can rename this folder
  - Works on USB drives (slower startup)
  - No registry entries created

Troubleshooting:
  - If the program doesn't start, ensure you have updated graphics drivers
  - Check that OpenGL 3.3+ is supported by your GPU
  - Some antivirus software may flag the executable

For more information:
  Homepage: https://github.com/YOUR_USERNAME/game-engine
  Issues: https://github.com/YOUR_USERNAME/game-engine/issues
EOF
    
    print_success "Documentation created"
}

# Create ZIP archive
create_zip() {
    print_header "Creating ZIP Archive"
    
    cd "$BUILD_DIR"
    
    local zipfile="GameEngine-${VERSION}-windows-portable.zip"
    print_info "Creating: $zipfile"
    
    if command -v zip &> /dev/null; then
        zip -r "$zipfile" GameEngine/
    elif command -v 7z &> /dev/null; then
        7z a "$zipfile" GameEngine/
    else
        print_error "zip or 7z not found. Cannot create archive."
        return 1
    fi
    
    print_success "Created: $zipfile ($(du -h "$zipfile" | cut -f1))"
}

# Generate checksums
generate_checksums() {
    print_header "Generating Checksums"
    
    cd "$BUILD_DIR"
    
    if command -v sha256sum &> /dev/null; then
        sha256sum GameEngine-*.zip > checksums.sha256
        print_success "SHA256 checksums generated"
    fi
    
    if command -v md5sum &> /dev/null; then
        md5sum GameEngine-*.zip > checksums.md5
        print_success "MD5 checksums generated"
    fi
}

# Summary
print_summary() {
    print_header "Portable Bundle Created"
    
    echo "Bundle Information:"
    echo "  Version: $VERSION"
    echo "  Location: $BUNDLE_DIR"
    echo "  Size: $(du -sh "$BUNDLE_DIR" | cut -f1)"
    echo ""
    echo "To distribute:"
    echo "  1. Compress: GameEngine-${VERSION}-windows-portable.zip"
    echo "  2. Share or copy to USB"
    echo ""
    echo "Users can:"
    echo "  1. Extract the ZIP anywhere"
    echo "  2. Run GameEngine.bat or GameEngine-silent.vbs"
    echo ""
    echo "Files ready at: $BUILD_DIR"
}

# Main
main() {
    print_header "Windows Portable Bundle Creator"
    
    create_windows_portable
    create_launcher
    create_docs
    
    if command -v zip &> /dev/null || command -v 7z &> /dev/null; then
        create_zip
        generate_checksums
    else
        print_info "Skipping archive creation (zip/7z not available)"
    fi
    
    print_summary
    print_success "Windows portable bundle created!"
}

main "$@"
