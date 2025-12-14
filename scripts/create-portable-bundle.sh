#!/bin/bash
# Portable Bundle Creator for Game Engine
# Creates self-contained, relocatable application bundles

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_ROOT/build/portable"
BUNDLE_DIR="$BUILD_DIR/GameEngine-portable"
VERSION="0.1.0"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Utility functions
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

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    commands=("cmake" "make" "ldd")
    missing=0
    
    for cmd in "${commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            print_success "Found: $cmd"
        else
            print_error "Missing: $cmd"
            missing=$((missing + 1))
        fi
    done
    
    if [ $missing -gt 0 ]; then
        print_error "Some prerequisites are missing. Please install them."
        exit 1
    fi
}

# Build the project
build_project() {
    print_header "Building Game Engine"
    
    # Clean previous build
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    print_info "Configuring..."
    cmake "$PROJECT_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$BUNDLE_DIR/usr/local" \
        -DBUILD_TESTS=OFF
    
    print_info "Compiling..."
    cmake --build . -j$(nproc)
    
    print_success "Build completed"
}

# Install to bundle
install_to_bundle() {
    print_header "Installing to Bundle"
    
    cd "$BUILD_DIR"
    
    print_info "Installing files..."
    cmake --install . --prefix "$BUNDLE_DIR/usr/local"
    
    print_success "Installation completed"
}

# Collect dependencies
collect_dependencies() {
    print_header "Collecting Dependencies"
    
    local exe="$BUNDLE_DIR/usr/local/bin/GameEngine"
    local lib_dir="$BUNDLE_DIR/lib64"
    mkdir -p "$lib_dir"
    
    print_info "Analyzing dependencies..."
    local deps=$(ldd "$exe" | grep "=>" | awk '{print $3}' | grep -v "^$")
    
    local count=0
    while IFS= read -r lib; do
        if [ -f "$lib" ]; then
            local libname=$(basename "$lib")
            print_info "Copying: $libname"
            cp "$lib" "$lib_dir/" 2>/dev/null || true
            count=$((count + 1))
        fi
    done <<< "$deps"
    
    print_success "Collected $count dependencies"
}

# Create wrapper script
create_wrapper_script() {
    print_header "Creating Launcher Script"
    
    local wrapper="$BUNDLE_DIR/GameEngine"
    cat > "$wrapper" << 'EOF'
#!/bin/bash
# Game Engine Portable Launcher

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUNDLE_DIR="$SCRIPT_DIR"
export LD_LIBRARY_PATH="$BUNDLE_DIR/lib64:$LD_LIBRARY_PATH"

# Launch game engine
"$BUNDLE_DIR/usr/local/bin/GameEngine" "$@"
EOF
    
    chmod +x "$wrapper"
    print_success "Created launcher script"
}

# Create documentation
create_bundle_docs() {
    print_header "Creating Documentation"
    
    local readme="$BUNDLE_DIR/README.txt"
    cat > "$readme" << 'EOF'
Game Engine - Portable Bundle
==============================

This is a self-contained, portable build of the Game Engine.
All dependencies are included in the lib64 directory.

Requirements:
- Linux x86_64 system
- GLIBC compatible (most modern Linux distributions)

Running the Game Engine:
    ./GameEngine

You can move this directory anywhere on your system.
The launcher script automatically sets up the environment.

System Requirements:
- 4GB RAM minimum (8GB recommended)
- OpenGL 3.3+ capable graphics card
- 500MB free disk space

Troubleshooting:
- If you get "libGL.so.1 not found", install mesa-libGL
- For GLFW issues, install libxrandr-dev libxinerama-dev libxi-dev

Homepage: https://github.com/YOUR_USERNAME/game-engine
EOF
    
    chmod 644 "$readme"
    print_success "Created README"
}

# Create tarball
create_tarball() {
    print_header "Creating Distribution Tarball"
    
    cd "$BUILD_DIR"
    
    local tarball="GameEngine-${VERSION}-linux-x86_64-portable.tar.xz"
    print_info "Creating: $tarball"
    
    tar -cJf "$tarball" GameEngine-portable/
    
    print_success "Created: $tarball ($(du -h "$tarball" | cut -f1))"
    print_info "Location: $BUILD_DIR/$tarball"
}

# Generate checksums
generate_checksums() {
    print_header "Generating Checksums"
    
    cd "$BUILD_DIR"
    
    print_info "Computing SHA256..."
    sha256sum GameEngine-*.tar.xz > checksums.sha256
    
    print_info "Computing MD5..."
    md5sum GameEngine-*.tar.xz > checksums.md5
    
    print_success "Checksums generated"
}

# Summary
print_summary() {
    print_header "Build Summary"
    
    echo "Portable Bundle Information:"
    echo "  Version: $VERSION"
    echo "  Location: $BUNDLE_DIR"
    echo "  Tarball: $BUILD_DIR/GameEngine-${VERSION}-linux-x86_64-portable.tar.xz"
    echo "  Size: $(du -sh "$BUNDLE_DIR" | cut -f1)"
    echo ""
    echo "To use:"
    echo "  1. Extract: tar -xJf GameEngine-${VERSION}-linux-x86_64-portable.tar.xz"
    echo "  2. Run: ./GameEngine-portable/GameEngine"
    echo ""
    echo "Distribution ready at: $BUILD_DIR"
}

# Main execution
main() {
    print_header "Game Engine Portable Bundle Creator"
    
    check_prerequisites
    build_project
    install_to_bundle
    collect_dependencies
    create_wrapper_script
    create_bundle_docs
    create_tarball
    generate_checksums
    print_summary
    
    print_success "Portable bundle created successfully!"
}

main "$@"
