#!/bin/bash
# build_go_scripts.sh - Build Go scripts for Linux/macOS

set -e

# Detect OS
if [[ "$OSTYPE" == "win32" || "$OSTYPE" == "msys" ]]; then
    EXT="dll"
    SHARED_FLAG="-buildmode=c-shared"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    EXT="so"
    SHARED_FLAG="-buildmode=c-shared"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    EXT="dylib"
    SHARED_FLAG="-buildmode=c-shared"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Check for Go
if ! command -v go &> /dev/null; then
    echo "Go not found in PATH. Please install Go 1.21+"
    exit 1
fi

# Create output directory
OUTPUT_DIR="scripts/build"
mkdir -p "$OUTPUT_DIR"

echo "Building Go scripts for $OSTYPE (extension: $EXT)..."

# Build each .go file
for script in scripts/*.go; do
    if [[ "$script" != "scripts/example_go_systems.go" ]]; then
        name=$(basename "$script" .go)
        echo "Building $name..."
        go build -o "$OUTPUT_DIR/$name.$EXT" $SHARED_FLAG "$script"
        if [ $? -ne 0 ]; then
            echo "Failed to build $name"
            exit 1
        fi
    fi
done

# Build example systems
echo "Building example_go_systems..."
go build -o "$OUTPUT_DIR/example_go_systems.$EXT" $SHARED_FLAG scripts/example_go_systems.go
if [ $? -ne 0 ]; then
    echo "Failed to build example_go_systems"
    exit 1
fi

echo ""
echo "Go scripts built successfully to $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR"/*.$EXT
