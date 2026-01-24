// GO_BUILD_GUIDE.md - Building Go Scripts for the Game Engine

## Overview

This guide explains how to build Go scripts for use with the Game Engine's Go runtime.

## Prerequisites

1. **Go 1.21+** installed
2. **C compiler** (gcc, clang, or MSVC)
3. **CGO enabled** (default in most Go installations)

## Building Go Scripts

### Quick Start

```bash
# Build a simple Go script for Windows
go build -o scripts/npc_ai.dll -buildmode=c-shared scripts/npc_ai.go

# Build for Linux
go build -o scripts/npc_ai.so -buildmode=c-shared scripts/npc_ai.go

# Build for macOS
go build -o scripts/npc_ai.dylib -buildmode=c-shared scripts/npc_ai.go
```

### Project Structure

```
game-engine/
├── scripts/
│   ├── go.mod                    # Go module definition
│   ├── example_go_systems.go     # Example implementations
│   ├── npc_behavior.go           # NPC AI logic
│   ├── physics_system.go         # Physics calculations
│   ├── network_sync.go           # Network replication
│   └── build_scripts.sh          # Build script
├── game-engine-go/               # Go package (optional)
│   ├── bindings/                 # C++ bindings
│   └── gamelogic/                # Game logic packages
```

### Building Individual Scripts

Each `.go` file can be compiled separately:

```bash
# Windows
go build -o npc_ai.dll -buildmode=c-shared npc_ai.go

# Linux
go build -o npc_ai.so -buildmode=c-shared npc_ai.go

# macOS
go build -o npc_ai.dylib -buildmode=c-shared npc_ai.go
```

### Building Multiple Scripts (Batch)

Create `build_scripts.sh`:

```bash
#!/bin/bash

# Build all Go scripts
OUTPUT_DIR="build"
mkdir -p "$OUTPUT_DIR"

# Detect OS
if [[ "$OSTYPE" == "win32" || "$OSTYPE" == "msys" ]]; then
    EXT="dll"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    EXT="so"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    EXT="dylib"
fi

# Build scripts
for script in *.go; do
    if [[ "$script" != "go_bindings.go" ]]; then
        name="${script%.go}"
        echo "Building $name..."
        go build -o "$OUTPUT_DIR/$name.$EXT" -buildmode=c-shared "$script"
    fi
done

echo "Build complete!"
```

### Adding C++ Bindings (CGO)

For more advanced scripts that need to call C++ functions:

```go
package main

/*
#cgo LDFLAGS: -L../build -lengine_core
#include "engine_core.h"

// Engine API stubs
void EngineUpdateFrame();
void EngineLogMessage(const char* msg);
*/
import "C"
import "fmt"

func InitializeEngine() {
    fmt.Println("Initializing engine...")
    // C.EngineLogMessage(C.CString("Go script initialized"))
}

//export UpdateGameFrame
func UpdateGameFrame(deltaTime float32) {
    fmt.Printf("Frame update: dt=%.3f\n", deltaTime)
    // C.EngineUpdateFrame()
}
```

Build with:
```bash
go build -o engine_core.dll -buildmode=c-shared game_core.go
```

## Compilation Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `-buildmode=c-shared` | Build as shared library | Required for dll/so/dylib |
| `-o <output>` | Output filename | `-o npc_ai.dll` |
| `-trimpath` | Remove paths for reproducible builds | Good for distribution |
| `-ldflags` | Linker flags | `'-ldflags=-s -w'` removes symbols |

### Optimization Flags

```bash
# Release build (optimized, no debug info)
go build -trimpath -ldflags="-s -w" -o npc_ai.dll -buildmode=c-shared npc_ai.go

# Debug build (with symbols, slower)
go build -o npc_ai_debug.dll -buildmode=c-shared npc_ai.go

# With profiling support
go build -tags profile -o npc_ai_profile.dll -buildmode=c-shared npc_ai.go
```

## Environment Variables

```bash
# Force specific Go version
GOVERSION=go1.21

# Cross-compilation
GOOS=windows GOARCH=amd64 go build -o script.dll -buildmode=c-shared script.go
GOOS=linux GOARCH=amd64 go build -o script.so -buildmode=c-shared script.go

# Debug logging
GODEBUG=gctrace=1 go run script.go  # Show GC traces
```

## Dependency Management

### Using go.mod

```toml
module game-engine-go

go 1.21

require (
    // Add external dependencies
)
```

### Installing Dependencies

```bash
cd scripts
go mod tidy          # Clean up go.mod
go mod download      # Download all dependencies
go mod vendor        # Create vendor directory
```

## Testing

### Running Tests

```bash
go test ./...        # Run all tests
go test -v ./...     # Verbose output
go test -cover ./... # Coverage report
```

### Example Test File

```go
// example_test.go
package main

import "testing"

func TestNPCBehavior(t *testing.T) {
    npcID := int32(1)
    // Test implementation
}

func BenchmarkPhysicsUpdate(b *testing.B) {
    for i := 0; i < b.N; i++ {
        ExampleParallelPhysicsUpdate(0.016, 100)
    }
}
```

## Troubleshooting

### Issue: "gcc not found"

**Solution**: Install a C compiler
- **Windows**: Install [MinGW](https://www.mingw-w64.org/) or use MSVC
- **Linux**: `sudo apt-get install build-essential`
- **macOS**: `xcode-select --install`

### Issue: CGO disabled

**Solution**: Enable CGO
```bash
CGO_ENABLED=1 go build -o script.dll -buildmode=c-shared script.go
```

### Issue: Symbol not found when loading .dll

**Solution**: Check export visibility
```go
// Must be capitalized to be exported
func ExportedFunc() { }  // ✓ Exported
func unexportedFunc() { } // ✗ Not exported
```

### Issue: Goroutine leaks

**Solution**: Ensure all goroutines terminate
```go
func SafeGoroutine() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Printf("Panic: %v\n", r)
        }
    }()
    
    // Work here
}
```

## Integration with C++

### Loading .go Scripts

```cpp
auto& registry = ScriptLanguageRegistry::GetInstance();

// Auto-detect .go extension
registry.ExecuteScript("scripts/example_go_systems.go");
```

### Calling Go Functions

```cpp
auto goSystem = dynamic_cast<GoScriptSystem*>(
    registry.GetScriptSystem(ScriptLanguage::Go)
);

// Call synchronously
std::vector<std::any> args = {100, 10};
goSystem->CallFunction("ExampleNPCBehavior", args);

// Or spawn as goroutine
int gid = goSystem->StartGoroutine("ExampleParallelPhysicsUpdate");
```

## Performance Tips

1. **Profile your Go code**:
   ```bash
   go test -cpuprofile=cpu.prof -memprofile=mem.prof -bench=.
   go tool pprof cpu.prof
   ```

2. **Use buffered channels** for high throughput
3. **Avoid goroutine explosion** - use worker pools
4. **Pre-allocate slices** when size is known
5. **Monitor memory** with `runtime.MemStats`

## Distribution

### Creating Release Builds

```bash
#!/bin/bash
mkdir -p release

# Windows
GOOS=windows GOARCH=amd64 go build -o release/game_scripts.dll -buildmode=c-shared *.go

# Linux
GOOS=linux GOARCH=amd64 go build -o release/game_scripts.so -buildmode=c-shared *.go

# macOS
GOOS=darwin GOARCH=amd64 go build -o release/game_scripts.dylib -buildmode=c-shared *.go
```

### Packaging

```bash
zip -r game-scripts-windows.zip release/*.dll
tar.gz -r game-scripts-linux.tar.gz release/*.so
```

## Next Steps

1. Write your game logic in Go
2. Build scripts to .dll/.so/.dylib
3. Load in engine via ScriptLanguageRegistry
4. Monitor performance with profiling
5. Optimize bottlenecks with profiling data

See GO_LANGUAGE_GUIDE.md for more examples!
