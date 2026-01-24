# WASM Tooling & Build Guide

## Compiling to WASM

### Prerequisites

#### Rust
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add wasm32 target
rustup target add wasm32-unknown-unknown
```

#### C/C++
```bash
# Ubuntu/Debian
sudo apt-get install clang lld

# macOS
brew install clang

# Windows (MSVC)
# Use clang from LLVM project or install separately
```

#### AssemblyScript
```bash
npm install -g @assemblyscript/loader
npm install --save-dev assemblyscript
```

### Compilation Examples

#### Rust → WASM

**Simple module:**
```bash
rustc --target wasm32-unknown-unknown -O script.rs --crate-type cdylib -o script.wasm
```

**With Cargo:**
```toml
[package]
name = "game_script"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Enable LTO
```

```bash
cargo build --target wasm32-unknown-unknown --release
# Output: target/wasm32-unknown-unknown/release/game_script.wasm
```

#### C → WASM

**Basic compilation:**
```bash
clang --target=wasm32 -O3 -nostdlib \
  -Wl,--no-entry \
  -Wl,--export=init \
  -Wl,--export=update \
  -Wl,--export=shutdown \
  script.c -o script.wasm
```

**With linker script (for more control):**
```bash
clang --target=wasm32 -O3 -nostdlib script.c -c -o script.o
wasm-ld script.o -o script.wasm \
  --no-entry \
  --export=init \
  --export=update \
  --export=shutdown \
  --allow-undefined
```

#### C++ → WASM

```bash
clang++ --target=wasm32 -O3 -nostdlib \
  -Wl,--no-entry \
  -fno-exceptions \
  -fno-rtti \
  script.cpp -o script.wasm
```

#### AssemblyScript → WASM

```bash
# Create project
npm init
npm install --save-dev assemblyscript

# Create script.ts
cat > script.ts << 'EOF'
export function init(): void {}
export function update(deltaTime: f32): void {}
export function shutdown(): void {}
EOF

# Compile
npx asc script.ts -O3 -o script.wasm
```

### Optimization Tips

#### Size Optimization
```bash
# Rust
rustc --target wasm32-unknown-unknown -O script.rs \
  --crate-type cdylib \
  -C opt-level=z \
  -C lto=fat \
  --edition 2021

# Then strip
wasm-opt -Oz script.wasm -o script.wasm

# Or use binaryen tools
wasm-opt -O4 -oz script.wasm -o script.wasm
```

#### Performance Optimization
```bash
# Rust - optimize for speed
rustc --target wasm32-unknown-unknown \
  -C opt-level=3 \
  -C target-cpu=generic \
  script.rs --crate-type cdylib

# C - aggressive optimization
clang --target=wasm32 -O3 -march=wasm32 \
  -ffast-math \
  script.c -o script.wasm
```

### Module Inspection Tools

#### wasm-objdump
```bash
# List exports
wasm-objdump -x script.wasm

# Dump disassembly
wasm-objdump -d script.wasm

# Export specific section
wasm-objdump --headers script.wasm
```

#### wasmtime (CLI execution)
```bash
# Install
curl https://wasmtime.dev/install.sh -sSf | bash

# Run WASM module directly
wasmtime script.wasm

# Call specific export
wasmtime --invoke init script.wasm
```

#### wasm2c
```bash
# Convert WASM to C
wasm2c script.wasm -o script.c

# Compile the generated C
gcc -O3 script.c -c -o script.o
```

## Build Automation

### CMake Integration

**In your project CMakeLists.txt:**

```cmake
# Find Rust toolchain
find_program(RUSTC rustc REQUIRED)
find_program(CARGO cargo REQUIRED)

# Function to build Rust to WASM
function(build_rust_wasm TARGET_NAME SOURCE_DIR)
    add_custom_target(${TARGET_NAME}
        COMMAND ${CARGO} build 
            --target wasm32-unknown-unknown 
            --release
            --manifest-path ${SOURCE_DIR}/Cargo.toml
        WORKING_DIRECTORY ${SOURCE_DIR}
        COMMENT "Building ${TARGET_NAME} WASM module"
    )
endfunction()

# Function to build C to WASM
function(build_c_wasm TARGET_NAME SOURCE_FILE EXPORTS)
    add_custom_command(
        OUTPUT ${TARGET_NAME}.wasm
        COMMAND ${CMAKE_C_COMPILER}
            --target=wasm32 -O3 -nostdlib
            ${SOURCE_FILE}
            -o ${TARGET_NAME}.wasm
            ${EXPORTS}
        DEPENDS ${SOURCE_FILE}
        COMMENT "Building ${TARGET_NAME} WASM module"
    )
endfunction()
```

### Build Script (Windows batch)

**build_wasm.bat:**
```batch
@echo off
REM Build all WASM modules

setlocal enabledelayedexpansion

set WASM_DIR=wasm_modules
set OUTPUT_DIR=assets/wasm

if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Build Rust modules
for /d %%D in (%WASM_DIR%/rust/*) do (
    echo Building %%D...
    cd %%D
    cargo build --target wasm32-unknown-unknown --release
    if exist target\wasm32-unknown-unknown\release\*.wasm (
        copy target\wasm32-unknown-unknown\release\*.wasm ..\..\%OUTPUT_DIR%\
    )
    cd ..\..
)

REM Build C modules
for %%F in (%WASM_DIR%/c/*.c) do (
    echo Building %%F...
    clang --target=wasm32 -O3 -nostdlib %%F ^
        -Wl,--no-entry ^
        -Wl,--export=init ^
        -Wl,--export=update ^
        -Wl,--export=shutdown ^
        -o %OUTPUT_DIR%/%%~nF.wasm
)

echo WASM build complete!
```

### Build Script (Linux/macOS)

**build_wasm.sh:**
```bash
#!/bin/bash

WASM_DIR="wasm_modules"
OUTPUT_DIR="assets/wasm"

mkdir -p "$OUTPUT_DIR"

# Build Rust modules
for dir in $WASM_DIR/rust/*/; do
    [ -d "$dir" ] || continue
    echo "Building $(basename $dir)..."
    cd "$dir"
    cargo build --target wasm32-unknown-unknown --release
    find target/wasm32-unknown-unknown/release -name "*.wasm" -exec cp {} ../../$OUTPUT_DIR/ \;
    cd ../..
done

# Build C modules
for file in $WASM_DIR/c/*.c; do
    [ -f "$file" ] || continue
    echo "Building $file..."
    clang --target=wasm32 -O3 -nostdlib "$file" \
        -Wl,--no-entry \
        -Wl,--export=init \
        -Wl,--export=update \
        -Wl,--export=shutdown \
        -o "$OUTPUT_DIR/$(basename $file .c).wasm"
done

echo "WASM build complete!"
```

## Development Workflow

### 1. Create WASM Module Directory

```
game-engine/
├── wasm_modules/
│   ├── rust/
│   │   └── game_logic/
│   │       ├── Cargo.toml
│   │       └── src/lib.rs
│   ├── c/
│   │   └── physics.c
│   └── scripts/
│       └── script.ts
└── assets/
    └── wasm/
        ├── game_logic.wasm
        ├── physics.wasm
        └── script.wasm
```

### 2. Develop Module

**Rust example (game_logic/src/lib.rs):**
```rust
#[no_mangle]
pub extern "C" fn init() {
    // Init code
}

#[no_mangle]
pub extern "C" fn update(delta: f32) {
    // Update logic
}

#[no_mangle]
pub extern "C" fn shutdown() {
    // Cleanup
}
```

### 3. Build

```bash
# Build all modules
./build_wasm.sh

# Or build specific module
cd wasm_modules/rust/game_logic
cargo build --target wasm32-unknown-unknown --release
cp target/wasm32-unknown-unknown/release/*.wasm ../../../assets/wasm/
```

### 4. Load in Engine

```cpp
WasmScriptSystem& wasmSys = WasmScriptSystem::GetInstance();
wasmSys.LoadWasmModule("assets/wasm/game_logic.wasm");

auto instance = wasmSys.GetModuleInstance("game_logic");
instance->Call("init");
instance->Call("update", {WasmValue::F32(deltaTime)});
```

### 5. Debug

```bash
# Inspect module
wasm-objdump -x assets/wasm/game_logic.wasm

# Disassemble
wasm-objdump -d assets/wasm/game_logic.wasm | head -50

# Validate
wasm-validate assets/wasm/game_logic.wasm
```

## Performance Profiling

### Enable Profiling in Engine

```cpp
auto instance = wasmSys.GetModuleInstance("game_logic");
instance->SetProfilingEnabled(true);

// Run some frames
for (int i = 0; i < 100; ++i) {
    instance->Call("update", {WasmValue::F32(0.016f)});
}

// Print results
auto data = instance->GetProfilingData();
for (const auto& d : data) {
    std::cout << d.functionName << ":\n"
              << "  Calls: " << d.callCount << "\n"
              << "  Total: " << d.totalTime << "ms\n"
              << "  Avg: " << d.averageTime << "ms\n"
              << "  Min: " << d.minTime << "ms\n"
              << "  Max: " << d.maxTime << "ms\n";
}
```

### WASM Code Profiling (Native)

```bash
# Use wasmtime with profiling
wasmtime --profile=perfmap script.wasm > perfmap.txt
perf report -i perf.data

# Or use native profiling (Rust)
perf record -g ./target/debug/myapp
perf report
```

## Testing WASM Modules

### Unit Testing (Rust)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        init();
        // Assert state
    }

    #[test]
    fn test_update() {
        init();
        update(0.016);
        update(0.016);
        // Assert results
    }
}
```

Compile test:
```bash
rustc --test script.rs -o script_test && ./script_test
```

### Integration Testing (C++)

Use Google Test with WASM modules:

```cpp
TEST(WasmIntegrationTest, GameLogic) {
    WasmScriptSystem& wasmSys = WasmScriptSystem::GetInstance();
    wasmSys.LoadWasmModule("assets/wasm/game_logic.wasm");
    
    auto instance = wasmSys.GetModuleInstance("game_logic");
    ASSERT_NE(instance, nullptr);
    
    instance->Call("init");
    instance->Call("update", {WasmValue::F32(0.016f)});
    
    // Verify results
    EXPECT_FALSE(instance->GetLastError().empty() == false);
}
```

## Troubleshooting Build Issues

### Module Won't Load

```cpp
auto& runtime = WasmRuntime::GetInstance();
std::cout << "Error: " << runtime.GetLastError() << std::endl;
```

Common issues:
- Missing magic number (0x6d736100)
- Incorrect version byte
- Corrupted binary
- Missing required exports

### Function Not Found

```bash
# Check exports
wasm-objdump -x script.wasm | grep Export

# Verify function name matches exactly
```

### Memory Issues

```cpp
std::cout << "Memory size: " << instance->GetMemorySize() << std::endl;
auto stats = instance->GetStats();
std::cout << "Used: " << stats.usedMemory << "/" << stats.totalMemory << std::endl;
```

### Performance Problems

```bash
# Profile WASM execution
wasm-opt -O4 script.wasm -o script_opt.wasm

# Check binary size
ls -lh script.wasm

# Analyze functions
wasm-objdump --details script.wasm
```

