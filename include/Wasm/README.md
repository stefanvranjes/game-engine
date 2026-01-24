# WASM Subsystem README

## Overview

The WASM (WebAssembly) subsystem provides a complete runtime for loading, executing, and managing WebAssembly modules within the game engine. This allows you to write high-performance game scripts in Rust, C/C++, AssemblyScript, and other WASM-compatible languages.

## Features

- **🔧 Complete WASM Runtime** - wasm3-based execution environment
- **📦 Module Management** - Load, unload, and introspect WASM modules
- **🎮 Engine Integration** - Seamless integration with physics, audio, rendering
- **🔒 Memory Safety** - Bounds-checked memory access with protection
- **⚡ Performance** - Profiling and optimization support
- **🔄 Hot-Reload** - Automatic module reloading during development
- **📝 Type-Safe** - C++20 templates for compile-time type safety
- **🧵 Multi-Language** - Support for Rust, C, C++, AssemblyScript, Go, etc.

## Quick Start

### 1. Load a WASM Module

```cpp
#include "Wasm/WasmScriptSystem.h"

WasmScriptSystem& wasmSys = WasmScriptSystem::GetInstance();
wasmSys.Init();

// Load from file
wasmSys.LoadWasmModule("scripts/game_logic.wasm", "game_logic");
```

### 2. Call WASM Functions

```cpp
auto instance = wasmSys.GetModuleInstance("game_logic");

// Call with no arguments
instance->Call("init");

// Call with arguments
instance->Call("update", {WasmValue::F32(deltaTime)});

// Type-safe call
instance->CallTyped("doSomething", 42, 3.14f);
```

### 3. Access Memory Safely

```cpp
// Read WASM memory
auto bytes = instance->ReadMemory(0, 16);

// Write to WASM memory
std::vector<uint8_t> data = {1, 2, 3, 4};
instance->WriteMemory(0, data);

// String operations
instance->WriteString(128, "Hello from C++");
std::string message = instance->ReadString(64);
```

### 4. Profile Performance

```cpp
instance->SetProfilingEnabled(true);

// ... run some frames ...

auto profilingData = instance->GetProfilingData();
for (const auto& data : profilingData) {
    std::cout << data.functionName << ": " 
              << data.averageTime << "ms" << std::endl;
}
```

## File Organization

```
include/Wasm/
├── WasmRuntime.h          # Core runtime (module loading, memory)
├── WasmModule.h           # Module representation (exports, validation)
├── WasmInstance.h         # Instance execution (function calls, memory)
├── WasmScriptSystem.h     # IScriptSystem integration
├── WasmEngineBindings.h   # Engine function exposure
└── WasmHelper.h           # Utility functions

src/Wasm/
├── WasmRuntime.cpp        # Implementation
├── WasmModule.cpp         # Implementation
├── WasmInstance.cpp       # Implementation
├── WasmScriptSystem.cpp   # Implementation
├── WasmEngineBindings.cpp # Implementation
└── WasmHelper.cpp         # Implementation

tests/
└── WasmTest.cpp           # Unit tests

cmake/
└── WasmSupport.cmake      # CMake configuration

Documentation/
├── WASM_SUPPORT_GUIDE.md           # Complete reference
├── WASM_EXAMPLES.md                # Code examples
├── WASM_TOOLING_GUIDE.md           # Build instructions
├── WASM_SUPPORT_INDEX.md           # Feature index
└── WASM_IMPLEMENTATION_SUMMARY.md  # This summary
```

## Architecture

### Component Hierarchy

```
WasmScriptSystem
  └─ WasmRuntime
      ├─ WasmModule
      │   └─ WasmInstance (multiple)
      │       └─ Memory Management
      └─ WasmEngineBindings
          ├─ Physics Bindings
          ├─ Audio Bindings
          ├─ Rendering Bindings
          ├─ Input Bindings
          └─ Debug Bindings
```

### Data Flow

```
C++ Code
   ↓
WasmScriptSystem
   ↓
WasmInstance (executes function)
   ↓
WASM Memory
   ↓
Engine Bindings (call back to engine)
   ↓
Engine Systems (Physics, Audio, Rendering)
   ↓
Results (returned to WASM and C++)
```

## Core Classes

### WasmRuntime
- Manages WASM environment
- Handles module loading
- Controls memory limits
- Enforces execution timeouts

**Key Methods:**
- `Initialize()` - Set up runtime
- `LoadModule(filepath)` - Load WASM file
- `GetModule(name)` - Access loaded module
- `UnloadModule(name)` - Remove module

### WasmModule
- Represents compiled WASM binary
- Introspects exports
- Validates structure
- Creates instances

**Key Methods:**
- `GetExportedFunctions()` - List functions
- `HasExportedFunction(name)` - Check export
- `CreateInstance()` - New instance
- `Validate()` - Check structure

### WasmInstance
- Executes WASM code
- Manages linear memory
- Tracks execution state
- Provides profiling

**Key Methods:**
- `Call(name)` - Execute function
- `CallTyped(name, args...)` - Type-safe call
- `ReadMemory(offset, size)` - Safe read
- `WriteMemory(offset, data)` - Safe write
- `ReadString(offset)` - Read C string
- `WriteString(offset, str)` - Write C string
- `Malloc(size)` - Allocate heap
- `Free(offset)` - Deallocate heap

### WasmScriptSystem
- Implements IScriptSystem
- Manages instances
- Supports lifecycle hooks
- Hot-reload capability

**Key Methods:**
- `LoadWasmModule(path)` - Load module
- `CallWasmFunction(module, func)` - Call function
- `BindGameObject(module, obj)` - Bind object
- `EnableHotReload(enabled)` - Enable reloading
- `GetModuleInstance(name)` - Access instance

### WasmEngineBindings
- Exposes engine to WASM
- Physics, audio, rendering functions
- Extensible binding system
- Debug utilities

**Binding Categories:**
- Physics (force, velocity, ray casting)
- Audio (play, stop, volume)
- Rendering (color, materials, shadows)
- Input (key/mouse state)
- Debug (logging, drawing)

## Writing WASM Modules

### Rust Example

```rust
#[no_mangle]
pub extern "C" fn init() {
    // Called once at module load
}

#[no_mangle]
pub extern "C" fn update(delta_time: f32) {
    // Called every frame
}

#[no_mangle]
pub extern "C" fn shutdown() {
    // Called at module unload
}
```

Compile:
```bash
rustc --target wasm32-unknown-unknown -O script.rs --crate-type cdylib
```

### C Example

```c
void init(void) {
    // Called once at module load
}

void update(float delta_time) {
    // Called every frame
}

void shutdown(void) {
    // Called at module unload
}
```

Compile:
```bash
clang --target=wasm32 -O3 -nostdlib \
  -Wl,--no-entry \
  -Wl,--export=init \
  -Wl,--export=update \
  -Wl,--export=shutdown \
  script.c -o script.wasm
```

### AssemblyScript Example

```typescript
export function init(): void {
    // Called once at module load
}

export function update(deltaTime: f32): void {
    // Called every frame
}

export function shutdown(): void {
    // Called at module unload
}
```

Compile:
```bash
asc script.ts -O3 -o script.wasm
```

## Binding Engine Functions to WASM

### Pre-built Bindings

```cpp
WasmEngineBindings& bindings = WasmEngineBindings::GetInstance();
bindings.RegisterBindings(instance);

// Now these functions are available in WASM:
// - physics_apply_force()
// - audio_play()
// - render_set_color()
// - input_is_key_pressed()
// - debug_log()
```

### Custom Bindings

```cpp
bindings.RegisterBinding(instance, "my_function",
    [](std::shared_ptr<WasmInstance> inst, const std::vector<WasmValue>& args) {
        // Read arguments from WASM
        if (args.size() > 0) {
            auto value = std::any_cast<int32_t>(args[0].value);
            // Do something...
        }
        
        // Return value to WASM
        return WasmValue::I32(42);
    }
);
```

## Memory Management

### Safe Memory Access

```cpp
auto instance = wasmSys.GetModuleInstance("my_script");

// Get memory size
size_t size = instance->GetMemorySize();  // Default: 256 KB

// Safe read
auto bytes = instance->ReadMemory(0, 16);
if (!bytes.empty()) {
    // Process bytes...
}

// Safe write
std::vector<uint8_t> data = {1, 2, 3, 4};
bool success = instance->WriteMemory(0, data);

// String read/write
instance->WriteString(128, "Hello World");
std::string message = instance->ReadString(128);
```

### Heap Allocation

```cpp
// Allocate WASM heap memory
uint32_t ptr = instance->Malloc(256);

if (ptr != 0) {
    // Write to allocated memory
    std::vector<uint8_t> data = {/* ... */};
    instance->WriteMemory(ptr, data);
    
    // Use allocated memory...
    
    // Free when done
    instance->Free(ptr);
}
```

### Memory Statistics

```cpp
auto stats = instance->GetStats();
std::cout << "Total memory: " << stats.totalMemory << std::endl;
std::cout << "Used memory: " << stats.usedMemory << std::endl;
std::cout << "Call depth: " << stats.callDepth << std::endl;
```

## Error Handling

### Check for Errors

```cpp
auto instance = wasmSys.GetModuleInstance("my_script");

instance->Call("do_something");

// Check for errors
if (!instance->GetLastError().empty()) {
    std::cerr << "Error: " << instance->GetLastError() << std::endl;
}
```

### Runtime Errors

```cpp
auto& runtime = WasmRuntime::GetInstance();
runtime.ClearLastError();

// Load module...

if (!runtime.GetLastError().empty()) {
    std::cerr << "Runtime error: " << runtime.GetLastError() << std::endl;
}
```

## Performance Optimization

### Enable Profiling

```cpp
instance->SetProfilingEnabled(true);

// Run some code...

auto profilingData = instance->GetProfilingData();
for (const auto& data : profilingData) {
    std::cout << data.functionName << ":\n"
              << "  Calls: " << data.callCount << "\n"
              << "  Total: " << data.totalTime << "ms\n"
              << "  Avg: " << data.averageTime << "ms\n"
              << "  Min: " << data.minTime << "ms\n"
              << "  Max: " << data.maxTime << "ms\n";
}
```

### Set Memory Limits

```cpp
WasmRuntime::GetInstance().SetMaxMemorySize(512);  // 512 MB

WasmScriptSystem::GetInstance().SetDefaultMemoryLimit(256);  // 256 MB
```

### Execution Timeout

```cpp
WasmRuntime::GetInstance().SetExecutionTimeout(true, 5000);  // 5 seconds
```

## Hot-Reload

### Enable During Development

```cpp
WasmScriptSystem& wasmSys = WasmScriptSystem::GetInstance();
wasmSys.EnableHotReload(true);

// Modules will automatically reload when files change
```

## Integration with Other Systems

### With Other Script Systems

```cpp
// All share IScriptSystem interface
LuaScriptSystem::GetInstance().RunScript("logic.lua");
PythonScriptSystem::GetInstance().RunScript("ai.py");
WasmScriptSystem::GetInstance().RunScript("critical.wasm");
```

### With ECS

```cpp
auto entity = entityManager->CreateEntity();
auto scriptComp = entity->AddComponent<ScriptComponent>();
scriptComp->SetWasmModule("my_script.wasm");
```

### With GameObject System

```cpp
auto gameObj = std::make_shared<GameObject>("enemy");
wasmSys.BindGameObject("enemy_ai", gameObj);
```

## Documentation

- **[WASM_SUPPORT_GUIDE.md](WASM_SUPPORT_GUIDE.md)** - Complete reference and usage guide
- **[WASM_EXAMPLES.md](WASM_EXAMPLES.md)** - Real-world code examples
- **[WASM_TOOLING_GUIDE.md](WASM_TOOLING_GUIDE.md)** - Build and compilation instructions
- **[WASM_SUPPORT_INDEX.md](WASM_SUPPORT_INDEX.md)** - Feature inventory
- **[WASM_IMPLEMENTATION_SUMMARY.md](WASM_IMPLEMENTATION_SUMMARY.md)** - Implementation overview

## Troubleshooting

### Module Won't Load

```cpp
auto& runtime = WasmRuntime::GetInstance();
std::cout << runtime.GetLastError() << std::endl;
```

Common causes:
- Invalid WASM magic number
- Corrupted binary
- Unsupported WASM version
- Missing required sections

### Function Not Found

```bash
# Check module exports
wasm-objdump -x module.wasm | grep Export
```

### Memory Access Errors

- Use `ReadMemory()` / `WriteMemory()` for safety
- Check bounds with `GetMemorySize()`
- Enable `SetMemoryProtection(true)`

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Speed | 2-10x slower than native C++ |
| Memory | 256 MB default per module |
| Load time | <100ms typical |
| Function call overhead | ~1-5 μs |
| Binary size | 10KB - 1MB typical |

## Next Steps

1. Read [WASM_SUPPORT_GUIDE.md](WASM_SUPPORT_GUIDE.md) for detailed API reference
2. Check [WASM_EXAMPLES.md](WASM_EXAMPLES.md) for code samples
3. Follow [WASM_TOOLING_GUIDE.md](WASM_TOOLING_GUIDE.md) for compilation
4. Start writing WASM modules in your preferred language
5. Integrate with your game using the examples provided

## Support

For issues or questions:
1. Check the troubleshooting section in WASM_SUPPORT_GUIDE.md
2. Review code examples in WASM_EXAMPLES.md
3. Enable profiling and debug output
4. Validate modules with wasm-validate

