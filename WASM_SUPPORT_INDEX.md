# WebAssembly (WASM) Support Implementation Index

## Overview
Complete WebAssembly support has been added to the game engine, allowing you to write game scripts and plugins in compiled WASM modules. This enables better performance and interoperability with multiple languages (Rust, C, AssemblyScript, etc.).

## Files Added

### Core WASM System

| File | Purpose |
|------|---------|
| [include/Wasm/WasmRuntime.h](include/Wasm/WasmRuntime.h) | WASM runtime manager, module loading |
| [include/Wasm/WasmModule.h](include/Wasm/WasmModule.h) | WASM module representation and introspection |
| [include/Wasm/WasmInstance.h](include/Wasm/WasmInstance.h) | WASM instance execution and memory management |
| [include/Wasm/WasmScriptSystem.h](include/Wasm/WasmScriptSystem.h) | IScriptSystem integration for WASM |
| [include/Wasm/WasmEngineBindings.h](include/Wasm/WasmEngineBindings.h) | Engine function bridges for WASM |
| [include/Wasm/WasmHelper.h](include/Wasm/WasmHelper.h) | Utility functions for WASM integration |
| [src/Wasm/WasmRuntime.cpp](src/Wasm/WasmRuntime.cpp) | Runtime implementation |
| [src/Wasm/WasmModule.cpp](src/Wasm/WasmModule.cpp) | Module implementation |
| [src/Wasm/WasmInstance.cpp](src/Wasm/WasmInstance.cpp) | Instance implementation |
| [src/Wasm/WasmScriptSystem.cpp](src/Wasm/WasmScriptSystem.cpp) | Script system implementation |
| [src/Wasm/WasmEngineBindings.cpp](src/Wasm/WasmEngineBindings.cpp) | Engine bindings implementation |
| [src/Wasm/WasmHelper.cpp](src/Wasm/WasmHelper.cpp) | Helper utilities implementation |

### Documentation

| File | Purpose |
|------|---------|
| [WASM_SUPPORT_GUIDE.md](WASM_SUPPORT_GUIDE.md) | Complete usage guide and reference |
| [WASM_EXAMPLES.md](WASM_EXAMPLES.md) | Practical examples in Rust, C, and AssemblyScript |
| [WASM_SUPPORT_INDEX.md](WASM_SUPPORT_INDEX.md) | This file |

## Key Features

### ✅ Implemented
- **WASM Module Loading** - Load compiled .wasm files
- **Multiple Instances** - Create multiple instances from one module
- **Safe Memory Access** - Bounds-checked reads/writes
- **Type-Safe Calls** - Variadic templates for function calls
- **Lifecycle Hooks** - init(), update(), shutdown() support
- **Engine Bindings** - Call engine functions from WASM
- **Memory Protection** - Optional bounds checking
- **Profiling** - Per-function execution time tracking
- **Error Handling** - Comprehensive error reporting
- **Hot-Reload** - File change detection (optional)
- **Script System Integration** - Works with IScriptSystem interface

### Architecture Diagrams

```
┌─────────────────────────────────────────────────────────────┐
│                      Application                             │
├─────────────────────────────────────────────────────────────┤
│  WasmScriptSystem (Implements IScriptSystem)                │
├─────────────────────────────────────────────────────────────┤
│        ┌──────────────────────────────────────┐             │
│        │       WasmRuntime                    │             │
│        │  (Manages environment, modules)      │             │
│        └──────────────────────────────────────┘             │
│          ↓                                                   │
│        ┌──────────────────────────────────────┐             │
│        │    WasmModule  WasmModule ...        │             │
│        │  (Compiled binaries)                 │             │
│        └──────────────────────────────────────┘             │
│          ↓                                                   │
│        ┌──────────────────────────────────────┐             │
│        │  Instance  Instance  Instance        │             │
│        │ (Execution contexts)                 │             │
│        └──────────────────────────────────────┘             │
│          ↓          ↓          ↓                            │
│        ┌────────────────────────────────────┐              │
│        │    WasmEngineBindings               │              │
│        │  (Physics, Audio, Rendering, etc)   │              │
│        └────────────────────────────────────┘              │
│          ↓                                                   │
│        Engine Systems (Physics, Audio, Rendering, etc)     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Enable WASM in CMakeLists.txt
```cmake
option(ENABLE_WASM_SUPPORT "Enable WebAssembly support" ON)
```

### 2. Initialize in Application
```cpp
#include "Wasm/WasmScriptSystem.h"

WasmScriptSystem::GetInstance().Init();
```

### 3. Load a WASM Module
```cpp
WasmScriptSystem& wasmSys = WasmScriptSystem::GetInstance();
wasmSys.LoadWasmModule("scripts/game_logic.wasm");
```

### 4. Call Functions
```cpp
auto instance = wasmSys.GetModuleInstance("game_logic");
instance->Call("init");
instance->Call("update", {WasmValue::F32(deltaTime)});
```

## Usage Examples

### Running Lua, Python, Go, Kotlin, Rust WASM Together
```cpp
// All script systems share the IScriptSystem interface
LuaScriptSystem::GetInstance().RunScript("logic.lua");
PythonScriptSystem::GetInstance().RunScript("ai.py");
GoScriptSystem::GetInstance().RunScript("networking.go");
WasmScriptSystem::GetInstance().RunScript("performance_critical.wasm");
```

### Binding GameObjects
```cpp
auto gameObj = std::make_shared<GameObject>("player");
wasmSys.BindGameObject("player_controller", gameObj);
```

### Custom Engine Bindings
```cpp
auto instance = wasmSys.GetModuleInstance("my_script");
WasmEngineBindings& bindings = WasmEngineBindings::GetInstance();
bindings.RegisterBinding(instance, "custom_function",
    [](std::shared_ptr<WasmInstance> inst, const std::vector<WasmValue>& args) {
        // Custom logic here
        return WasmValue::I32(42);
    }
);
```

## Writing WASM Modules

### Rust Example
```rust
#[no_mangle]
pub extern "C" fn init() { }

#[no_mangle]
pub extern "C" fn update(delta_time: f32) { }

#[no_mangle]
pub extern "C" fn shutdown() { }
```

Compile:
```bash
rustc --target wasm32-unknown-unknown -O script.rs --crate-type cdylib
```

### C Example
```c
void init(void) { }
void update(float delta_time) { }
void shutdown(void) { }
```

### AssemblyScript Example
```typescript
export function init(): void { }
export function update(deltaTime: f32): void { }
export function shutdown(): void { }
```

## Memory Management

### Safe Access
```cpp
// Safe read with bounds checking
auto bytes = instance->ReadMemory(0, 16);

// Safe write with bounds checking
std::vector<uint8_t> data = {1, 2, 3, 4};
bool success = instance->WriteMemory(0, data);

// String operations
std::string message = instance->ReadString(64);
instance->WriteString(128, "Hello");

// Direct allocation
uint32_t ptr = instance->Malloc(256);
instance->Free(ptr);
```

## Engine Bindings Categories

### Physics
- `physics_apply_force()`
- `physics_set_velocity()`
- `physics_cast_ray()`
- `physics_create_rigid_body()`

### Audio
- `audio_play()`
- `audio_stop()`
- `audio_set_volume()`

### Rendering
- `render_set_color()`
- `render_set_material()`
- `render_enable_shadow()`

### Input
- `input_is_key_pressed()`
- `input_get_mouse_pos()`
- `input_is_mouse_button_down()`

### Debug
- `debug_log()`
- `debug_draw_line()`
- `debug_draw_sphere()`

## Performance Characteristics

### Speed
- **Execution**: ~2-10x slower than native C++
- **Memory**: Default 256 MB per module
- **Startup**: <100ms typical module load time

### Profiling
```cpp
instance->SetProfilingEnabled(true);
auto data = instance->GetProfilingData();
for (const auto& d : data) {
    std::cout << d.functionName << ": " 
              << d.averageTime << "ms" << std::endl;
}
```

## Integration Points

### With ECS
```cpp
auto entity = entityManager->CreateEntity();
auto scriptComp = entity->AddComponent<ScriptComponent>();
scriptComp->SetWasmModule("my_script");
```

### With GameObject System
```cpp
auto gameObj = std::make_shared<GameObject>("enemy");
wasmSys.BindGameObject("enemy_ai", gameObj);
```

### With Scripting System
```cpp
IScriptSystem* system = &WasmScriptSystem::GetInstance();
system->Init();
system->RunScript("module.wasm");
system->Update(deltaTime);
```

## CMake Integration

The WASM system requires `wasm3` as a dependency:

```cmake
option(ENABLE_WASM_SUPPORT "Enable WebAssembly support" ON)
if(ENABLE_WASM_SUPPORT)
    FetchContent_Declare(wasm3
        GIT_REPOSITORY https://github.com/wasm3/wasm3.git
        GIT_TAG v0.5.0
    )
    FetchContent_MakeAvailable(wasm3)
    
    add_compile_definitions(ENABLE_WASM_SUPPORT)
    target_link_libraries(GameEngine PRIVATE wasm3)
endif()
```

## Troubleshooting

### Module Loading Fails
```cpp
auto& runtime = WasmRuntime::GetInstance();
std::cout << runtime.GetLastError() << std::endl;
```

### Function Call Issues
```cpp
instance->Call("function_name");
std::cout << instance->GetLastError() << std::endl;
```

### Memory Problems
- Use `GetMemorySize()` to check bounds
- Enable memory protection with `SetMemoryProtection(true)`
- Use `ReadMemory()`/`WriteMemory()` for safe access

## Future Enhancements

- [ ] JIT compilation for better performance
- [ ] WASM-to-native code generation
- [ ] Asynchronous function calls
- [ ] Memory snapshots and serialization
- [ ] Debugger integration with source maps
- [ ] Multi-threading support
- [ ] Component system code generation

## Related Documentation

- [WASM_SUPPORT_GUIDE.md](WASM_SUPPORT_GUIDE.md) - Complete feature reference
- [WASM_EXAMPLES.md](WASM_EXAMPLES.md) - Code examples in multiple languages
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

## Version Info

- **WASM Runtime**: wasm3 v0.5.0+
- **C++ Standard**: C++20
- **Platforms**: Windows, Linux, macOS
- **Build**: CMake 3.10+

