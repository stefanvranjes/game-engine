# WebAssembly (WASM) Support for Game Engine

## Overview

This game engine now includes comprehensive WebAssembly support, allowing you to:

- **Load and execute WASM modules** as game scripts
- **Interface WASM with engine systems** (physics, audio, rendering, input)
- **Hot-reload WASM modules** during development
- **Profile WASM execution** for performance optimization
- **Memory-safe interaction** with WASM linear memory

## Architecture

### Core Components

#### WasmRuntime
- Manages WASM module loading and instantiation
- Handles memory allocation and protection
- Executes functions with timeout enforcement
- Provides error reporting and diagnostics

#### WasmModule
- Represents a compiled WASM binary
- Provides module introspection (exports, memory requirements)
- Validates module structure
- Creates isolated instances

#### WasmInstance
- Represents an instantiated WASM module
- Executes exported functions with type-safe arguments
- Manages linear memory access (safe reads/writes)
- Supports memory allocation via `malloc`/`free`
- Provides profiling data

#### WasmScriptSystem
- Integrates WASM with IScriptSystem interface
- Manages multiple module instances
- Lifecycle callbacks: `init()`, `update(deltaTime)`, `shutdown()`
- Binds GameObjects to WASM scripts
- Hot-reload support

#### WasmEngineBindings
- Bridge between WASM and C++ engine
- Exposes engine functions to WASM modules
- Handles physics, audio, rendering, input, UI interactions
- Extensible binding system

## Usage

### 1. Initialize WASM System

In your Application initialization:

```cpp
#include "Wasm/WasmScriptSystem.h"

// In Application::Init()
WasmScriptSystem::GetInstance().Init();
```

### 2. Load a WASM Module

```cpp
WasmScriptSystem& wasmSys = WasmScriptSystem::GetInstance();

// Load from file
wasmSys.LoadWasmModule("scripts/game_logic.wasm", "game_logic");

// Or use RunScript()
wasmSys.RunScript("scripts/enemy_ai.wasm");
```

### 3. Call WASM Functions

```cpp
// Call function with no arguments
wasmSys.CallWasmFunction("game_logic", "initialize");

// The WASM instance can be accessed for more control
auto instance = wasmSys.GetModuleInstance("game_logic");
if (instance) {
    // Call with type-safe arguments
    instance->Call("update", {WasmValue::F32(deltaTime)});
    
    // Access instance memory
    std::string result = instance->ReadString(0);
    
    // Direct function call
    instance->CallTyped("doSomething", 42, 3.14f);
}
```

### 4. Bind GameObjects to WASM

```cpp
auto enemy = std::make_shared<GameObject>("enemy");
wasmSys.BindGameObject("enemy_ai", enemy);

// The WASM module can now access this object through bindings
```

### 5. Register Custom Bindings

```cpp
WasmEngineBindings& bindings = WasmEngineBindings::GetInstance();

auto instance = wasmSys.GetModuleInstance("my_script");
bindings.RegisterBindings(instance);

// Custom binding
bindings.RegisterBinding(instance, "custom_function",
    [](std::shared_ptr<WasmInstance> inst, const std::vector<WasmValue>& args) {
        // Access engine from inst->GetEngineObject()
        // Read WASM memory if needed
        return WasmValue::I32(42);
    }
);
```

### 6. Enable Hot-Reload

```cpp
WasmScriptSystem& wasmSys = WasmScriptSystem::GetInstance();
wasmSys.EnableHotReload(true);

// Modules will automatically reload when files change
```

### 7. Profile WASM Execution

```cpp
auto instance = wasmSys.GetModuleInstance("my_script");
instance->SetProfilingEnabled(true);

// Run some code...

// Get profiling data
auto profilingData = instance->GetProfilingData();
for (const auto& data : profilingData) {
    std::cout << data.functionName << ": " 
              << data.averageTime << "ms (calls: " 
              << data.callCount << ")" << std::endl;
}
```

## Memory Management

### Safe Memory Access

```cpp
auto instance = wasmSys.GetModuleInstance("my_script");

// Safe read with bounds checking
auto bytes = instance->ReadMemory(0, 16);

// Safe write with bounds checking
std::vector<uint8_t> data = {1, 2, 3, 4};
bool success = instance->WriteMemory(0, data);

// String operations
std::string message = instance->ReadString(64);
instance->WriteString(128, "Hello from C++");

// Get memory information
size_t totalMemory = instance->GetMemorySize();
auto stats = instance->GetStats();
std::cout << "Used memory: " << stats.usedMemory << "/" << stats.totalMemory << std::endl;
```

### Memory Allocation

```cpp
// Allocate WASM heap memory
uint32_t ptr = instance->Malloc(256);
if (ptr != 0) {
    // Write data to allocated memory
    instance->WriteMemory(ptr, data);
    
    // Use it...
    
    // Free when done
    instance->Free(ptr);
}
```

## Engine Bindings

### Available Binding Categories

#### Physics
- `physics_apply_force()` - Apply force to rigid body
- `physics_set_velocity()` - Set velocity directly
- `physics_cast_ray()` - Cast physics ray
- `physics_create_rigid_body()` - Create rigid body

#### Audio
- `audio_play()` - Play sound at position
- `audio_stop()` - Stop sound
- `audio_set_volume()` - Set audio volume

#### Rendering
- `render_set_color()` - Set object color
- `render_set_material()` - Set material
- `render_enable_shadow()` - Enable/disable shadow

#### Input
- `input_is_key_pressed()` - Check key state
- `input_get_mouse_pos()` - Get mouse position
- `input_is_mouse_button_down()` - Check mouse button

#### Debug
- `debug_log()` - Output debug message
- `debug_draw_line()` - Draw debug line
- `debug_draw_sphere()` - Draw debug sphere

## Lifecycle Hooks

WASM modules can export optional lifecycle functions:

```wasm
;; Called once when module loads
(export "init" (func $init))
(func $init
  ;; Initialize module state
)

;; Called every frame
(export "update" (func $update))
(func $update (param $deltaTime f32)
  ;; Update game logic
)

;; Called when module unloads
(export "shutdown" (func $shutdown))
(func $shutdown
  ;; Cleanup resources
)
```

## Writing WASM Modules

### Rust Example

```rust
#[no_mangle]
pub extern "C" fn init() {
    // Initialization
}

#[no_mangle]
pub extern "C" fn update(delta_time: f32) {
    // Game logic here
    let mut x = 0.0;
    x += 5.0 * delta_time;
}

#[no_mangle]
pub extern "C" fn shutdown() {
    // Cleanup
}
```

Compile with:
```bash
rustc --target wasm32-unknown-unknown -O script.rs -o script.wasm
```

### C Example

```c
void init(void) {
    // Initialization
}

void update(float delta_time) {
    // Game logic
}

void shutdown(void) {
    // Cleanup
}
```

Compile with:
```bash
clang --target=wasm32 -O2 script.c -c -o script.o
wasm-ld script.o -o script.wasm --no-entry --export=init --export=update --export=shutdown
```

### AssemblyScript Example

```typescript
export function init(): void {
    // Initialization
}

export function update(deltaTime: f32): void {
    // Game logic
}

export function shutdown(): void {
    // Cleanup
}
```

Compile with:
```bash
asc script.ts -O3 -o script.wasm
```

## Performance Considerations

### Memory Limits
```cpp
// Set max WASM memory per module
WasmRuntime::GetInstance().SetMaxMemorySize(512);  // 512 MB
WasmScriptSystem::GetInstance().SetDefaultMemoryLimit(256);  // 256 MB
```

### Execution Timeout
```cpp
// Prevent runaway scripts
WasmRuntime::GetInstance().SetExecutionTimeout(true, 5000);  // 5 seconds
```

### Profiling
```cpp
instance->SetProfilingEnabled(true);

// After execution
auto profile = instance->GetProfilingData();
for (const auto& data : profile) {
    std::cout << data.functionName << ": " 
              << data.totalTime << "ms total, "
              << data.averageTime << "ms avg" << std::endl;
}
```

## Debugging

### Validate Modules

```cpp
auto module = WasmRuntime::GetInstance().GetModule("my_script");
if (WasmHelper::ValidateModuleInterface(module)) {
    std::cout << "Module valid" << std::endl;
}
```

### Print Module Info

```cpp
WasmHelper::PrintWasmModule(module);
```

### Memory Dump

```cpp
WasmHelper::PrintWasmMemory(instance, 0, 256);
```

### Error Handling

```cpp
auto instance = wasmSys.GetModuleInstance("my_script");
instance->Call("do_something");

if (!instance->GetLastError().empty()) {
    std::cerr << "Error: " << instance->GetLastError() << std::endl;
}

// Check runtime errors
auto& runtime = WasmRuntime::GetInstance();
if (!runtime.GetLastError().empty()) {
    std::cerr << "Runtime error: " << runtime.GetLastError() << std::endl;
}
```

## Integration with Existing Systems

### With Scripting System

```cpp
// All script systems share the same interface
IScriptSystem* wasmSys = &WasmScriptSystem::GetInstance();
wasmSys->Init();
wasmSys->RunScript("my_wasm_module.wasm");
wasmSys->Update(deltaTime);
```

### With ECS

```cpp
// Bind WASM script to entity
auto entity = entityManager->CreateEntity();
auto scriptComp = entity->AddComponent<ScriptComponent>();
scriptComp->SetWasmModule("my_script");
```

### With GameObject System

```cpp
auto gameObj = std::make_shared<GameObject>("enemy");
auto scriptComp = gameObj->AddComponent<ScriptComponent>();
wasmSys.BindGameObject("enemy_ai", gameObj);
wasmSys.LoadWasmModule("enemy_ai.wasm", "enemy_ai");
```

## CMake Integration

The WASM support requires the `wasm3` library. Add to CMakeLists.txt:

```cmake
# WASM support
option(ENABLE_WASM "Enable WebAssembly support" ON)
if(ENABLE_WASM)
    include(FetchContent)
    FetchContent_Declare(wasm3
        GIT_REPOSITORY https://github.com/wasm3/wasm3.git
        GIT_TAG v0.5.0
    )
    FetchContent_MakeAvailable(wasm3)
    
    add_compile_definitions(ENABLE_WASM_SUPPORT)
    target_link_libraries(GameEngine PRIVATE wasm3)
endif()
```

## Limitations

1. **Performance**: WASM interpreter is slower than native code (~2-10x)
2. **Memory**: Default 256 MB limit per module (configurable)
3. **No Direct GFX**: WASM can't directly call graphics APIs (use bindings)
4. **Floating Point**: Some platforms limit SIMD operations
5. **Threading**: WASM is single-threaded by design

## Future Enhancements

- [ ] WASM JIT compilation support
- [ ] Asynchronous function calls
- [ ] Memory snapshots and serialization
- [ ] Debugger integration
- [ ] Source maps for debugging
- [ ] Component system generation from WASM

## Troubleshooting

### Module fails to load
```cpp
auto& runtime = WasmRuntime::GetInstance();
std::cout << runtime.GetLastError() << std::endl;
```

### Function call fails
```cpp
instance->Call("function_name");
std::cout << instance->GetLastError() << std::endl;
```

### Memory access errors
- Check bounds with `GetMemorySize()`
- Use `ReadMemory()` and `WriteMemory()` for safe access
- Use `Malloc()` for heap allocations

### Performance issues
- Enable profiling to identify slow functions
- Reduce WASM memory footprint
- Use hot reload only in development
- Consider native implementation for critical paths

