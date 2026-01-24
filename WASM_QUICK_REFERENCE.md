# WASM Quick Reference Card

## Initialization

```cpp
#include "Wasm/WasmScriptSystem.h"
#include "Wasm/WasmEngineBindings.h"

// Initialize system
WasmScriptSystem::GetInstance().Init();
```

## Loading Modules

```cpp
WasmScriptSystem& wasmSys = WasmScriptSystem::GetInstance();

// Load from file
wasmSys.LoadWasmModule("scripts/game_logic.wasm");
wasmSys.LoadWasmModule("scripts/ai.wasm", "enemy_ai");

// Access module
auto instance = wasmSys.GetModuleInstance("game_logic");
```

## Function Calls

```cpp
auto instance = wasmSys.GetModuleInstance("game_logic");

// No arguments
instance->Call("init");

// With arguments
instance->Call("update", {WasmValue::F32(deltaTime)});
instance->Call("damage", {WasmValue::I32(0), WasmValue::F32(25.0f)});

// Type-safe (C++20)
instance->CallTyped("doSomething", 42, 3.14f, true);
```

## Memory Access

```cpp
// Read memory
auto bytes = instance->ReadMemory(0, 16);

// Write memory
std::vector<uint8_t> data = {1, 2, 3, 4};
instance->WriteMemory(0, data);

// Strings
instance->WriteString(128, "Hello");
std::string msg = instance->ReadString(128);

// Get memory info
size_t totalSize = instance->GetMemorySize();
auto stats = instance->GetStats();
```

## Lifecycle Hooks

```wasm
;; Rust
#[no_mangle]
pub extern "C" fn init() { }
pub extern "C" fn update(delta_time: f32) { }
pub extern "C" fn shutdown() { }

;; C
void init(void) { }
void update(float delta_time) { }
void shutdown(void) { }

;; TypeScript
export function init(): void { }
export function update(deltaTime: f32): void { }
export function shutdown(): void { }
```

Engine calls automatically:
- `init()` - When module loads
- `update(deltaTime)` - Each frame
- `shutdown()` - When unloading

## Engine Bindings

```cpp
// Register bindings
WasmEngineBindings& bindings = WasmEngineBindings::GetInstance();
bindings.RegisterBindings(instance);

// Available in WASM:
// physics_apply_force()
// audio_play()
// render_set_color()
// input_is_key_pressed()
// debug_log()
```

## Custom Bindings

```cpp
bindings.RegisterBinding(instance, "my_function",
    [](std::shared_ptr<WasmInstance> inst, const std::vector<WasmValue>& args) {
        // Read WASM arguments
        if (args.size() > 0) {
            auto val = std::any_cast<int32_t>(args[0].value);
        }
        
        // Return to WASM
        return WasmValue::I32(result);
    }
);
```

## Memory Allocation

```cpp
// Allocate
uint32_t ptr = instance->Malloc(256);

if (ptr != 0) {
    // Use...
    instance->WriteMemory(ptr, data);
    
    // Free
    instance->Free(ptr);
}
```

## Profiling

```cpp
instance->SetProfilingEnabled(true);

// ... run code ...

auto data = instance->GetProfilingData();
for (const auto& d : data) {
    std::cout << d.functionName << ": "
              << d.averageTime << "ms (calls: " 
              << d.callCount << ")" << std::endl;
}
```

## Error Handling

```cpp
instance->Call("do_something");

if (!instance->GetLastError().empty()) {
    std::cerr << "Error: " << instance->GetLastError() << std::endl;
}

// Runtime errors
auto& runtime = WasmRuntime::GetInstance();
std::cout << runtime.GetLastError() << std::endl;
```

## Binding GameObjects

```cpp
auto gameObj = std::make_shared<GameObject>("player");
wasmSys.BindGameObject("player_script", gameObj);

// Access in C++
auto boundObj = wasmSys.GetModuleInstance("player_script")
                        ->GetEngineObject();
```

## Hot-Reload

```cpp
wasmSys.EnableHotReload(true);
// Modules auto-reload when files change (development only)
```

## Value Types

```cpp
WasmValue::I32(42)          // 32-bit integer
WasmValue::I64(100000)      // 64-bit integer
WasmValue::F32(3.14f)       // 32-bit float
WasmValue::F64(2.71828)     // 64-bit double
WasmValue::Ptr(address)     // Memory pointer (as I32)
```

## Common Patterns

### Load and Call

```cpp
wasmSys.LoadWasmModule("script.wasm");
auto instance = wasmSys.GetModuleInstance("script");
instance->Call("init");
instance->Call("update", {WasmValue::F32(deltaTime)});
```

### With Bindings

```cpp
wasmSys.LoadWasmModule("script.wasm");
auto instance = wasmSys.GetModuleInstance("script");
WasmEngineBindings::GetInstance().RegisterBindings(instance);
instance->Call("init");
```

### Per-Frame Update

```cpp
void Application::Update(float deltaTime) {
    auto& wasmSys = WasmScriptSystem::GetInstance();
    
    for (const auto& moduleName : wasmSys.GetLoadedModules()) {
        auto instance = wasmSys.GetModuleInstance(moduleName);
        instance->Call("update", {WasmValue::F32(deltaTime)});
    }
}
```

### Data Exchange

```cpp
// Write data
instance->WriteMemory(0, {1, 2, 3, 4});

// Call function that processes data
instance->Call("process", {WasmValue::I32(0), WasmValue::I32(4)});

// Read result
auto result = instance->ReadMemory(256, 16);
```

## Compilation Commands

### Rust
```bash
rustc --target wasm32-unknown-unknown -O script.rs --crate-type cdylib
```

### C
```bash
clang --target=wasm32 -O3 -nostdlib \
  -Wl,--no-entry \
  -Wl,--export=init,--export=update,--export=shutdown \
  script.c -o script.wasm
```

### AssemblyScript
```bash
asc script.ts -O3 -o script.wasm
```

## Configuration

```cpp
// Memory
WasmRuntime::GetInstance().SetMaxMemorySize(512);  // MB

// Timeout
WasmRuntime::GetInstance().SetExecutionTimeout(true, 5000);  // ms

// Memory protection
WasmRuntime::GetInstance().SetMemoryProtection(true);

// Profiling
instance->SetProfilingEnabled(true);

// Hot-reload
wasmSys.EnableHotReload(true);
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module won't load | Check `WasmRuntime::GetInstance().GetLastError()` |
| Function not found | Use `wasm-objdump -x module.wasm \| grep Export` |
| Memory error | Use `ReadMemory()`/`WriteMemory()` for safe access |
| Slow execution | Enable profiling, check function hotspots |
| Crashes | Enable memory protection, check bounds |

## Resources

- **Full Guide:** WASM_SUPPORT_GUIDE.md
- **Examples:** WASM_EXAMPLES.md
- **Build Guide:** WASM_TOOLING_GUIDE.md
- **API Reference:** WASM_SUPPORT_INDEX.md
- **Implementation:** WASM_IMPLEMENTATION_SUMMARY.md

## Header Files

```cpp
#include "Wasm/WasmRuntime.h"          // Core runtime
#include "Wasm/WasmModule.h"           // Module representation
#include "Wasm/WasmInstance.h"         // Instance execution
#include "Wasm/WasmScriptSystem.h"     // Script system integration
#include "Wasm/WasmEngineBindings.h"   // Engine functions
#include "Wasm/WasmHelper.h"           // Utility functions
```

## Quick Start (5 Minutes)

```cpp
// 1. Initialize
WasmScriptSystem::GetInstance().Init();

// 2. Load module
WasmScriptSystem::GetInstance()
    .LoadWasmModule("scripts/game.wasm");

// 3. Get instance
auto instance = WasmScriptSystem::GetInstance()
    .GetModuleInstance("game");

// 4. Register bindings
WasmEngineBindings::GetInstance()
    .RegisterBindings(instance);

// 5. Call functions
instance->Call("init");
instance->Call("update", {WasmValue::F32(0.016f)});
```

