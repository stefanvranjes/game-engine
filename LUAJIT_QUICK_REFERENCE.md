# LuaJIT Quick Reference

## Enable LuaJIT (Default)

```cmake
# In CMakeLists.txt (enabled by default)
option(ENABLE_LUAJIT "Enable LuaJIT (JIT-compiled Lua) for 10x+ performance" ON)
```

## Basic Usage

```cpp
#include "LuaJitScriptSystem.h"

auto& jit = LuaJitScriptSystem::GetInstance();
jit.Init();
jit.RunScript("script.lua");
jit.Shutdown();
```

## Via ScriptLanguageRegistry

```cpp
#include "ScriptLanguageRegistry.h"

auto& registry = ScriptLanguageRegistry::GetInstance();
registry.Init();

// Run with LuaJIT (10x faster)
registry.ExecuteScript("script.lua", ScriptLanguage::LuaJIT);

// Run with standard Lua
registry.ExecuteScript("config.lua", ScriptLanguage::Lua);
```

## Profiling

```cpp
auto& jit = LuaJitScriptSystem::GetInstance();
jit.SetProfilingEnabled(true);

jit.RunScript("test.lua");

auto stats = jit.GetProfilingStats();
// stats.totalExecutionTime
// stats.callCount
// stats.avgExecutionTime
// stats.jitCoveragePercent
```

## Hot Reload

```cpp
jit.SetHotReloadEnabled(true);
jit.HotReloadScript("script.lua");
```

## JIT Control

```cpp
jit.SetJitEnabled(true);   // Enable JIT (10x+ faster)
jit.SetJitEnabled(false);  // Disable JIT (more compatible)
jit.IsJitEnabled();        // Check status
```

## Memory Management

```cpp
auto stats = jit.GetMemoryStats();
jit.SetMemoryLimit(256);  // MB
jit.ForceGarbageCollection();
jit.ClearState();  // Reset between levels
```

## Key Performance Tips

| Good | Bad |
|------|-----|
| Use local variables | Access globals in loops |
| Pre-allocate tables | Dynamic table growth |
| Direct calls | Polymorphic dispatch |
| Fixed-size arrays | Dynamic insertion |
| Numeric loops | Generic iteration |
| Math-heavy code | Balanced operations |

## Performance Gains

| Operation | Speedup |
|-----------|---------|
| Loops | 10-20x |
| Math | 15-20x |
| Tables | 5-10x |
| Strings | 2-5x |
| I/O | ~1x |

## API Reference

### Lifecycle
- `Init()` - Initialize
- `Shutdown()` - Cleanup
- `Update(float)` - Per-frame update

### Execution
- `RunScript(filepath)` - Load and execute
- `ExecuteString(source)` - Execute inline code
- `CallFunction(name, args)` - Call Lua function from C++
- `HasFunction(name)` - Check if function exists

### Configuration
- `SetJitEnabled(bool)` - Toggle JIT compilation
- `SetProfilingEnabled(bool)` - Enable metrics
- `SetHotReloadEnabled(bool)` - Enable dev reload
- `SetMemoryLimit(MB)` - Set memory cap

### Introspection
- `IsJitEnabled()` - Check JIT status
- `GetProfilingStats()` - Get performance metrics
- `GetMemoryStats()` - Get memory usage
- `GetLuaState()` - Get raw Lua state

### Type Registration
- `RegisterTypes()` - Register C++ types (Vec3, etc.)
- `RegisterNativeFunction(name, fn)` - Register C++ function

### Variables
- `SetGlobalVariable(name, value)` - Set Lua global
- `GetGlobalVariable(name)` - Get Lua global

### Advanced
- `HotReloadScript(path)` - Reload during dev
- `ClearState()` - Reset all Lua state
- `ForceGarbageCollection()` - Trigger GC

## Building Without LuaJIT

```bash
cmake -B build -DENABLE_LUAJIT=OFF
cmake --build build
```

## Expected Performance

```
Standard Lua:  Time = T
LuaJIT:        Time = T/10 to T/20  (with JIT warmup)
First run:     Similar to Lua (JIT compiling)
Steady state:  10-20x faster
```

## Compatibility

- **Lua Version**: 5.1 + optional 5.2 compat
- **Most Scripts**: Work without changes
- **Bitwise Ops**: Use FFI
- **Goto**: Limited support
- **Weak Tables**: Partial support

## Common Issues

| Problem | Solution |
|---------|----------|
| "Not initialized" | Call `Init()` first |
| Slow first run | JIT warmup is normal |
| Memory spike | Force GC with `ForceGarbageCollection()` |
| Compatibility | Switch to `ScriptLanguage::Lua` |

## Benchmark Example

```cpp
auto& jit = LuaJitScriptSystem::GetInstance();
jit.Init();
jit.SetProfilingEnabled(true);

for (int i = 0; i < 100; i++) {
    jit.ExecuteString("sum = 0; for j=1,10000 do sum=sum+j end");
}

auto stats = jit.GetProfilingStats();
double avgTime = stats.totalExecutionTime / stats.callCount;
std::cout << "Avg: " << avgTime << " μs\n";
std::cout << "JIT: " << stats.jitCoveragePercent << "%\n";
```

## Files Modified

- `CMakeLists.txt` - Added LuaJIT fetch
- `include/IScriptSystem.h` - Added ScriptLanguage::LuaJIT
- `include/LuaJitScriptSystem.h` - LuaJIT wrapper class
- `src/LuaJitScriptSystem.cpp` - Implementation
- `src/ScriptLanguageRegistry.cpp` - LuaJIT registration
- `LUAJIT_INTEGRATION_GUIDE.md` - Full documentation

## See Also

- LUAJIT_INTEGRATION_GUIDE.md - Complete guide with examples
- SCRIPTING_QUICK_START.md - Multi-language scripting overview
- MULTI_LANGUAGE_SCRIPTING_GUIDE.md - Language comparisons
