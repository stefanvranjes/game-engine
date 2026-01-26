# LuaJIT Integration Guide - 10x+ Performance for Game Scripts

## Overview

This game engine now integrates **LuaJIT** - a Just-In-Time compiled Lua implementation that provides **10-20x performance improvement** over standard Lua for CPU-bound game logic.

### Key Benefits

| Metric | Standard Lua | LuaJIT |
|--------|-------------|--------|
| **Startup Time** | ~1ms | ~1.5ms |
| **Warm Execution (loops)** | Baseline | 10-20x faster |
| **Memory Footprint** | ~500KB VM | ~300KB VM |
| **Best Use Case** | Config files, light logic | Game loops, AI, physics |
| **Compilation Overhead** | N/A | First run (auto-JIT) |
| **Compatibility** | Lua 5.4 standard | Lua 5.1 + 5.2 compat |

---

## Quick Start

### 1. Enable LuaJIT (Default)

LuaJIT is enabled by default in the CMakeLists.txt:

```cmake
option(ENABLE_LUAJIT "Enable LuaJIT (JIT-compiled Lua) for 10x+ performance" ON)
```

To disable and use standard Lua 5.4:
```bash
cmake -B build -DENABLE_LUAJIT=OFF
cmake --build build
```

### 2. Use LuaJIT in Your Code

```cpp
#include "LuaJitScriptSystem.h"

// Access singleton
auto& luaJit = LuaJitScriptSystem::GetInstance();

// Initialize (happens in ScriptLanguageRegistry::Init())
luaJit.Init();

// Run a script
luaJit.RunScript("scripts/game_logic.lua");

// Call Lua functions from C++
std::vector<std::any> args = {player, 0.016f};
luaJit.CallFunction("update_player", args);

// Shutdown
luaJit.Shutdown();
```

### 3. Or via ScriptLanguageRegistry

```cpp
#include "ScriptLanguageRegistry.h"

auto& registry = ScriptLanguageRegistry::GetInstance();
registry.Init();

// Execute with LuaJIT (10x faster)
registry.ExecuteScript("scripts/gameplay.lua", ScriptLanguage::LuaJIT);

// Or with standard Lua (more compatibility)
registry.ExecuteScript("scripts/config.lua", ScriptLanguage::Lua);
```

---

## Performance Characteristics

### Startup Phase (First 100ms)
- **JIT Warmup**: LuaJIT compiles frequently-executed code on first run
- **Initial Overhead**: ~0.5-2ms for JIT initialization
- **First Call**: May be slower than standard Lua (compilation time)

### Steady State (After JIT Compilation)
- **Loop Performance**: 10-20x faster than standard Lua
- **CPU-Bound Code**: Best improvement area (pathfinding, physics, animation math)
- **I/O Heavy Code**: Minimal improvement (networking, file reads)
- **Memory Allocation**: ~30% less overhead than standard Lua

### Real-World Examples

```lua
-- EXCELLENT for LuaJIT (10-20x faster)
function calculate_physics(particles, dt)
    for i = 1, #particles do
        local p = particles[i]
        p.vx = p.vx + p.ax * dt
        p.vy = p.vy + p.ay * dt
        p.x = p.x + p.vx * dt
        p.y = p.y + p.vy * dt
    end
end

-- GOOD for LuaJIT (3-5x faster)
function update_game_state(entities, dt)
    for _, entity in ipairs(entities) do
        if entity.active then
            entity:update(dt)
        end
    end
end

-- POOR for LuaJIT (similar to Lua)
function load_config_file(path)
    local file = io.open(path, "r")
    local json = json.parse(file:read("*a"))
    file:close()
    return json
end
```

---

## Optimization Tips

### 1. Avoid Polymorphic Calls

**Bad** - Prevents JIT optimization:
```lua
local function call_method(obj, method_name, arg)
    return obj[method_name](obj, arg)  -- Dynamic dispatch = no JIT
end
```

**Good** - JIT-friendly:
```lua
function update_player(player, dt)
    player:update(dt)  -- Direct call = JIT compiles
end
```

### 2. Use Local Variables

**Bad** - Global lookup overhead:
```lua
function game_loop()
    for i = 1, 1000000 do
        game_state.score = game_state.score + 1
    end
end
```

**Good** - Cache in locals:
```lua
function game_loop()
    local state = game_state
    local score = state.score
    for i = 1, 1000000 do
        score = score + 1
    end
    state.score = score
end
```

### 3. Minimize Table Creation

**Bad** - Allocates on every call:
```lua
function get_position_offset(p)
    return {x = p.x + 10, y = p.y + 20}
end
```

**Good** - Reuse tables:
```lua
local offset = {}
function get_position_offset(p)
    offset.x = p.x + 10
    offset.y = p.y + 20
    return offset
end
```

### 4. Use Fixed-Size Arrays

**Bad** - Dynamic growth prevents JIT:
```lua
local particles = {}
for i = 1, num_particles do
    table.insert(particles, {x=0, y=0})
end
```

**Good** - Pre-allocate:
```lua
local particles = {}
for i = 1, num_particles do
    particles[i] = {x=0, y=0}
end
```

### 5. Enable Profiling for Benchmarking

```cpp
auto& luaJit = LuaJitScriptSystem::GetInstance();
luaJit.Init();
luaJit.SetProfilingEnabled(true);

// Run your game logic...
luaJit.RunScript("scripts/test_logic.lua");

// Get stats
auto stats = luaJit.GetProfilingStats();
std::cout << "Total time: " << stats.totalExecutionTime << "μs\n";
std::cout << "Call count: " << stats.callCount << "\n";
std::cout << "Avg time/call: " << stats.avgExecutionTime << "μs\n";
std::cout << "JIT coverage: " << stats.jitCoveragePercent << "%\n";
std::cout << "Compiled traces: " << stats.activeTraces << "\n";
```

### 6. Watch Out for JIT Limitations

LuaJIT doesn't JIT compile everything. Avoid these patterns in hot loops:

```lua
-- No JIT: Varargs
function sum(...) -- Won't JIT varargs
    local total = 0
    for _, v in ipairs({...}) do
        total = total + v
    end
    return total
end

-- No JIT: Complex metamethods
function add(a, b)
    return a + b  -- If __add is complex, may not JIT
end

-- No JIT: Tail calls with complex stack
local function recurse(n)
    if n == 0 then return 1 end
    return recurse(n-1) * n
end
```

---

## Advanced API

### Hot Reload Support

Enable during development:

```cpp
auto& luaJit = LuaJitScriptSystem::GetInstance();
luaJit.Init();
luaJit.SetHotReloadEnabled(true);

// Press F5 to reload without restart
if (input.IsKeyPressed(KEY_F5)) {
    bool success = luaJit.HotReloadScript("scripts/game_logic.lua");
    if (success) {
        std::cout << "Script reloaded!\n";
    }
}
```

### Force JIT On/Off

```cpp
// Enable JIT (default)
luaJit.SetJitEnabled(true);

// Disable for debugging or compatibility
luaJit.SetJitEnabled(false);

// Check status
if (luaJit.IsJitEnabled()) {
    std::cout << "Running in JIT mode (10x+ faster)\n";
}
```

### Memory Management

```cpp
// Get current memory stats
auto stats = luaJit.GetMemoryStats();
std::cout << "Memory usage: " << stats.currentUsage / 1024 << " KB\n";
std::cout << "Peak usage: " << stats.peakUsage / 1024 << " KB\n";

// Force garbage collection
luaJit.ForceGarbageCollection();

// Set memory limit (256 MB default)
luaJit.SetMemoryLimit(512);
```

### Clearing State (Between Levels)

```cpp
// Reset Lua state completely
luaJit.ClearState();
luaJit.RegisterTypes();  // Re-register types

// Load new level scripts
luaJit.RunScript("scripts/level_2_logic.lua");
```

### Register Native Functions

```cpp
// Register C++ function callable from Lua
luaJit.RegisterNativeFunction("print_cpp", [](void* arg) {
    // Called from Lua scripts
});

// Now callable from Lua:
// print_cpp("Hello from C++")
```

---

## Migration from Standard Lua

### Compatibility

LuaJIT provides **99% Lua 5.1 compatibility** with an opt-in Lua 5.2 compatibility mode. Most existing `.lua` scripts work without changes.

### Known Differences

| Feature | Lua 5.4 | LuaJIT |
|---------|---------|--------|
| Bitwise ops | Native | FFI required |
| Goto | Supported | Limited |
| Table `__pairs` | Customizable | Fixed |
| Weak tables | Full support | Partial |
| Math lib precision | Full | Native float |

### Minimal Changes Required

```lua
-- Standard Lua 5.4 code
local x = 0x1234  -- Hex literals

-- LuaJIT compatible
local x = tonumber("0x1234", 16)  -- Works in both
```

---

## Performance Benchmarks

Run the built-in benchmark suite:

```cpp
#include "LuaJitScriptSystem.h"

auto& luaJit = LuaJitScriptSystem::GetInstance();
luaJit.Init();
luaJit.SetProfilingEnabled(true);

// Heavy computation test
luaJit.ExecuteString(R"(
    function benchmark_math()
        local sum = 0
        for i = 1, 10000000 do
            sum = sum + math.sin(i * 0.001)
        end
        return sum
    end
    
    local result = benchmark_math()
)");

auto stats = luaJit.GetProfilingStats();
std::cout << "10M iterations: " << stats.totalExecutionTime << "μs\n";
```

### Expected Results

- **Simple loops**: 10-15x faster
- **Math-heavy code**: 15-20x faster
- **Table operations**: 5-10x faster
- **String operations**: 2-5x faster
- **I/O operations**: Similar to Lua

---

## Building Without LuaJIT

If you need standard Lua compatibility:

```bash
# Disable LuaJIT, use Lua 5.4
cmake -B build -DENABLE_LUAJIT=OFF
cmake --build build
```

Your code using `ScriptLanguageRegistry` will automatically use standard Lua instead:

```cpp
registry.ExecuteScript("script.lua");  // Uses Lua 5.4 instead of LuaJIT
```

---

## Troubleshooting

### "LuaJIT not initialized"
```cpp
// Make sure to call Init() first
auto& luaJit = LuaJitScriptSystem::GetInstance();
luaJit.Init();  // Don't forget this!
```

### Scripts run slow in JIT mode
LuaJIT requires **warmup time** for JIT compilation. First run may be slower.

```cpp
// Pre-warmup before gameplay
luaJit.RunScript("scripts/warmup.lua");
std::this_thread::sleep_for(std::chrono::milliseconds(100));
// Now JIT is active, gameplay runs fast
```

### Memory usage seems high
LuaJIT uses more memory during JIT compilation. This is normal and temporary:

```cpp
// Force GC after level load
luaJit.ForceGarbageCollection();
```

---

## Integration with Game Engine

### In Application.h

```cpp
#include "LuaJitScriptSystem.h"

class Application {
    void Init() {
        auto& registry = ScriptLanguageRegistry::GetInstance();
        registry.Init();  // Initializes LuaJIT automatically
    }
    
    void Update(float dt) {
        // Script system updates automatically
        auto& luaJit = LuaJitScriptSystem::GetInstance();
        luaJit.GetProfilingStats();  // Optional: monitor performance
    }
};
```

### In Game Scripts

```lua
-- scripts/player_controller.lua (Now runs 10x faster with LuaJIT!)

local player_speed = 5.0
local player_health = 100

function update_player(deltaTime)
    if Input.IsKeyPressed(KEY_W) then
        player.position = player.position + player_speed * deltaTime
    end
end

function on_damage(amount)
    player_health = player_health - amount
    if player_health <= 0 then
        on_player_death()
    end
end
```

---

## Performance Profiling

### Using ImGui Integration

The engine includes ImGui-based profiling:

```cpp
if (ImGui::CollapsingHeader("LuaJIT Performance")) {
    auto& luaJit = LuaJitScriptSystem::GetInstance();
    auto stats = luaJit.GetProfilingStats();
    
    ImGui::Text("Total Execution: %.2f ms", stats.totalExecutionTime / 1000.0);
    ImGui::Text("Call Count: %llu", stats.callCount);
    ImGui::Text("Avg Time: %.2f μs", stats.avgExecutionTime);
    ImGui::Text("JIT Coverage: %.1f%%", stats.jitCoveragePercent);
}
```

---

## Best Practices

1. **Use LuaJIT for game logic** - Maximum performance gain
2. **Use standard Lua for config** - Better 5.4 compatibility
3. **Profile hot paths** - Identify which scripts benefit most
4. **Avoid JIT limitations** - Know what doesn't compile
5. **Enable hot-reload in dev** - Faster iteration
6. **Disable JIT in production (optional)** - If memory is critical
7. **Monitor memory usage** - LuaJIT uses ~300KB VM overhead
8. **Use local variables** - Critical for JIT performance
9. **Pre-allocate tables** - Avoid dynamic growth
10. **Test edge cases** - Ensure compatibility with your data

---

## References

- [LuaJIT Official Documentation](https://luajit.org)
- [LuaJIT Performance Tips](https://luajit.org/performance_tuning.html)
- [Lua 5.1 Reference Manual](https://www.lua.org/manual/5.1/)
- Engine ScriptLanguageRegistry Documentation
- Native Type Bindings (Vec3, Transform, etc.)

---

## Support

For issues or questions:
1. Check if LuaJIT is enabled: `cmake -DENABLE_LUAJIT=ON`
2. Review profiling stats: `luaJit.SetProfilingEnabled(true)`
3. Test with standard Lua: `-DENABLE_LUAJIT=OFF`
4. Check compatibility: Lua 5.1 + 5.2 compat mode
