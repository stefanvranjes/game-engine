# Scripting Profiler UI - Examples & Recipes

## Example 1: Basic Profiling

### Setup
```cpp
// Application is already set up. Just open the profiler:
// Tools → Scripting Profiler (Ctrl+Shift+P)
```

### Lua Script to Profile
```lua
-- scripts/profile_test.lua

function fibonacci(n)
    if n <= 1 then
        return n
    end
    return fibonacci(n - 1) + fibonacci(n - 2)
end

function update_game(deltaTime)
    -- This will be profiled automatically
    local result = fibonacci(20)
    print("Fibonacci result: " .. result)
end

function process_input()
    -- Another function to profile
    local inputs = 0
    for i = 1, 1000 do
        inputs = inputs + 1
    end
    return inputs
end
```

### Profiling Session
1. Load script: `luaJit.RunScript("scripts/profile_test.lua")`
2. Open Scripting Profiler (Ctrl+Shift+P)
3. Go to "Language Details" → "LuaJIT" tab
4. Watch metrics update as functions are called
5. Note JIT coverage increasing over time

### Expected Output
```
LuaJIT Performance Metrics:
  Total Execution Time: 5.234 ms
  Call Count: 1024
  Avg Time Per Call: 5.11 μs
  JIT Statistics:
    Active Traces: 12
    JIT Compiled Functions: 3
    JIT Coverage: 87.5%
```

---

## Example 2: Performance Comparison

### Compare Two Script Implementations

#### Version 1: Unoptimized
```lua
-- scripts/fibonacci_slow.lua
function fib_recursive(n)
    if n <= 1 then return n end
    return fib_recursive(n-1) + fib_recursive(n-2)
end

function main()
    for i = 1, 100 do
        fib_recursive(20)
    end
end
```

#### Version 2: Optimized
```lua
-- scripts/fibonacci_fast.lua
function fib_iterative(n)
    if n <= 1 then return n end
    local a, b = 0, 1
    for i = 2, n do
        a, b = b, a + b
    end
    return b
end

function main()
    for i = 1, 100 do
        fib_iterative(20)
    end
end
```

### Testing Process
```cpp
void ComparePerformance() {
    auto& profiler = /* access profiler UI */;
    
    // Test 1: Unoptimized
    auto& luaJit = LuaJitScriptSystem::GetInstance();
    profiler->ClearData();
    profiler->SetPaused(false);
    
    luaJit.RunScript("scripts/fibonacci_slow.lua");
    luaJit.CallFunction("main", {});
    
    auto statsV1 = luaJit.GetProfilingStats();
    std::cout << "V1 Time: " << statsV1.totalExecutionTime << " μs\n";
    
    // Export results
    profiler->ExportToJSON("profile_v1.json");
    
    // Test 2: Optimized
    profiler->ClearData();
    luaJit.RunScript("scripts/fibonacci_fast.lua");
    luaJit.CallFunction("main", {});
    
    auto statsV2 = luaJit.GetProfilingStats();
    std::cout << "V2 Time: " << statsV2.totalExecutionTime << " μs\n";
    std::cout << "Improvement: " 
              << (statsV1.totalExecutionTime / statsV2.totalExecutionTime)
              << "x faster\n";
    
    profiler->ExportToJSON("profile_v2.json");
}
```

### Analysis with CSV
```python
# analyze_profiles.py
import csv
import json

def compare_exports():
    with open('profile_v1.json') as f:
        v1 = json.load(f)
    with open('profile_v2.json') as f:
        v2 = json.load(f)
    
    time_v1 = v1['languages'][0]['total_execution_time_ms']
    time_v2 = v2['languages'][0]['total_execution_time_ms']
    
    improvement = (time_v1 / time_v2) * 100 - 100
    print(f"Performance improvement: {improvement:.1f}%")
    print(f"V1: {time_v1:.3f}ms, V2: {time_v2:.3f}ms")

compare_exports()
```

---

## Example 3: Memory Leak Detection

### Script with Potential Memory Issue
```lua
-- scripts/memory_test.lua
-- This accumulates memory without cleanup

local tables = {}

function accumulate_memory()
    for i = 1, 100 do
        local t = {}
        for j = 1, 1000 do
            t[j] = { data = "x", value = math.random() }
        end
        table.insert(tables, t)  -- Keeps growing!
    end
end

function cleanup()
    tables = {}  -- Manual cleanup
    collectgarbage()
end
```

### Detection Process
```cpp
void DetectMemoryLeak() {
    auto& profiler = /* access profiler UI */;
    auto& luaJit = LuaJitScriptSystem::GetInstance();
    
    // Monitor memory over time
    profiler->SetMaxHistorySamples(100);
    profiler->SetPaused(false);
    
    // Run accumulating function multiple times
    for (int frame = 0; frame < 50; ++frame) {
        luaJit.CallFunction("accumulate_memory", {});
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Export memory history
    profiler->ExportToJSON("memory_profile.json");
    
    // Analyze: if memory keeps growing, it's a leak
}
```

### Detection Script
```python
# detect_memory_leak.py
import json

def detect_leak(filename):
    with open(filename) as f:
        data = json.load(f)
    
    stats = data['languages'][0]
    memory_history = stats.get('memory_history', [])
    
    if len(memory_history) > 2:
        start = memory_history[0]
        end = memory_history[-1]
        growth = end - start
        
        if growth > start * 0.5:  # 50% growth
            print(f"LEAK DETECTED: Memory grew {growth/start*100:.1f}%")
            print(f"Start: {start:.2f}MB, End: {end:.2f}MB")
        else:
            print("No significant memory leak detected")

detect_leak('memory_profile.json')
```

---

## Example 4: Frame Rate Analysis

### Game Update Loop Profiling
```cpp
void Application::Update(float deltaTime) {
    // Profiler automatically tracks this
    auto& profiler = m_ScriptingProfilerUI;
    
    // Script execution
    auto& luaJit = LuaJitScriptSystem::GetInstance();
    
    // Call game update
    luaJit.CallFunction("game_update", { deltaTime });
    
    // Profiler UI will show:
    // - Real-time execution times
    // - Frame-to-frame consistency
    // - JIT optimization progress
}
```

### Lua Game Script
```lua
-- scripts/game.lua

local frameCount = 0

function game_update(deltaTime)
    frameCount = frameCount + 1
    
    -- Typical game operations
    update_player(deltaTime)
    update_enemies(deltaTime)
    update_physics(deltaTime)
    render_scene()
    
    -- Debug: print every 60 frames
    if frameCount % 60 == 0 then
        print("Frame " .. frameCount .. " updated in " .. deltaTime .. "ms")
    end
end

function update_player(dt)
    -- Player logic
    for i = 1, 100 do
        local x = math.sin(i * dt)
    end
end

function update_enemies(dt)
    -- Enemy AI
    for i = 1, 50 do
        local behavior = math.cos(i * dt)
    end
end

function update_physics(dt)
    -- Physics simulation
    for i = 1, 1000 do
        local force = math.random()
    end
end

function render_scene()
    -- Scene rendering calls
end
```

### Monitoring Approach
1. Open Profiler (Ctrl+Shift+P)
2. Go to "Performance Charts" tab
3. Enable both charts
4. Play game scenario
5. Watch execution time graph for spikes
6. If spike found, note timing and check script logs

---

## Example 5: JIT Coverage Optimization

### Scenario: Low JIT Coverage

```lua
-- scripts/jit_issue.lua

-- Non-compilable pattern (causes JIT bailout)
function problematic_function(x)
    if x > 0 then
        return tostring(x)  -- String conversion isn't JIT-able
    else
        return x
    end
end

-- Optimized version (JIT-friendly)
function optimized_function(x)
    if x > 0 then
        return x  -- Keep as number
    else
        return x
    end
end
```

### Testing JIT Coverage
```cpp
void TestJITOptimization() {
    auto& luaJit = LuaJitScriptSystem::GetInstance();
    luaJit.Init();
    luaJit.SetProfilingEnabled(true);
    luaJit.RunScript("scripts/jit_issue.lua");
    
    // Warm up JIT
    for (int i = 0; i < 10000; ++i) {
        luaJit.CallFunction("problematic_function", { i });
    }
    
    auto stats = luaJit.GetProfilingStats();
    std::cout << "Problematic JIT Coverage: " 
              << stats.jitCoveragePercent << "%\n";
    
    // Now test optimized version
    luaJit.ResetProfilingStats();
    for (int i = 0; i < 10000; ++i) {
        luaJit.CallFunction("optimized_function", { i });
    }
    
    stats = luaJit.GetProfilingStats();
    std::cout << "Optimized JIT Coverage: " 
              << stats.jitCoveragePercent << "%\n";
}
```

### Profiler UI Check
1. Open Language Details → LuaJIT
2. Compare JIT Coverage % between runs
3. Lower coverage = more bailouts = slower code
4. Optimize hot loops to be JIT-compilable

---

## Example 6: Multi-Language Comparison

### Running Different Script Languages

```cpp
void CompareScriptLanguages() {
    auto& profiler = /* access profiler UI */;
    
    // Test LuaJIT
    {
        auto& luaJit = LuaJitScriptSystem::GetInstance();
        luaJit.Init();
        profiler->SetLanguageProfilingEnabled("LuaJIT", true);
        
        luaJit.RunScript("scripts/benchmark.lua");
        for (int i = 0; i < 1000; ++i) {
            luaJit.CallFunction("benchmark_function", {});
        }
        profiler->ExportToJSON("benchmark_luajit.json");
    }
    
    // Test AngelScript
    {
        auto& angel = AngelScriptSystem::GetInstance();
        angel.Init();
        profiler->SetLanguageProfilingEnabled("AngelScript", true);
        
        angel.RunScript("scripts/benchmark.as");
        for (int i = 0; i < 1000; ++i) {
            angel.CallFunction("benchmark_function", {});
        }
        profiler->ExportToJSON("benchmark_angel.json");
    }
    
    // Compare results
    CompareResults("benchmark_luajit.json", "benchmark_angel.json");
}

void CompareResults(const std::string& file1, const std::string& file2) {
    // Load and compare JSON files
    // (Python script would be better for this)
}
```

---

## Example 7: Automated Regression Testing

### Continuous Integration Integration

```cpp
// performance_test.cpp
#include "ScriptingProfilerUI.h"
#include "LuaJitScriptSystem.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct PerformanceThreshold {
    std::string testName;
    double maxExecutionTimeMs;
    size_t maxMemoryBytes;
};

bool RunPerformanceTest(
    const std::string& scriptPath,
    const std::string& functionName,
    const PerformanceThreshold& threshold,
    const std::string& reportPath)
{
    auto& luaJit = LuaJitScriptSystem::GetInstance();
    auto profiler = std::make_unique<ScriptingProfilerUI>();
    
    profiler->Init();
    luaJit.SetProfilingEnabled(true);
    luaJit.RunScript(scriptPath);
    
    // Run test
    for (int i = 0; i < 100; ++i) {
        luaJit.CallFunction(functionName, {});
    }
    
    auto stats = luaJit.GetProfilingStats();
    double avgTime = stats.totalExecutionTime / stats.callCount / 1000.0; // Convert to ms
    
    // Check thresholds
    bool passed = avgTime <= threshold.maxExecutionTimeMs;
    
    // Report
    json report = json::object({
        {"test_name", threshold.testName},
        {"passed", passed},
        {"execution_time_ms", avgTime},
        {"threshold_ms", threshold.maxExecutionTimeMs},
        {"margin", (threshold.maxExecutionTimeMs - avgTime) / threshold.maxExecutionTimeMs * 100}
    });
    
    std::ofstream file(reportPath);
    file << report.dump(2);
    
    profiler->ExportToJSON(reportPath + ".detailed.json");
    
    return passed;
}

// Usage in CI
int main() {
    std::vector<PerformanceThreshold> tests = {
        { "game_update", 5.0, 1024 * 1024 },      // 5ms, 1MB
        { "physics_step", 2.0, 512 * 1024 },      // 2ms, 512KB
        { "render_frame", 10.0, 2048 * 1024 }     // 10ms, 2MB
    };
    
    bool all_passed = true;
    for (const auto& test : tests) {
        bool passed = RunPerformanceTest(
            "scripts/game.lua",
            test.testName,
            test,
            "report_" + test.testName + ".json"
        );
        all_passed = all_passed && passed;
    }
    
    return all_passed ? 0 : 1;  // CI exit code
}
```

---

## Example 8: Real-Time Optimization

### Before/After Profiling

#### Before
```lua
function inefficient_filter(data)
    local result = {}
    for i = 1, #data do
        if data[i] > 0 then
            table.insert(result, data[i])  -- Slow!
        end
    end
    return result
end
```

Profiler shows: 50ms per call

#### After
```lua
function efficient_filter(data)
    local result = {}
    local count = 0
    for i = 1, #data do
        if data[i] > 0 then
            count = count + 1
            result[count] = data[i]  -- Faster (pre-allocated)
        end
    end
    return result
end
```

Profiler shows: 5ms per call (10x faster!)

### Verification Steps
```cpp
void VerifyOptimization() {
    auto& profiler = /* access UI */;
    auto& luaJit = LuaJitScriptSystem::GetInstance();
    
    // Before
    profiler->ClearData();
    luaJit.RunScript("scripts/inefficient.lua");
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        luaJit.CallFunction("inefficient_filter", { /* data */ });
    }
    
    auto beforeTime = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - startTime
    ).count();
    
    profiler->ExportToJSON("before.json");
    
    // After (same process)
    // ...
    
    std::cout << "Improvement: " << beforeTime / afterTime << "x\n";
}
```

---

## Summary

These examples demonstrate various profiling workflows:

1. **Basic Profiling** - Get started with the UI
2. **Performance Comparison** - Compare implementations
3. **Memory Leak Detection** - Find memory issues
4. **Frame Rate Analysis** - Monitor game performance
5. **JIT Optimization** - Maximize JIT compilation
6. **Multi-Language** - Compare languages
7. **CI/CD Integration** - Automated testing
8. **Real-Time Optimization** - Verify improvements

All examples integrate with the Scripting Profiler UI for visual monitoring and data export.
