# LuaJIT Integration Examples

## Example 1: Basic Game Loop Integration

```cpp
// In Application.cpp

#include "LuaJitScriptSystem.h"
#include "ScriptLanguageRegistry.h"

class Application {
private:
    LuaJitScriptSystem* m_LuaJIT;
    
public:
    void Init() {
        // Initialize scripting registry (includes LuaJIT)
        auto& registry = ScriptLanguageRegistry::GetInstance();
        registry.Init();
        
        m_LuaJIT = dynamic_cast<LuaJitScriptSystem*>(
            registry.GetScriptSystem(ScriptLanguage::LuaJIT)
        );
        
        // Enable profiling for performance monitoring
        m_LuaJIT->SetProfilingEnabled(true);
        
        // Enable hot-reload for development
        m_LuaJIT->SetHotReloadEnabled(true);
        
        // Load game scripts
        m_LuaJIT->RunScript("scripts/game_init.lua");
    }
    
    void Update(float deltaTime) {
        // Call Lua update function
        std::vector<std::any> args = {deltaTime};
        m_LuaJIT->CallFunction("update_game", args);
        
        // Check profiling stats (optional)
        if (m_ShowProfiler) {
            auto stats = m_LuaJIT->GetProfilingStats();
            DebugPrint("LuaJIT: %.2f ms total, %.1f%%JIT coverage",
                stats.totalExecutionTime / 1000.0,
                stats.jitCoveragePercent);
        }
    }
    
    void Shutdown() {
        auto& registry = ScriptLanguageRegistry::GetInstance();
        registry.Shutdown();
    }
};
```

## Example 2: Scripted Game State Management

```lua
-- scripts/game_state.lua (Runs 10x faster with LuaJIT)

-- Game state that's called every frame
GameState = {
    active_entities = {},
    physics_queue = {},
    input_buffer = {}
}

-- Called from C++ every frame
function GameState.update(deltaTime)
    -- Local caching for JIT optimization
    local entities = GameState.active_entities
    local phys_queue = GameState.physics_queue
    
    -- GOOD: Direct iteration with local variable
    for i = 1, #entities do
        local e = entities[i]
        if e.active then
            e:update(deltaTime)
        end
    end
    
    -- Process physics (mathematical operations = best JIT gain)
    GameState:process_physics(deltaTime, phys_queue)
    
    -- Handle input
    GameState:process_input()
end

function GameState:process_physics(dt, queue)
    -- This loop runs 15-20x faster in LuaJIT!
    for i = 1, #queue do
        local body = queue[i]
        local ax, ay = body.ax, body.ay
        local vx, vy = body.vx, body.vy
        
        -- Pure math operations = optimal JIT performance
        body.vx = vx + ax * dt
        body.vy = vy + ay * dt
        body.x = body.x + vx * dt
        body.y = body.y + vy * dt
    end
end

function GameState:process_input()
    local input = GameState.input_buffer
    -- Process accumulated input
    for i = 1, #input do
        local cmd = input[i]
        GameState:execute_command(cmd)
    end
    table.clear(GameState.input_buffer)  -- Reuse table
end
```

## Example 3: Performance-Critical AI Script

```lua
-- scripts/ai_behavior.lua (10x speedup from LuaJIT)

local AIBehavior = {}

-- Pathfinding heavily benefits from JIT (loops + math)
function AIBehavior.find_path(agent, goal, grid)
    local path = {}
    local current = {x = agent.x, y = agent.y}
    local max_steps = 1000
    local step_count = 0
    
    -- This loop runs 10-15x faster with LuaJIT
    while step_count < max_steps do
        local dx = goal.x - current.x
        local dy = goal.y - current.y
        local dist_sq = dx * dx + dy * dy
        
        if dist_sq < 1.0 then
            break
        end
        
        -- Normalize direction
        local dist = math.sqrt(dist_sq)
        local nx = dx / dist
        local ny = dy / dist
        
        -- Move toward goal
        local new_x = current.x + nx * grid.step_size
        local new_y = current.y + ny * grid.step_size
        
        if grid:is_walkable(new_x, new_y) then
            current.x = new_x
            current.y = new_y
            table.insert(path, {x = new_x, y = new_y})
        else
            -- Find alternate path
            current = AIBehavior:find_alternate(current, goal, grid)
        end
        
        step_count = step_count + 1
    end
    
    return path
end

-- Behavior tree evaluation
function AIBehavior.evaluate_tree(agent, tree)
    local state = agent.behavior_state
    
    for i = 1, #tree.nodes do
        local node = tree.nodes[i]
        
        if node.condition(agent, state) then
            node.action(agent, state)
            break
        end
    end
end

return AIBehavior
```

## Example 4: Particle System Scripting

```lua
-- scripts/particle_logic.lua (Benefits from JIT)

ParticleLogic = {}

-- Called for each particle (can be 100k+ particles)
-- LuaJIT's 10x speedup is CRITICAL here
function ParticleLogic.update_particle(particle, dt)
    -- Local variables = JIT-friendly
    local x, y, z = particle.x, particle.y, particle.z
    local vx, vy, vz = particle.vx, particle.vy, particle.vz
    local ax, ay, az = particle.ax, particle.ay, 0 - 9.81  -- gravity
    
    -- Update velocity (math-heavy = best JIT case)
    vx = vx + ax * dt
    vy = vy + ay * dt
    vz = vz + az * dt
    
    -- Update position
    x = x + vx * dt
    y = y + vy * dt
    z = z + vz * dt
    
    -- Damping
    local damping = 0.99
    vx = vx * damping
    vy = vy * damping
    vz = vz * damping
    
    -- Write back (minimal table operations)
    particle.x = x
    particle.y = y
    particle.z = z
    particle.vx = vx
    particle.vy = vy
    particle.vz = vz
    
    -- Lifetime
    particle.lifetime = particle.lifetime - dt
    return particle.lifetime > 0  -- Continue alive?
end

-- Batch update (called once per frame)
function ParticleLogic.update_particles(particles, dt)
    local alive_count = 0
    
    -- This loop runs 15-20x faster in LuaJIT!
    for i = 1, #particles do
        local p = particles[i]
        if ParticleLogic.update_particle(p, dt) then
            alive_count = alive_count + 1
        end
    end
    
    return alive_count
end

return ParticleLogic
```

## Example 5: Performance Profiling & Monitoring

```cpp
// In ImGui overlay for development

void DebugUI::DrawScriptingPanel() {
    if (!ImGui::CollapsingHeader("LuaJIT Performance")) {
        return;
    }
    
    auto& jit = LuaJitScriptSystem::GetInstance();
    auto stats = jit.GetProfilingStats();
    auto memory = jit.GetMemoryStats();
    
    // Status
    ImGui::Checkbox("JIT Enabled", &jit_enabled);
    if (ImGui::IsItemEdited()) {
        jit.SetJitEnabled(jit_enabled);
    }
    
    // Performance metrics
    ImGui::SeparatorText("Performance");
    ImGui::Text("Total Execution: %.2f ms", stats.totalExecutionTime / 1000.0);
    ImGui::Text("Calls: %llu", stats.callCount);
    ImGui::Text("Avg per call: %.1f μs", stats.avgExecutionTime);
    
    // JIT stats
    ImGui::ProgressBar(stats.jitCoveragePercent / 100.0f);
    ImGui::SameLine();
    ImGui::Text("JIT Coverage: %.1f%%", stats.jitCoveragePercent);
    ImGui::Text("Compiled traces: %u", stats.activeTraces);
    ImGui::Text("JIT functions: %u", stats.jitCompiledFunctions);
    
    // Memory stats
    ImGui::SeparatorText("Memory");
    ImGui::Text("Current: %.1f KB", memory.currentUsage / 1024.0);
    ImGui::Text("Peak: %.1f KB", memory.peakUsage / 1024.0);
    ImGui::Text("Allocations: %u", memory.numAllocations);
    
    // Controls
    ImGui::SeparatorText("Controls");
    if (ImGui::Button("Force GC")) {
        jit.ForceGarbageCollection();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Stats")) {
        jit.ResetProfilingStats();
    }
    
    // Hot reload
    if (ImGui::Button("Reload Scripts (F5)")) {
        jit.HotReloadScript("scripts/game_logic.lua");
    }
}
```

## Example 6: Side-by-Side Comparison (Lua vs LuaJIT)

```cpp
#include "LuaScriptSystem.h"
#include "LuaJitScriptSystem.h"
#include <chrono>

void BenchmarkLuaVsLuaJIT() {
    const std::string test_code = R"(
        function benchmark()
            local sum = 0
            for i = 1, 10000000 do
                sum = sum + math.sin(i * 0.001)
            end
            return sum
        end
        benchmark()
    )";
    
    // Test Standard Lua
    {
        auto& lua = LuaScriptSystem::GetInstance();
        lua.Init();
        
        auto start = std::chrono::high_resolution_clock::now();
        lua.ExecuteString(test_code);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto lua_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Standard Lua: " << lua_time.count() << " ms\n";
        
        lua.Shutdown();
    }
    
    // Test LuaJIT
    {
        auto& jit = LuaJitScriptSystem::GetInstance();
        jit.Init();
        jit.SetProfilingEnabled(true);
        
        // Warmup for JIT
        jit.ExecuteString(test_code);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        auto start = std::chrono::high_resolution_clock::now();
        jit.ExecuteString(test_code);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto jit_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        auto stats = jit.GetProfilingStats();
        
        std::cout << "LuaJIT: " << jit_time.count() << " ms\n";
        std::cout << "JIT Coverage: " << stats.jitCoveragePercent << "%\n";
        std::cout << "Speedup: " << (lua_time.count() / (float)jit_time.count()) << "x\n";
        
        jit.Shutdown();
    }
}
```

## Example 7: Optimal Lua Script Structure

```lua
-- scripts/optimal_structure.lua
-- This structure maximizes LuaJIT performance

-- Global state initialized once
local GameLogic = {
    player = nil,
    enemies = {},
    particles = {},
    
    -- Reusable tables for calculations
    temp_vector = {x = 0, y = 0, z = 0},
    temp_matrix = {},
    
    -- Constants (JIT can inline these)
    GRAVITY = -9.81,
    MAX_ENTITIES = 1000,
    FRAME_TIME = 0.016
}

-- Physics update (math-heavy, best JIT performance)
function GameLogic:update_physics(dt)
    for i = 1, #self.enemies do
        local e = self.enemies[i]
        
        -- Local variables for cache efficiency
        local vx, vy = e.vx, e.vy
        local x, y = e.x, e.y
        
        -- Pure arithmetic operations (JIT loves this!)
        vx = vx * 0.99  -- Damping
        vy = vy + self.GRAVITY * dt
        
        x = x + vx * dt
        y = y + vy * dt
        
        -- Write back
        e.vx = vx
        e.vy = vy
        e.x = x
        e.y = y
    end
end

-- Logic update (mixed operations, good JIT performance)
function GameLogic:update_logic(dt)
    local player = self.player
    local enemies = self.enemies
    
    -- Cache frequently accessed values
    local px, py = player.x, player.y
    
    for i = 1, #enemies do
        local e = enemies[i]
        local dx = px - e.x
        local dy = py - e.y
        local dist_sq = dx * dx + dy * dy
        
        if dist_sq < 100 then  -- In range
            e:target_player(player)
        end
    end
end

-- Main update function (called from C++)
function update_game(dt)
    GameLogic:update_physics(dt)
    GameLogic:update_logic(dt)
end

return GameLogic
```

## Example 8: Checking LuaJIT Status

```cpp
// In your debug console or startup log

void LogScriptingInfo() {
    auto& registry = ScriptLanguageRegistry::GetInstance();
    auto& jit = LuaJitScriptSystem::GetInstance();
    
    std::cout << "=== Scripting System Info ===\n";
    std::cout << "LuaJIT Status: " << (jit.IsJitEnabled() ? "ENABLED ✓" : "DISABLED") << "\n";
    std::cout << "Expected Performance: " << (jit.IsJitEnabled() ? "10-20x faster" : "Standard speed") << "\n";
    
    auto memory = jit.GetMemoryStats();
    std::cout << "Memory Usage: " << (memory.currentUsage / 1024) << " KB\n";
    
    std::cout << "Available Languages:\n";
    for (auto lang : registry.GetSupportedLanguages()) {
        std::cout << "  - " << registry.GetLanguageName(lang) << "\n";
    }
}
```

## Key Takeaways

1. **LuaJIT is enabled by default** - No configuration needed
2. **Math-heavy code gets the most benefit** - 15-20x speedup
3. **Use local variables** - Critical for JIT optimization
4. **Pre-allocate tables** - Avoid dynamic growth in hot loops
5. **Profile your code** - Use `SetProfilingEnabled(true)`
6. **Hot-reload during dev** - Faster iteration
7. **Fallback available** - Switch to standard Lua if needed

See LUAJIT_INTEGRATION_GUIDE.md for detailed documentation.
