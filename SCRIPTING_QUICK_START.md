# Multi-Language Scripting System - Quick Start Guide

## Quick Overview

The Game Engine now supports **8 different scripting languages** for gameplay logic development:

1. **Lua** (.lua) - Fast, lightweight, general purpose
2. **Wren** (.wren) - Object-oriented, designed for games
3. **Python** (.py) - Data science, AI, rapid development
4. **C#** (.cs) - .NET integration (requires Mono)
5. **TypeScript/JavaScript** (.js/.ts) - Modern web-like scripting
6. **Rust** (.dll/.so) - Native compiled, maximum performance
7. **Squirrel** (.nut) - C-like syntax, game-focused
8. **Custom Bytecode** (.asm/.bc) - Lightweight VM

## 5-Minute Setup

### 1. Include the Registry
```cpp
#include "ScriptLanguageRegistry.h"
#include "ScriptComponentFactory.h"
```

### 2. Initialize in Your Application
```cpp
// In Application::Init()
ScriptLanguageRegistry::GetInstance().Init();

// Set up error callback for debugging
ScriptLanguageRegistry::GetInstance().SetErrorCallback(
    [](ScriptLanguage lang, const std::string& error) {
        std::cerr << "Script Error (" << static_cast<int>(lang) << "): " 
                  << error << std::endl;
    }
);
```

### 3. Create Script Components
```cpp
// Auto-detect language from file extension
auto playerScript = ScriptComponentFactory::CreateScriptComponent(
    "scripts/player.lua",
    playerGameObject
);

// Or explicitly specify language
auto npcScript = ScriptComponentFactory::CreateScriptComponent(
    "scripts/npc.wren",
    npcGameObject,
    ScriptLanguage::Wren
);
```

### 4. Execute Scripts
```cpp
auto& registry = ScriptLanguageRegistry::GetInstance();

// Load scripts directly
registry.ExecuteScript("scripts/gameplay.lua");
registry.ExecuteScript("scripts/ai.wren");
registry.ExecuteScript("scripts/ui.js");

// Execute code from string
registry.ExecuteString(R"(
    function test()
        print("Hello from Lua!")
    end
)", ScriptLanguage::Lua);
```

### 5. Call Script Functions
```cpp
// Call a function across all languages
std::vector<std::any> args = {gameObject, 0.016f};
registry.CallFunction("update", args);

// Or call in a specific language
registry.CallFunction(ScriptLanguage::Python, "update_ai", args);
```

### 6. Update in Game Loop
```cpp
// In Application::Update()
ScriptLanguageRegistry::GetInstance().Update(deltaTime);
```

### 7. Shutdown
```cpp
// In Application::Shutdown()
ScriptLanguageRegistry::GetInstance().Shutdown();
```

## Common Use Cases

### Use Case 1: Gameplay Logic (Recommend: Lua or Wren)
```cpp
// Load gameplay scripts
registry.ExecuteScript("scripts/player_controller.lua");
registry.ExecuteScript("scripts/enemy_ai.lua");
registry.ExecuteScript("scripts/level_manager.lua");

// Call each frame
registry.Update(deltaTime);
```

### Use Case 2: AI Systems (Recommend: Python)
```cpp
// Python excels at complex decision making
registry.ExecuteScript("scripts/ai/pathfinding.py");
registry.ExecuteScript("scripts/ai/behavior_tree.py");

// Call AI decision function
std::any decision = registry.CallFunction(
    ScriptLanguage::Python,
    "get_next_action",
    {npc_state, nearby_enemies}
);
```

### Use Case 3: Physics (Recommend: Rust)
```cpp
// Rust for performance-critical physics
RustScriptSystem& rust_sys = static_cast<RustScriptSystem&>(
    *registry.GetScriptSystem(ScriptLanguage::Rust)
);
rust_sys.LoadLibrary("physics_solver.dll");

// Call high-frequency physics updates
registry.CallFunction(ScriptLanguage::Rust, "solve_physics", {objects, dt});
```

### Use Case 4: Multi-Language Component
```cpp
// One GameObject with multiple language scripts
auto multiScript = ScriptComponentFactory::CreateMultiLanguageComponent(
    gameObject
);

// Mix languages as needed
multiScript->AddScript("scripts/input_handler.lua");      // Input
multiScript->AddScript("scripts/ai_pathfinding.py");      // AI
multiScript->AddScript("scripts/physics_solver.dll");     // Physics
multiScript->AddScript("scripts/ui_controller.js");       // UI
```

## Performance Tips

### For Maximum Performance
```cpp
// Use Rust for any hot loops
registry.ExecuteScript("scripts/core_simulation.dll");  // ~1.2x C++

// Use Lua/Wren for game logic that isn't frame-critical
registry.ExecuteScript("scripts/gameplay.lua");         // ~5x C++
```

### For Rapid Iteration
```cpp
// Use interpreted languages with hot-reload
registry.ExecuteScript("scripts/test_logic.lua");  // Fast iteration
registry.ExecuteScript("scripts/test_logic.wren"); // No compilation

// Press F5 to reload
if (input.IsKeyPressed(KEY_F5)) {
    registry.ReloadScript("scripts/test_logic.lua");
}
```

### For Data Science
```cpp
// Use Python for calculations, Rust/C++ for rendering
registry.ExecuteScript("scripts/procedural_generation.py");
registry.ExecuteScript("scripts/mesh_generation.dll");
```

## Hot-Reload Development

### Set up F5 Key for Reload
```cpp
void Application::Update(float deltaTime) {
    // ... game logic ...
    
    if (Input::IsKeyPressed(KEY_F5)) {
        std::cout << "Reloading scripts..." << std::endl;
        
        registry.ReloadScript("scripts/gameplay.lua");
        registry.ReloadScript("scripts/ai.wren");
        registry.ReloadScript("scripts/ui.js");
        
        std::cout << "Scripts reloaded!" << std::endl;
    }
}
```

## Example Scripts

### Lua Example: Player Controller
```lua
-- scripts/player.lua
Player = {}
Player.__index = Player

function Player.new(gameObject)
    local self = setmetatable({}, Player)
    self.gameObject = gameObject
    self.speed = 5.0
    self.health = 100
    return self
end

function Player:update(dt)
    local pos = self.gameObject:getPosition()
    if Input.IsKeyPressed(KEY_W) then
        pos.y = pos.y + self.speed * dt
    end
    self.gameObject:setPosition(pos)
end

function Player:takeDamage(damage)
    self.health = self.health - damage
    if self.health <= 0 then
        self:die()
    end
end

function Player:die()
    print("Player died!")
    self.gameObject:destroy()
end
```

### Wren Example: Enemy AI
```wren
// scripts/enemy.wren
class Enemy {
    construct new(gameObject) {
        _gameObject = gameObject
        _speed = 3.0
        _health = 50
        _target = null
    }
    
    update(dt) {
        if (_target) {
            var direction = _target - _gameObject.position
            var distance = direction.length()
            
            if (distance > 0.1) {
                _gameObject.position = _gameObject.position + direction.normalized() * _speed * dt
            }
        }
    }
    
    setTarget(target) {
        _target = target
    }
    
    takeDamage(damage) {
        _health = _health - damage
        System.print("Enemy health: %(_health)")
    }
}
```

### Rust Example: Physics Solver
```rust
// scripts/physics.rs - Compile with: cargo build --release
#[repr(C)]
pub struct Transform {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[no_mangle]
pub extern "C" fn solve_gravity(
    transforms: *mut Transform,
    velocities: *mut f32,
    count: u32,
    dt: f32
) {
    unsafe {
        for i in 0..count as usize {
            // Apply gravity
            velocities.add(i * 3 + 1).write(
                velocities.add(i * 3 + 1).read() - 9.81 * dt
            );
            
            // Update position
            transforms.add(i).y += velocities.add(i * 3 + 1).read() * dt;
        }
    }
}
```

## Troubleshooting

### Script Not Loading?
```cpp
// Check if language is ready
if (!registry.IsLanguageReady(ScriptLanguage::Lua)) {
    std::cout << "Lua system not initialized!" << std::endl;
}

// Check for errors
if (registry.HasErrors()) {
    auto errors = registry.GetAllErrors();
    for (const auto& [lang, error] : errors) {
        std::cout << lang << ": " << error << std::endl;
    }
}
```

### Performance Issues?
```cpp
// Monitor memory usage
uint64_t total_memory = registry.GetTotalMemoryUsage();
std::cout << "Script memory: " << total_memory / 1024 << " KB" << std::endl;

// Check execution times
double lua_time = registry.GetLastExecutionTime(ScriptLanguage::Lua);
double rust_time = registry.GetLastExecutionTime(ScriptLanguage::Rust);
```

### Language Not Available?
```cpp
// List all supported languages
auto languages = registry.GetSupportedLanguages();
for (auto lang : languages) {
    std::cout << registry.GetLanguageName(lang) << ": "
              << registry.GetFileExtension(lang) << std::endl;
}
```

## Next Steps

1. Create a `scripts/` directory in your project
2. Add a simple Lua test script: `scripts/test.lua`
3. Integrate `ScriptLanguageRegistry::GetInstance().Init()` in `Application::Init()`
4. Load the test script: `registry.ExecuteScript("scripts/test.lua")`
5. Call test functions from C++
6. Expand with other languages as needed

## Further Reading

- [MULTI_LANGUAGE_SCRIPTING_GUIDE.md](MULTI_LANGUAGE_SCRIPTING_GUIDE.md) - Comprehensive feature matrix
- [WREN_SCRIPTING_GUIDE.md](WREN_SCRIPTING_GUIDE.md) - Wren-specific examples
- [Header Files](include/ScriptLanguageRegistry.h) - API documentation
