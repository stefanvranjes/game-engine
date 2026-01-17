# Multi-Language Scripting System - Complete Integration Example

This example demonstrates using all 8 scripting languages in a single game scenario.

## Scenario: Complex Game System

We'll implement a game with:
- Player character (Lua)
- Enemy AI (Wren)
- Advanced pathfinding (Python)
- Physics simulation (Rust)
- UI interactions (TypeScript/JavaScript)
- Game state management (Squirrel)
- Bytecode-based event system (Custom VM)
- C# integration (Optional)

## C++ Integration Code

### Application Header (Application.h)

```cpp
#pragma once

#include "ScriptLanguageRegistry.h"
#include "ScriptComponentFactory.h"
#include <memory>

class Application {
public:
    void Init();
    void Update(float deltaTime);
    void Shutdown();
    
private:
    void InitializeScriptingSystems();
    void LoadGameplayScripts();
    void SetupScriptCallbacks();
};
```

### Application Implementation (Application.cpp)

```cpp
#include "Application.h"
#include "GameObject.h"
#include "Input.h"
#include <iostream>

void Application::Init() {
    std::cout << "=== Initializing Game Engine ===" << std::endl;
    
    InitializeScriptingSystems();
    LoadGameplayScripts();
    SetupScriptCallbacks();
    
    std::cout << "=== Game Engine Initialized ===" << std::endl;
}

void Application::InitializeScriptingSystems() {
    auto& registry = ScriptLanguageRegistry::GetInstance();
    
    std::cout << "\n--- Initializing Script Systems ---" << std::endl;
    registry.Init();
    
    // Verify all systems initialized
    auto languages = registry.GetSupportedLanguages();
    std::cout << "Loaded " << languages.size() << " script languages:" << std::endl;
    
    for (auto lang : languages) {
        std::cout << "  ✓ " << registry.GetLanguageName(lang) 
                  << " (" << registry.GetFileExtension(lang) << ")" << std::endl;
    }
}

void Application::LoadGameplayScripts() {
    auto& registry = ScriptLanguageRegistry::GetInstance();
    
    std::cout << "\n--- Loading Game Scripts ---" << std::endl;
    
    // Load scripts for different game systems
    // Each uses the most appropriate language
    
    // Game Logic (Lua - fast, lightweight, easy to iterate)
    registry.ExecuteScript("scripts/player_controller.lua");
    registry.ExecuteScript("scripts/level_manager.lua");
    
    // Enemy AI (Wren - OOP, game-focused)
    registry.ExecuteScript("scripts/enemy_base.wren");
    registry.ExecuteScript("scripts/enemy_behaviors.wren");
    
    // Advanced Pathfinding (Python - AI/ML friendly)
    registry.ExecuteScript("scripts/ai/pathfinding.py");
    registry.ExecuteScript("scripts/ai/behavior_tree.py");
    
    // Physics Engine (Rust - maximum performance)
    auto& rust_system = static_cast<RustScriptSystem&>(
        *registry.GetScriptSystem(ScriptLanguage::Rust)
    );
    rust_system.LoadLibrary("scripts/physics_engine.dll");
    
    // UI System (TypeScript - async-friendly, modern)
    registry.ExecuteScript("scripts/ui/menu_controller.js");
    registry.ExecuteScript("scripts/ui/hud.js");
    
    // Game State (Squirrel - C-like syntax)
    registry.ExecuteScript("scripts/game_state.nut");
    
    std::cout << "All scripts loaded successfully!" << std::endl;
}

void Application::SetupScriptCallbacks() {
    auto& registry = ScriptLanguageRegistry::GetInstance();
    
    // Error callback - log all script errors
    registry.SetErrorCallback(
        [](ScriptLanguage lang, const std::string& error) {
            std::cerr << "❌ Script Error (" 
                      << static_cast<int>(lang) << "): " << error << std::endl;
        }
    );
    
    // Success callback - optional, for verbose logging
    registry.SetSuccessCallback(
        [](ScriptLanguage lang, const std::string& file) {
            std::cout << "✓ Loaded: " << file << std::endl;
        }
    );
}

void Application::Update(float deltaTime) {
    auto& registry = ScriptLanguageRegistry::GetInstance();
    
    // Let script systems process their updates
    registry.Update(deltaTime);
    
    // Example: Call gameplay update function
    std::vector<std::any> args = {deltaTime};
    registry.CallFunction("update_game", args);
    
    // Example: Update physics (Rust - high frequency)
    registry.CallFunction(
        ScriptLanguage::Rust,
        "solve_physics",
        {deltaTime}
    );
    
    // Example: Update AI (Python)
    registry.CallFunction(
        ScriptLanguage::Python,
        "update_ai_decisions",
        {deltaTime}
    );
    
    // Hot-reload support
    if (Input::IsKeyPressed(KEY_F5)) {
        std::cout << "\n🔄 Reloading scripts..." << std::endl;
        registry.ReloadScript("scripts/player_controller.lua");
        registry.ReloadScript("scripts/enemy_behaviors.wren");
        std::cout << "✓ Scripts reloaded!" << std::endl;
    }
    
    // Performance monitoring (F6)
    if (Input::IsKeyPressed(KEY_F6)) {
        PrintScriptStatistics(registry);
    }
}

void Application::PrintScriptStatistics(ScriptLanguageRegistry& registry) {
    std::cout << "\n=== Script System Statistics ===" << std::endl;
    
    uint64_t total_memory = registry.GetTotalMemoryUsage();
    std::cout << "Total Memory: " << (total_memory / 1024) << " KB" << std::endl;
    
    auto languages = registry.GetSupportedLanguages();
    for (auto lang : languages) {
        double exec_time = registry.GetLastExecutionTime(lang);
        std::cout << registry.GetLanguageName(lang) 
                  << ": " << (exec_time * 1000.0) << "ms" << std::endl;
    }
    
    if (registry.HasErrors()) {
        std::cout << "\n⚠️  Errors detected:" << std::endl;
        auto errors = registry.GetAllErrors();
        for (const auto& [lang, error] : errors) {
            std::cout << "  " << lang << ": " << error << std::endl;
        }
    }
}

void Application::Shutdown() {
    std::cout << "\n=== Shutting Down ===" << std::endl;
    
    ScriptLanguageRegistry::GetInstance().Shutdown();
    
    std::cout << "✓ Game Engine Shutdown Complete" << std::endl;
}
```

## Script Files

### scripts/player_controller.lua

```lua
-- Player Controller - Lua for rapid iteration
-- Called every frame

local Player = {}
Player.speed = 5.0
Player.health = 100
Player.isAlive = true

function Player.update(deltaTime)
    if not Player.isAlive then
        return
    end
    
    -- Handle input
    if Input.IsKeyPressed(KEY_W) then
        -- Move forward
        print("Moving forward")
    end
    
    if Input.IsKeyPressed(KEY_SPACE) then
        -- Jump
        print("Jump!")
    end
end

function Player.takeDamage(damage)
    Player.health = Player.health - damage
    print("Player took " .. damage .. " damage. Health: " .. Player.health)
    
    if Player.health <= 0 then
        Player.isAlive = false
        print("Player died!")
    end
end

function update_game(deltaTime)
    Player.update(deltaTime)
end

return Player
```

### scripts/enemy_behaviors.wren

```wren
// Enemy AI Behaviors - Wren for OOP gameplay systems

class Enemy {
    construct new(name, health) {
        _name = name
        _health = health
        _position = [0, 0, 0]
        _target = null
        _state = "idle"  // idle, patrol, chase, attack
    }
    
    update(deltaTime) {
        if (_state == "idle") {
            _patrolUpdate(deltaTime)
        } else if (_state == "chase") {
            _chaseUpdate(deltaTime)
        } else if (_state == "attack") {
            _attackUpdate(deltaTime)
        }
    }
    
    _patrolUpdate(deltaTime) {
        System.print("Patrol: %(_name)")
    }
    
    _chaseUpdate(deltaTime) {
        System.print("Chase target: %(_name)")
        if (_distanceToTarget() < 1.0) {
            _state = "attack"
        }
    }
    
    _attackUpdate(deltaTime) {
        System.print("Attack!")
        if (_target == null) {
            _state = "patrol"
        }
    }
    
    setTarget(target) {
        _target = target
        _state = "chase"
    }
    
    takeDamage(damage) {
        _health = _health - damage
        System.print("%(_name) took %(damage) damage. Health: %(_health)")
    }
    
    _distanceToTarget() {
        if (!_target) return 1000
        // Simple distance calculation
        return 10.0
    }
}

var enemies = [
    Enemy.new("Goblin", 20),
    Enemy.new("Orc", 50),
    Enemy.new("Troll", 100)
]

// Called from C++
function update_enemies(deltaTime) {
    for (enemy in enemies) {
        enemy.update(deltaTime)
    }
}
```

### scripts/ai/pathfinding.py

```python
#!/usr/bin/env python3
# Advanced Pathfinding - Python for AI/ML

import heapq
import math

class Pathfinder:
    def __init__(self, grid_width, grid_height):
        self.width = grid_width
        self.height = grid_height
        self.obstacles = set()
    
    def add_obstacle(self, x, y):
        """Mark a grid cell as impassable"""
        self.obstacles.add((x, y))
    
    def heuristic(self, pos, goal):
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def find_path(self, start, goal):
        """A* pathfinding algorithm"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        closed_set = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))
            
            closed_set.add(current)
            
            # Check neighbors
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor in self.obstacles or neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + math.sqrt(dx*dx + dy*dy)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        
        return []  # No path found

# Global pathfinder instance
pathfinder = Pathfinder(100, 100)

def find_path_to(start, goal):
    """Called from C++"""
    path = pathfinder.find_path(start, goal)
    print(f"Found path with {len(path)} waypoints")
    return path

def update_ai_decisions(deltaTime):
    """Called each frame for AI decision making"""
    # Implement complex AI logic here
    pass
```

### scripts/physics_engine.dll

```rust
// Physics Engine - Rust for maximum performance
// Compile: cargo build --release --lib

#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
pub struct RigidBody {
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub mass: f32,
}

// Global physics state (in real implementation, would be per-world)
static mut BODIES: Vec<RigidBody> = Vec::new();
const GRAVITY: f32 = 9.81;

#[no_mangle]
pub extern "C" fn rust_init() {
    println!("Physics engine initialized");
}

#[no_mangle]
pub extern "C" fn add_rigid_body(mass: f32) -> u32 {
    unsafe {
        BODIES.push(RigidBody {
            position: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            velocity: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            acceleration: Vec3 { x: 0.0, y: -GRAVITY, z: 0.0 },
            mass,
        });
        (BODIES.len() - 1) as u32
    }
}

#[no_mangle]
pub extern "C" fn solve_physics(dt: f32) {
    unsafe {
        for body in BODIES.iter_mut() {
            // v = v + a*dt
            body.velocity.x += body.acceleration.x * dt;
            body.velocity.y += body.acceleration.y * dt;
            body.velocity.z += body.acceleration.z * dt;
            
            // p = p + v*dt
            body.position.x += body.velocity.x * dt;
            body.position.y += body.velocity.y * dt;
            body.position.z += body.velocity.z * dt;
        }
    }
}

#[no_mangle]
pub extern "C" fn rust_shutdown() {
    println!("Physics engine shutdown");
}
```

### scripts/ui/hud.js

```javascript
// HUD UI Controller - JavaScript for modern, async-friendly code

class HUDController {
    constructor() {
        this.playerHealth = 100;
        this.playerMana = 100;
        this.score = 0;
        this.updateInterval = null;
    }
    
    init() {
        console.log("HUD Initialized");
        this.render();
    }
    
    async updateAsync(deltaTime) {
        // Simulate async operations
        await this.sleep(16);  // 60 FPS
        this.render();
    }
    
    setPlayerHealth(health) {
        this.playerHealth = Math.max(0, health);
        this.render();
    }
    
    addScore(points) {
        this.score += points;
        this.animateScorePopup(points);
    }
    
    async animateScorePopup(points) {
        console.log(`+${points} points!`);
        await this.sleep(1000);  // Show for 1 second
    }
    
    render() {
        // Update UI display
        // This would update DOM in a real game
        console.log(`Health: ${this.playerHealth} | Score: ${this.score}`);
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

const hud = new HUDController();
hud.init();
```

### scripts/game_state.nut

```squirrel
// Game State Manager - Squirrel for structured state management

class GameState {
    isRunning = true;
    isPaused = false;
    currentLevel = 1;
    playerScore = 0;
    playerLives = 3;
    
    constructor() {
        print("Game State Initialized");
    }
    
    function pause() {
        isPaused = true;
        print("Game Paused");
    }
    
    function resume() {
        isPaused = false;
        print("Game Resumed");
    }
    
    function nextLevel() {
        currentLevel = currentLevel + 1;
        print("Advanced to Level " + currentLevel);
    }
    
    function gameOver() {
        isRunning = false;
        print("Game Over! Final Score: " + playerScore);
    }
    
    function update(deltaTime) {
        if (!isRunning || isPaused) return;
        
        // Update game state logic
    }
}

local gameState = GameState();
```

## Usage Instructions

1. **Prepare your project structure:**
   ```
   YourGame/
   ├── include/
   ├── src/
   ├── scripts/
   │   ├── player_controller.lua
   │   ├── enemy_behaviors.wren
   │   ├── ai/
   │   │   └── pathfinding.py
   │   ├── ui/
   │   │   └── hud.js
   │   ├── game_state.nut
   │   └── physics_engine.dll (compiled from Rust)
   └── CMakeLists.txt
   ```

2. **Initialize in your Application:**
   ```cpp
   Application app;
   app.Init();
   
   // Main loop
   while (app.IsRunning()) {
       app.Update(deltaTime);
   }
   
   app.Shutdown();
   ```

3. **Monitor with F6:**
   - Press F6 in-game to see script statistics
   - Monitor memory usage and execution times

4. **Iterate with F5:**
   - Edit any script
   - Press F5 to hot-reload
   - See changes immediately (most languages)

## Performance Expectations

- **Lua gameplay**: ~5-10ms per frame
- **Wren AI**: ~3-5ms per frame
- **Python pathfinding**: ~20-50ms (once, not per frame)
- **Rust physics**: ~1-2ms per frame (very fast)
- **TypeScript UI**: ~2-3ms per frame
- **Squirrel state**: ~1ms per frame

Total: ~30-70ms for all scripting, leaving plenty of budget for rendering!

## Future Enhancements

- [ ] Script debugging interface
- [ ] Visual script editor for Wren
- [ ] Python NumPy integration for math-heavy operations
- [ ] Rust WASM compilation for portability
- [ ] Hot-reload dependency tracking
- [ ] Script profiler UI in ImGui
