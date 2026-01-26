# AngelScript Integration Guide

## Overview

**AngelScript** is a lightweight, flexible scripting language designed specifically for use in games and performance-critical applications. It features C++-like syntax, making it intuitive for game developers while maintaining excellent performance and a small footprint.

### Key Advantages

| Feature | Benefit |
|---------|---------|
| **C++-like Syntax** | Familiar to game developers, easy to learn |
| **Lightweight** | ~2-3MB engine footprint, minimal dependencies |
| **High Performance** | 2-5x slower than native C++, comparable to Lua |
| **Hot-Reload** | Rapid development iteration without engine restart |
| **Static Typing** | Type safety with inference support |
| **Object-Oriented** | Classes, inheritance, polymorphism support |
| **Bytecode Compilation** | Compile once, run multiple times efficiently |
| **Simple Binding** | Easy C++ integration with minimal boilerplate |

### Performance Comparison

| Metric | AngelScript | Lua | LuaJIT | C++ |
|--------|-------------|-----|--------|-----|
| Startup Time | 5-10ms | 1ms | 1.5ms | N/A |
| Execution (warm) | 1x baseline | 1x baseline | 10-20x faster | 500-1000x faster |
| Memory (engine) | ~2-3MB | ~500KB | ~300KB | N/A |
| Best for | Game logic, AI | Config, light logic | Hot loops | Everything |

---

## Quick Start

### 1. Build with AngelScript Support

AngelScript is **enabled by default**. To explicitly enable/disable it:

```bash
cmake -B build -DENABLE_ANGELSCRIPT=ON
cmake --build build
```

### 2. Basic Usage

```cpp
#include "AngelScriptSystem.h"
#include "ScriptLanguageRegistry.h"

// Via registry (recommended)
auto& registry = ScriptLanguageRegistry::GetInstance();
registry.Init();
registry.ExecuteScript("scripts/game_logic.as", ScriptLanguage::AngelScript);

// Or direct access
auto& angel = AngelScriptSystem::GetInstance();
angel.Init();
angel.RunScript("scripts/player.as");
angel.Shutdown();
```

### 3. Call Functions from C++

```cpp
auto& angel = AngelScriptSystem::GetInstance();

// Call a global function
std::vector<std::any> args = {player, 0.016f};  // deltaTime
angel.CallFunction("UpdatePlayer", args);

// Check for errors
if (angel.HasErrors()) {
    std::cerr << "Error: " << angel.GetLastError() << std::endl;
}
```

---

## AngelScript Language Basics

### Simple Script Example

Create `scripts/player.as`:

```angelscript
// Define a player class
class Player {
    float speed;
    Vec3 position;
    float health;
    
    // Constructor
    Player(float spd) {
        speed = spd;
        health = 100.0f;
    }
    
    // Update method
    void Update(float dt) {
        // Simulate movement
        position.x += speed * dt;
    }
    
    // Handle damage
    void TakeDamage(float damage) {
        health -= damage;
        if (health < 0) {
            health = 0;
        }
    }
}

// Global function callable from C++
Player@ CreatePlayer(float initialSpeed) {
    Player p(initialSpeed);
    return @p;
}

// Update function (called each frame)
void UpdateGameLogic(float dt) {
    // Game logic here
}
```

### Type System

AngelScript supports:

```angelscript
// Basic types
int count = 5;
float speed = 10.5f;
double precision = 3.14159;
bool isActive = true;
string message = "Hello, AngelScript!";

// Arrays
int[] numbers = {1, 2, 3, 4, 5};
string[] names = {"Player", "Enemy", "NPC"};

// Object references (with @ operator)
Player@ player = @CreatePlayer(5.0f);
```

### Functions and Methods

```angelscript
// Global function
void PrintMessage(string msg) {
    print("Message: " + msg);
}

// Function with return value
int Add(int a, int b) {
    return a + b;
}

// Variadic function
void LogMultiple(string[] messages) {
    for (uint i = 0; i < messages.length(); i++) {
        print(messages[i]);
    }
}

// Method in a class
class Enemy {
    float attackPower;
    
    Enemy(float power) {
        attackPower = power;
    }
    
    float GetDamage() {
        return attackPower;
    }
}
```

### Control Structures

```angelscript
// If/else
if (health > 50) {
    print("Healthy");
} else if (health > 0) {
    print("Damaged");
} else {
    print("Dead");
}

// Loops
for (int i = 0; i < 10; i++) {
    print("Count: " + i);
}

while (isRunning) {
    Update();
}

// Switch statement
switch (state) {
    case 1:
        print("State 1");
        break;
    case 2:
        print("State 2");
        break;
    default:
        print("Unknown state");
}
```

---

## Integration with Game Engine

### Module System

Organize scripts into modules for better management:

```cpp
auto& angel = AngelScriptSystem::GetInstance();

// Create modules
angel.CreateModule("GameLogic");
angel.CreateModule("UI");

// Load scripts into specific modules
angel.SetActiveModule("GameLogic");
angel.RunScript("scripts/game_logic.as");

angel.SetActiveModule("UI");
angel.RunScript("scripts/ui_controller.as");

// Call functions from specific modules
auto& registry = ScriptLanguageRegistry::GetInstance();
registry.ExecuteScript("scripts/player_behavior.as", ScriptLanguage::AngelScript);
```

### Type Registration

Register C++ types to be used from AngelScript:

```cpp
void AngelScriptSystem::RegisterGameObjectTypes() {
    if (!m_Engine) return;

    // Register Vec3
    m_Engine->RegisterObjectType("Vec3", sizeof(glm::vec3), 
        asOBJ_VALUE | asOBJ_POD | asOBJ_APP_CLASS_ALLINTS);
    m_Engine->RegisterObjectProperty("Vec3", "float x", asOFFSET(glm::vec3, x));
    m_Engine->RegisterObjectProperty("Vec3", "float y", asOFFSET(glm::vec3, y));
    m_Engine->RegisterObjectProperty("Vec3", "float z", asOFFSET(glm::vec3, z));

    // Register Transform
    m_Engine->RegisterObjectType("Transform", 0, asOBJ_REF);
    m_Engine->RegisterObjectMethod("Transform", 
        "void SetPosition(const Vec3&in)", 
        asMETHOD(Transform, SetPosition), asCALL_THISCALL);
    
    // Register GameObject
    m_Engine->RegisterObjectType("GameObject", 0, asOBJ_REF);
    // ... register GameObject methods
}
```

### Hot-Reload for Development

Enable rapid iteration during development:

```cpp
auto& angel = AngelScriptSystem::GetInstance();
angel.SetDebugEnabled(true);

// During development
while (gameIsRunning) {
    // Monitor for file changes and hot-reload
    if (FileHasChanged("scripts/player.as")) {
        angel.ReloadScript("scripts/player.as");
    }
    
    // Game loop
    Update();
}
```

---

## Advanced Features

### Error Handling

```cpp
auto& angel = AngelScriptSystem::GetInstance();

// Set custom error handler
angel.SetErrorHandler([](const std::string& error) {
    std::cerr << "AngelScript Error: " << error << std::endl;
});

// Check for errors after execution
if (angel.HasErrors()) {
    std::cout << angel.GetLastError() << std::endl;
}
```

### Print Handler

```cpp
auto& angel = AngelScriptSystem::GetInstance();

// Capture script output
angel.SetPrintHandler([](const std::string& message) {
    std::cout << "[AngelScript] " << message << std::endl;
});

// Now script print() calls are captured
```

### Optimization

```cpp
auto& angel = AngelScriptSystem::GetInstance();

// Enable optimization during compilation
angel.SetOptimizationEnabled(true);

// This improves execution speed at the cost of longer compile time
```

### Memory Management

```cpp
auto& angel = AngelScriptSystem::GetInstance();

// Force garbage collection
angel.ForceGarbageCollection();

// Get memory statistics
auto stats = angel.GetCompileStats();
std::cout << "Total functions: " << stats.totalFunctions << std::endl;
std::cout << "Total classes: " << stats.totalClasses << std::endl;
std::cout << "Memory used: " << stats.memoryUsed << " bytes" << std::endl;

// Clear all scripts
angel.ClearState();
```

### Performance Profiling

```cpp
auto& angel = AngelScriptSystem::GetInstance();

// Get execution time
auto start = std::chrono::high_resolution_clock::now();
angel.CallFunction("HeavyComputation", {});
auto end = std::chrono::high_resolution_clock::now();

auto elapsed = std::chrono::duration<double>(end - start).count();
std::cout << "Execution time: " << elapsed * 1000 << " ms" << std::endl;
std::cout << "Last execution: " << angel.GetLastExecutionTime() * 1000 << " ms" << std::endl;
```

---

## Complete Example: Game Loop Integration

### Application.h

```cpp
#include "AngelScriptSystem.h"
#include <memory>

class Application {
private:
    AngelScriptSystem* m_AngelScript = nullptr;
    bool m_Running = true;

public:
    void Init() {
        auto& registry = ScriptLanguageRegistry::GetInstance();
        registry.Init();
        
        m_AngelScript = dynamic_cast<AngelScriptSystem*>(
            registry.GetScriptSystem(ScriptLanguage::AngelScript)
        );
        
        // Load game scripts
        m_AngelScript->RunScript("scripts/game_init.as");
        m_AngelScript->RunScript("scripts/player.as");
        m_AngelScript->RunScript("scripts/enemies.as");
        
        // Call initialization function
        std::vector<std::any> args;
        m_AngelScript->CallFunction("OnGameStart", args);
    }
    
    void Update(float deltaTime) {
        if (!m_AngelScript) return;
        
        // Call update function from scripts
        std::vector<std::any> args = {deltaTime};
        m_AngelScript->CallFunction("OnUpdate", args);
    }
    
    void Shutdown() {
        if (m_AngelScript) {
            std::vector<std::any> args;
            m_AngelScript->CallFunction("OnGameEnd", args);
        }
        
        auto& registry = ScriptLanguageRegistry::GetInstance();
        registry.Shutdown();
    }
};
```

### scripts/game_init.as

```angelscript
// Global game state
class GameState {
    float timeElapsed = 0.0f;
    int score = 0;
    bool isPaused = false;
}

GameState@ g_GameState = @GameState();

void OnGameStart() {
    print("Game Started!");
    g_GameState.timeElapsed = 0.0f;
    g_GameState.score = 0;
}

void OnUpdate(float dt) {
    if (!g_GameState.isPaused) {
        g_GameState.timeElapsed += dt;
    }
}

void OnGameEnd() {
    print("Game Ended! Final Score: " + g_GameState.score);
}

int GetScore() {
    return g_GameState.score;
}

void AddScore(int points) {
    g_GameState.score += points;
}
```

### scripts/player.as

```angelscript
class Player {
    float health = 100.0f;
    float speed = 10.0f;
    Vec3 position = Vec3(0, 0, 0);
    
    Player(float initialHealth, float initialSpeed) {
        health = initialHealth;
        speed = initialSpeed;
    }
    
    void Update(float dt) {
        // Movement logic
        position.x += speed * dt;
    }
    
    void TakeDamage(float damage) {
        health -= damage;
        if (health < 0) health = 0;
        print("Player health: " + health);
    }
    
    float GetHealth() {
        return health;
    }
    
    bool IsAlive() {
        return health > 0;
    }
}

Player@ g_Player = @Player(100.0f, 5.0f);

void UpdatePlayer(float dt) {
    if (g_Player !is null) {
        g_Player.Update(dt);
    }
}
```

---

## Common Patterns

### Event System

```angelscript
// Define event callback type
typedef void EventCallback();

class EventSystem {
    array<EventCallback@> listeners;
    
    void Subscribe(EventCallback@ callback) {
        listeners.insertLast(callback);
    }
    
    void Emit() {
        for (uint i = 0; i < listeners.length(); i++) {
            if (listeners[i] !is null) {
                listeners[i]();
            }
        }
    }
}

EventSystem@ g_Events = @EventSystem();

void OnPlayerDied() {
    print("Player died!");
}

void OnPlayerSpawned() {
    print("Player spawned!");
}
```

### State Machine

```angelscript
enum PlayerState {
    Idle,
    Moving,
    Attacking,
    Dead
}

class PlayerStateMachine {
    PlayerState currentState = Idle;
    
    void ChangeState(PlayerState newState) {
        if (currentState == newState) return;
        
        OnStateExit(currentState);
        currentState = newState;
        OnStateEnter(currentState);
    }
    
    void OnStateEnter(PlayerState state) {
        if (state == Idle) {
            print("Entering Idle state");
        } else if (state == Moving) {
            print("Entering Moving state");
        }
    }
    
    void OnStateExit(PlayerState state) {
        if (state == Attacking) {
            print("Exiting Attack state");
        }
    }
    
    void Update(float dt) {
        if (currentState == Moving) {
            // Update position
        }
    }
}
```

### Behavior Tree Node

```angelscript
enum NodeStatus {
    Success,
    Failure,
    Running
}

class BTNode {
    string name;
    
    BTNode(string nodeName) {
        name = nodeName;
    }
    
    NodeStatus Execute(float dt) {
        print("Executing: " + name);
        return Success;
    }
}

class BTSequence : BTNode {
    array<BTNode@> children;
    
    BTSequence(string seqName) : BTNode(seqName) {}
    
    void AddChild(BTNode@ child) {
        children.insertLast(child);
    }
    
    NodeStatus Execute(float dt) {
        for (uint i = 0; i < children.length(); i++) {
            NodeStatus status = children[i].Execute(dt);
            if (status != Success) {
                return status;
            }
        }
        return Success;
    }
}
```

---

## Troubleshooting

### "AngelScript not initialized"

Ensure you call `Init()` before using:

```cpp
auto& angel = AngelScriptSystem::GetInstance();
angel.Init();  // Don't forget!
angel.RunScript("script.as");
```

### "Function not found"

Verify the function is defined in a loaded script:

```cpp
if (angel.HasFunction("MyFunction")) {
    angel.CallFunction("MyFunction", {});
}
```

### Compilation errors

Check the error message and verify script syntax:

```cpp
angel.SetErrorHandler([](const std::string& error) {
    std::cout << "Compile error: " << error << std::endl;
});
```

### Performance issues

- Enable optimization: `angel.SetOptimizationEnabled(true)`
- Use profiling to identify bottlenecks
- Consider moving hot loops to C++
- Use bytecode caching for frequently-run scripts

### Memory leaks

Enable garbage collection:

```cpp
angel.ForceGarbageCollection();  // After gameplay session
```

---

## API Reference

### Core Methods

```cpp
// Lifecycle
void Init();                    // Initialize engine
void Shutdown();                // Shutdown engine
void Update(float deltaTime);   // Update (called per frame)

// Script execution
bool RunScript(const std::string& filepath);
bool ExecuteString(const std::string& source);

// Function calling
std::any CallFunction(const std::string& functionName,
                     const std::vector<std::any>& args);
std::any CallMethod(const std::string& objectName,
                   const std::string& methodName,
                   const std::vector<std::any>& args);

// Module management
bool CreateModule(const std::string& moduleName);
bool BuildModule(const std::string& moduleName);
void DiscardModule(const std::string& moduleName);
void SetActiveModule(const std::string& moduleName);
asIScriptModule* GetModule(const std::string& moduleName) const;

// Variables
void SetGlobalVariable(const std::string& varName, const std::any& value);
std::string GetGlobalVariable(const std::string& varName) const;

// Hot-reload
bool SupportsHotReload() const { return true; }
void ReloadScript(const std::string& filepath);

// Error handling
bool HasErrors() const;
std::string GetLastError() const;
void SetErrorHandler(MessageHandler handler);

// Output
void SetPrintHandler(MessageHandler handler);

// Optimization
void SetOptimizationEnabled(bool enabled);
void SetDebugEnabled(bool enabled);
void ForceGarbageCollection();
void ClearState();

// Statistics
CompileStats GetCompileStats() const;
uint64_t GetMemoryUsage() const;
double GetLastExecutionTime() const;
```

### Metadata

```cpp
ScriptLanguage GetLanguage() const;        // Returns AngelScript
ScriptExecutionMode GetExecutionMode() const;  // Returns Bytecode
std::string GetLanguageName() const;       // Returns "AngelScript"
std::string GetFileExtension() const;      // Returns ".as"

bool HasType(const std::string& typeName) const;
bool HasFunction(const std::string& functionName) const;
```

---

## Performance Tips

1. **Pre-compile scripts**: Compile during initialization, not at runtime
2. **Use modules**: Organize code into modules for better management
3. **Enable optimization**: Trade compilation time for faster execution
4. **Limit function calls**: Batch operations where possible
5. **Monitor memory**: Use `GetMemoryUsage()` to track script memory
6. **Profile execution**: Use `GetLastExecutionTime()` to identify bottlenecks
7. **Cache frequently called functions**: Store function pointers if calling repeatedly
8. **Use static typing**: Avoid dynamic types where possible for better performance

---

## File Structure

```
include/
  ├─ AngelScriptSystem.h       # Header definition
  └─ IScriptSystem.h            # Base interface
  
src/
  ├─ AngelScriptSystem.cpp      # Implementation
  └─ ScriptLanguageRegistry.cpp # Registry integration
  
scripts/
  ├─ game_logic.as
  ├─ player.as
  ├─ enemies.as
  └─ ui.as
```

---

## Build Configuration

### Enable/Disable AngelScript

```cmake
# In CMakeLists.txt
option(ENABLE_ANGELSCRIPT "Enable AngelScript language support" ON)

# Or from command line
cmake -B build -DENABLE_ANGELSCRIPT=ON
```

### Link Against AngelScript

The system automatically links AngelScript when `ENABLE_ANGELSCRIPT=ON`.

---

## Related Documentation

- [SCRIPTING_QUICK_START.md](SCRIPTING_QUICK_START.md) - Multi-language overview
- [MULTI_LANGUAGE_SCRIPTING_GUIDE.md](MULTI_LANGUAGE_SCRIPTING_GUIDE.md) - Language comparisons
- [LUAJIT_INTEGRATION_GUIDE.md](LUAJIT_INTEGRATION_GUIDE.md) - LuaJIT alternative
- [WREN_INTEGRATION_SETUP.md](WREN_INTEGRATION_SETUP.md) - Wren scripting

---

## Contributing

To extend AngelScript support:

1. Add new type registrations in `RegisterGameObjectTypes()`
2. Implement additional C++ bindings in `RegisterTypes()`
3. Add new callback handlers in `SetupCallbacks()`
4. Document new features in this guide

---

## License & Credits

- **AngelScript**: Written by Andreas Jönsson
- **Engine Integration**: Game Engine team
- **Documentation**: This guide

---

**Last Updated**: January 2026
