# AngelScript Quick Reference

## Enable AngelScript

```cmake
# Enabled by default in CMakeLists.txt
option(ENABLE_ANGELSCRIPT "Enable AngelScript language support" ON)
```

## Basic Usage

```cpp
#include "AngelScriptSystem.h"

auto& angel = AngelScriptSystem::GetInstance();
angel.Init();
angel.RunScript("script.as");
angel.Shutdown();
```

## Via ScriptLanguageRegistry

```cpp
auto& registry = ScriptLanguageRegistry::GetInstance();
registry.Init();
registry.ExecuteScript("script.as", ScriptLanguage::AngelScript);
```

## Call Functions

```cpp
// Simple call
angel.CallFunction("MyFunction", {});

// With arguments
std::vector<std::any> args = {player, deltaTime, score};
angel.CallFunction("UpdateGame", args);

// Check for errors
if (angel.HasErrors()) {
    std::cerr << angel.GetLastError() << std::endl;
}
```

## AngelScript Syntax

### Types

```angelscript
int x = 5;
float f = 3.14f;
bool b = true;
string s = "text";
int[] arr = {1, 2, 3};
```

### Classes

```angelscript
class Player {
    float health;
    Vec3 position;
    
    Player(float hp) {
        health = hp;
    }
    
    void Update(float dt) {
        // Logic here
    }
}
```

### Functions

```angelscript
void PrintMessage(string msg) {
    print(msg);
}

int Add(int a, int b) {
    return a + b;
}

Player@ CreatePlayer(float health) {
    return @Player(health);
}
```

### Control Flow

```angelscript
if (condition) {
    // ...
} else {
    // ...
}

for (int i = 0; i < 10; i++) {
    print(i);
}

while (isRunning) {
    Update();
}

switch (state) {
    case 1: DoSomething(); break;
    case 2: DoOtherThing(); break;
}
```

## Module Management

```cpp
angel.CreateModule("GameLogic");
angel.SetActiveModule("GameLogic");
angel.RunScript("scripts/logic.as");
angel.BuildModule("GameLogic");
angel.DiscardModule("GameLogic");
```

## Hot-Reload

```cpp
// Enable debug mode
angel.SetDebugEnabled(true);

// Reload script during development
angel.ReloadScript("scripts/player.as");
```

## Error Handling

```cpp
// Set custom error handler
angel.SetErrorHandler([](const std::string& error) {
    std::cerr << "Error: " << error << std::endl;
});

// Set print handler
angel.SetPrintHandler([](const std::string& msg) {
    std::cout << "[Script] " << msg << std::endl;
});
```

## Performance

```cpp
// Enable optimization
angel.SetOptimizationEnabled(true);

// Get execution time
auto execTime = angel.GetLastExecutionTime();
std::cout << "Execution: " << execTime * 1000 << " ms" << std::endl;

// Force garbage collection
angel.ForceGarbageCollection();

// Get statistics
auto stats = angel.GetCompileStats();
std::cout << "Functions: " << stats.totalFunctions << std::endl;
```

## File Extension

Scripts use `.as` extension:
- `scripts/player.as`
- `scripts/enemies.as`
- `scripts/ui.as`

## Engine Objects (When Registered)

```angelscript
Vec3 pos = Vec3(1, 2, 3);
Transform@ transform = GetTransform();
GameObject@ obj = GetGameObject();
```

## Common Patterns

### Event System

```angelscript
typedef void EventCallback();

class EventSystem {
    array<EventCallback@> listeners;
    
    void Subscribe(EventCallback@ cb) {
        listeners.insertLast(cb);
    }
    
    void Emit() {
        for (uint i = 0; i < listeners.length(); i++) {
            listeners[i]();
        }
    }
}
```

### State Machine

```angelscript
enum State { Idle, Moving, Attacking }

class StateMachine {
    State current = Idle;
    
    void ChangeState(State newState) {
        current = newState;
    }
    
    void Update(float dt) {
        if (current == Moving) {
            // ...
        }
    }
}
```

## Debugging

```cpp
// Enable debug output
angel.SetDebugEnabled(true);

// Check if function exists
if (angel.HasFunction("UpdatePlayer")) {
    angel.CallFunction("UpdatePlayer", {});
}

// Check for errors
if (angel.HasErrors()) {
    std::cout << "Last error: " << angel.GetLastError() << std::endl;
}
```

## Key Differences from C++

| Feature | AngelScript | C++ |
|---------|-------------|-----|
| Type inference | Yes | Limited |
| Dynamic arrays | `int[] arr` | `std::vector<int>` |
| String type | Built-in `string` | `std::string` |
| Garbage collection | Automatic | Manual |
| Null safety | `@` operator for references | Pointers |
| Performance | 2-5x slower | Baseline |

## Memory Management

```cpp
// Automatic garbage collection
angel.ForceGarbageCollection();

// Clear all scripts
angel.ClearState();

// Get memory usage
auto memUsage = angel.GetMemoryUsage();
std::cout << "Memory: " << memUsage << " bytes" << std::endl;
```

## Tips & Tricks

1. **Use `@` for object references**: `Player@ p = @CreatePlayer();`
2. **Use `!is null` to check references**: `if (player !is null) { ... }`
3. **Arrays are dynamic**: No need to specify size
4. **Functions can be overloaded**: Same name, different signatures
5. **Use `const` for immutability**: `const float PI = 3.14159f;`
6. **Pass by reference**: `void Modify(int& ref) { ref = 10; }`

## API Quick Lookup

### Class Methods
```
Init() / Shutdown() / Update(dt)
RunScript(path) / ExecuteString(source)
CallFunction(name, args)
CreateModule(name) / SetActiveModule(name)
ForceGarbageCollection() / ClearState()
HasFunction(name) / HasErrors()
GetLastError() / GetLastExecutionTime()
```

### Configuration
```
SetDebugEnabled(bool)
SetOptimizationEnabled(bool)
SetPrintHandler(callback)
SetErrorHandler(callback)
```

---

**For complete examples, see ANGELSCRIPT_INTEGRATION_GUIDE.md**
