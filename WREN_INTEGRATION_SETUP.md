# Wren Integration & Setup Guide

## Overview

This guide walks through integrating Wren scripting into your game engine for gameplay logic development.

## Architecture

```
┌─────────────────────────────────────┐
│   WrenScriptSystem (Singleton)      │
│   - Manages VM lifecycle            │
│   - Registers all bindings          │
│   - Loads/executes scripts          │
└──────────────┬──────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌────────────────┐  ┌──────────────────┐
│ WrenVM (Native)│  │ Script Bindings  │
│ - Execution    │  │ - GameObject     │
│ - Memory Mgmt  │  │ - Transform      │
│ - Compilation  │  │ - Physics        │
└────────────────┘  │ - Audio          │
                    │ - Particles      │
                    └──────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ WrenScriptComponent                 │
│ - Attached to GameObject            │
│ - Manages script lifecycle          │
│ - Calls init/update/destroy         │
└─────────────────────────────────────┘
```

## Installation Steps

### 1. Build System (CMake)

The Wren dependency is already added to CMakeLists.txt:

```cmake
# Fetch Wren from GitHub
FetchContent_Declare(
    wren
    GIT_REPOSITORY https://github.com/wren-lang/wren.git
    GIT_TAG 0.4.0
)

# Build as static library
file(GLOB WREN_SOURCES
    "${wren_SOURCE_DIR}/src/vm/*.c"
    "${wren_SOURCE_DIR}/src/compiler/*.c"
    "${wren_SOURCE_DIR}/src/optional/*.c"
)

add_library(wren STATIC ${WREN_SOURCES})
target_include_directories(wren PUBLIC "${wren_SOURCE_DIR}/src/include")
```

**Build the engine:**
```bash
build.bat              # Debug build
cmake --build build --config Release  # Release build
```

### 2. Initialize Wren in Application

In [Application.cpp](include/Application.h):

```cpp
#include "WrenScriptSystem.h"

void Application::Initialize() {
    // ... existing initialization code ...
    
    // Initialize Wren scripting system
    WrenScriptSystem::GetInstance().Init();
    
    // Set output handlers (optional)
    WrenScriptSystem::GetInstance().SetPrintHandler([](const std::string& msg) {
        std::cout << "[WREN] " << msg << std::endl;
    });
    
    WrenScriptSystem::GetInstance().SetErrorHandler([](const std::string& msg) {
        std::cerr << "[WREN ERROR] " << msg << std::endl;
    });
}
```

In destructor:

```cpp
Application::~Application() {
    WrenScriptSystem::GetInstance().Shutdown();
}
```

### 3. Attach Scripts to GameObjects

```cpp
#include "WrenScriptComponent.h"

// Create GameObject
auto playerObject = std::make_shared<GameObject>("Player");

// Create and attach script component
auto scriptComp = std::make_shared<WrenScriptComponent>(playerObject);
scriptComp->LoadScript("assets/scripts/player_behavior.wren");
playerObject->SetScriptComponent(scriptComp);

// Initialize script
scriptComp->Init();

// Add to scene
scene->AddGameObject(playerObject);
```

### 4. Update GameObjects in Main Loop

In [Application::Update()](src/Application.cpp):

```cpp
void Application::Update(float deltaTime) {
    // ... existing updates ...
    
    // Update all script components
    for (auto& gameObject : scene->GetGameObjects()) {
        auto scriptComp = gameObject->GetScriptComponent();
        if (scriptComp) {
            scriptComp->Update(deltaTime);
        }
    }
}
```

### 5. Handle GameObject Destruction

```cpp
void Application::DestroyGameObject(std::shared_ptr<GameObject> gameObject) {
    // Call script destroy callback
    auto scriptComp = gameObject->GetScriptComponent();
    if (scriptComp) {
        scriptComp->Destroy();
    }
    
    // Remove from scene
    scene->RemoveGameObject(gameObject);
}
```

## Creating Your First Script

### Step 1: Create Script File

Create `assets/scripts/my_first_script.wren`:

```wren
// Simple behavior demonstrating script lifecycle

class Behavior {
    construct new(gameObject) {
        _gameObject = gameObject
        _counter = 0
    }
    
    update(dt) {
        _counter = _counter + 1
        
        // Log every 60 frames (~1 second at 60 FPS)
        if (_counter % 60 == 0) {
            System.print("Frame: %(_counter)")
        }
    }
}

var _behavior = null

// Called once when script loads
construct init() {
    System.print("Script initialized")
    _behavior = Behavior.new(_gameObject)
}

// Called every frame with delta time
construct update(dt) {
    if (_behavior) {
        _behavior.update(dt)
    }
}

// Called when GameObject is destroyed
construct destroy() {
    System.print("Script destroyed")
}
```

### Step 2: Attach to GameObject

```cpp
auto obj = std::make_shared<GameObject>("TestObject");
auto scriptComp = std::make_shared<WrenScriptComponent>(obj);
scriptComp->LoadScript("assets/scripts/my_first_script.wren");
obj->SetScriptComponent(scriptComp);
scriptComp->Init();
```

### Step 3: Run and Check Console Output

```
[WREN] Script initialized
[WREN] Frame: 60
[WREN] Frame: 120
...
```

## Advanced Integration

### Hot-Reloading Scripts

For editor/debug builds:

```cpp
#ifdef DEBUG_BUILD
    if (Input.getKeyDown("F5")) {
        WrenScriptSystem::GetInstance().ReloadAll();
    }
#endif
```

Or reload specific component:

```cpp
scriptComponent->Reload();
```

### Custom Bindings

Register C++ functions callable from Wren:

```cpp
WrenScriptSystem::GetInstance().RegisterNativeMethod(
    "CustomClass",
    "myMethod",
    2,  // number of parameters
    [](WrenVM* vm) {
        // Implementation
        auto arg = wrenGetSlotString(vm, 1);
        wrenSetSlotDouble(vm, 0, 42.0);
    }
);
```

### Error Handling

Wren errors are caught and passed to error handler:

```cpp
WrenScriptSystem::GetInstance().SetErrorHandler([](const std::string& error) {
    std::cerr << "Script Error: " << error << std::endl;
    // Log to file, display in UI, etc.
});
```

### Script Debugging

Add print statements in scripts:

```wren
Debug.log("Player position: %(pos)")
Debug.warn("Health low!")
Debug.error("Invalid state")
```

Check console output:

```
[DEBUG] Player position: (10, 5, 20)
[WARN] Health low!
[ERROR] Invalid state
```

## File Organization

Recommended structure:

```
game-engine/
├── assets/
│   └── scripts/
│       ├── player/
│       │   ├── player_behavior.wren
│       │   ├── player_combat.wren
│       │   └── player_animation.wren
│       ├── enemy/
│       │   ├── enemy_ai.wren
│       │   ├── enemy_states.wren
│       │   └── boss_ai.wren
│       ├── items/
│       │   ├── collectible.wren
│       │   ├── weapon.wren
│       │   └── consumable.wren
│       ├── levels/
│       │   ├── level_01_manager.wren
│       │   ├── level_02_manager.wren
│       │   └── level_director.wren
│       ├── ui/
│       │   ├── hud.wren
│       │   └── menu_controller.wren
│       └── utils.wren
```

## Testing Scripts

### Unit Testing

Create `test_scripts.wren`:

```wren
class Tests {
    static run() {
        testVectorMath()
        testCollections()
        testMathUtils()
        System.print("All tests passed")
    }
    
    static testVectorMath() {
        var v1 = Vec3.new(1, 2, 3)
        var v2 = Vec3.new(4, 5, 6)
        var dist = (v1 - v2).magnitude
        System.print("Vector distance: %(dist)")
    }
    
    static testCollections() {
        var list = [1, 2, 3]
        System.print("List count: %(list.count)")
    }
    
    static testMathUtils() {
        var clipped = Mathf.clamp(10, 0, 5)
        System.print("Clamp result: %(clipped)")
    }
}

// Run tests on startup
Tests.run()
```

### Integration Testing

In C++ test code:

```cpp
#include "WrenScriptSystem.h"

TEST(WrenIntegration, SimpleScript) {
    WrenScriptSystem::GetInstance().Init();
    
    bool result = WrenScriptSystem::GetInstance().RunScript(
        "assets/scripts/test_scripts.wren"
    );
    
    EXPECT_TRUE(result);
    
    WrenScriptSystem::GetInstance().Shutdown();
}
```

## Performance Considerations

### Memory Usage

- Wren VM: ~1-2 MB base
- Per-script: ~10-50 KB typical
- Foreign objects: Minimal (just pointers)

### Execution Speed

- Script initialization: < 10 ms typical
- Frame update: < 1 ms per script (for simple logic)
- Physics queries: Depends on scene complexity

### Optimization Tips

1. **Cache Component References**
   ```wren
   _rb = _gameObject.getComponent("RigidBody")
   ```

2. **Avoid Allocations in Hot Paths**
   ```wren
   var pos = _transform.position
   _transform.setPosition(pos.x + 1, pos.y, pos.z)
   ```

3. **Use Pooling for Frequent Objects**
   ```wren
   var projectile = _projectilePool.get()
   ```

4. **Profile with Logging**
   ```wren
   var start = Time.time
   expensiveOperation()
   Debug.log("Op took: %(Time.time - start) seconds")
   ```

## Troubleshooting

### Scripts Not Loading

**Check:**
- File path is correct
- File exists and is readable
- Syntax is valid Wren

**Debug:**
```cpp
bool result = WrenScriptSystem::GetInstance().RunScript(filepath);
if (!result) {
    // Check console for error details
}
```

### Bindings Not Working

**Verify:**
- Binding exists in system initialization
- Function signature is correct
- Parameters match expected types

### Performance Issues

**Profile with:**
```wren
var start = Time.time
criticalSection()
var elapsed = Time.time - start
Debug.log("Section: %(elapsed) ms")
```

### Memory Leaks

**Check:**
- Foreign objects properly managed
- No circular references
- Wren garbage collection running

## Migration from Other Languages

### From Lua
```lua
-- Lua
function update(dt)
    x = x + 1
end

-- Wren
update(dt) {
    x = x + 1
}
```

### From Python
```python
# Python
def update(dt):
    x += 1

# Wren
update(dt) {
    x = x + 1
}
```

### From C#
```csharp
// C#
public void Update(float dt) {
    x += 1;
}

// Wren
update(dt) {
    x = x + 1
}
```

## Next Steps

1. **Read**: [WREN_SCRIPTING_GUIDE.md](./WREN_SCRIPTING_GUIDE.md) for comprehensive documentation
2. **Reference**: [WREN_QUICK_REFERENCE.md](./WREN_QUICK_REFERENCE.md) for syntax and bindings
3. **Examples**: Review example scripts in `assets/scripts/`
4. **Experiment**: Create simple test scripts
5. **Build**: Integrate into game systems

## Support Resources

- **Wren Language**: https://wren.io
- **Engine Code**: `include/WrenScriptSystem.h`, `src/WrenScriptSystem.cpp`
- **Examples**: `assets/scripts/` directory
- **Tests**: `tests/test_wren_scripting.cpp`

## Summary

You now have:
- ✅ Wren VM integrated into game engine
- ✅ Script components for GameObjects
- ✅ Built-in bindings for game engine types
- ✅ Example scripts demonstrating gameplay logic
- ✅ Hot-reload support for rapid iteration
- ✅ Documentation and quick reference

Start scripting gameplay logic with Wren today!
