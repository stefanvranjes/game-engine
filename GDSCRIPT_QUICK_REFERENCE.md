# GDScript Integration Quick Reference

## File Extension
- `.gd` - GDScript files

## Quick Start

### Initialize GDScript System
```cpp
#include "ScriptLanguageRegistry.h"

auto& registry = ScriptLanguageRegistry::GetInstance();
registry.Init();  // GDScript automatically registered
```

### Load and Run a Script
```cpp
registry.ExecuteScript("scripts/player.gd");  // Auto-detect by extension
registry.ExecuteScript("scripts/player.gd", ScriptLanguage::GDScript);  // Explicit
```

## GDScript Basic Syntax

### Variables and Types
```gdscript
var position: Vector3 = Vector3(0, 0, 0)
var health: float = 100.0
var name: String = "Player"
var is_alive: bool = true
var count: int = 0
```

### Functions
```gdscript
func _ready():  # Called when script loads
    print("Script loaded")

func _process(delta: float):  # Called every frame
    print("Frame time: ", delta)

func custom_function(param: int) -> String:  # Custom function with return
    return "Result: " + str(param)
```

### Classes
```gdscript
class_name Player
extends Node

var health: float = 100.0

func take_damage(amount: float):
    health -= amount
```

### Signals (Events)
```gdscript
signal health_changed(new_health: float)
signal died()

func take_damage(amount: float):
    health -= amount
    emit_signal("health_changed", health)
    if health <= 0:
        emit_signal("died")
```

## C++ Integration Examples

### Call GDScript Function from C++
```cpp
auto& gdscript = GDScriptSystem::GetInstance();

// No arguments
gdscript.CallFunction("initialize", {});

// With arguments
std::vector<std::any> args = {10.5f, 20};
std::any result = gdscript.CallFunction("calculate", args);
```

### Bind C++ Function for GDScript
```cpp
gdscript.BindFunction("get_time", [](const std::vector<std::any>& args) -> std::any {
    return 0.016f;  // Return delta time
});

// Use in GDScript:
# var dt = get_time()
```

### Connect to GDScript Signals
```cpp
gdscript.ConnectSignal("player", "health_changed", 
    [](const std::vector<std::any>& args) {
        if (args.size() > 0) {
            float health = std::any_cast<float>(args[0]);
            std::cout << "Health: " << health << std::endl;
        }
    });
```

### Emit Signal from C++
```cpp
gdscript.EmitSignal("game", "on_level_start", {});
```

## Common GDScript Patterns

### Game Loop
```gdscript
extends Node

var game_running: bool = false

func _ready():
    game_running = true

func _process(delta: float):
    if game_running:
        update_game(delta)

func update_game(delta: float):
    # Game logic
    pass
```

### State Machine
```gdscript
var current_state: String = "idle"

func _process(delta: float):
    match current_state:
        "idle":
            idle_state(delta)
        "move":
            move_state(delta)
        "attack":
            attack_state(delta)

func change_state(new_state: String):
    current_state = new_state
```

### Component-Based System
```gdscript
class_name GameObject
extends Node

var components: Array = []

func add_component(component):
    components.append(component)
    add_child(component)

func get_component(component_type):
    for comp in components:
        if comp.get_class() == component_type:
            return comp
    return null
```

## Performance Tips

1. **Use Type Hints** - Better performance and error detection
   ```gdscript
   # Good
   func update(delta: float) -> void:
       pass
   
   # Avoid
   func update(delta):
       pass
   ```

2. **Cache References** - Don't lookup repeatedly
   ```gdscript
   var cached_transform: Transform
   
   func _ready():
       cached_transform = global_transform
   ```

3. **Use Signals** - Loose coupling between systems
   ```gdscript
   # Good - Decoupled
   signal on_death()
   emit_signal("on_death")
   
   # Avoid - Tight coupling
   game_manager.on_enemy_death(self)
   ```

4. **Profile Execution**
   ```cpp
   auto& gdscript = GDScriptSystem::GetInstance();
   double time = gdscript.GetLastExecutionTime();
   std::cout << "Execution time: " << time << "ms" << std::endl;
   ```

## Error Handling

```cpp
auto& gdscript = GDScriptSystem::GetInstance();

bool success = gdscript.RunScript("script.gd");
if (!success) {
    std::cerr << "Error: " << gdscript.GetLastError() << std::endl;
}
```

## File Organization

```
assets/scripts/
├── core/
│   ├── game_manager.gd
│   └── main_game.gd
├── entities/
│   ├── player_controller.gd
│   └── enemy_ai.gd
├── ui/
│   └── ui_manager.gd
└── utils/
    └── helpers.gd
```

## Key Methods

### GDScriptSystem Methods
| Method | Purpose |
|--------|---------|
| `Init()` | Initialize system |
| `Shutdown()` | Clean up |
| `RunScript(path)` | Load and execute script |
| `ExecuteString(source)` | Execute code from string |
| `CallFunction(name, args)` | Call GDScript function |
| `BindFunction(name, callable)` | Bind C++ function |
| `ConnectSignal(obj, signal, callback)` | Connect to signal |
| `EmitSignal(obj, signal, args)` | Emit signal |
| `SetHotReloadEnabled(bool)` | Enable hot-reload |
| `ReloadScript(path)` | Reload script file |
| `GetMemoryUsage()` | Get memory used |
| `GetLastExecutionTime()` | Get execution time |
| `HasErrors()` | Check for errors |
| `GetLastError()` | Get error message |

### Registry Methods
| Method | Purpose |
|--------|---------|
| `ExecuteScript(path)` | Auto-detect and execute |
| `ExecuteScript(path, lang)` | Execute with language |
| `GetScriptSystem(lang)` | Get language system |
| `CallFunction(name, args)` | Call in any system |
| `GetTotalMemoryUsage()` | All systems memory |
| `GetAllErrors()` | All system errors |

## Example: Integration Flow

```cpp
// C++ main
#include "ScriptLanguageRegistry.h"

int main() {
    auto& registry = ScriptLanguageRegistry::GetInstance();
    registry.Init();
    
    // Load main game script
    registry.ExecuteScript("scripts/main_game.gd");
    
    // Get GDScript system for direct access
    auto& gdscript = GDScriptSystem::GetInstance();
    
    // Game loop
    while (running) {
        // Update game
        gdscript.CallFunction("_process", {delta_time});
        
        // Get game state
        auto state = gdscript.CallFunction("get_game_state", {});
        
        // Check errors
        if (gdscript.HasErrors()) {
            std::cout << "Error: " << gdscript.GetLastError() << std::endl;
        }
    }
    
    registry.Shutdown();
    return 0;
}
```

```gdscript
# scripts/main_game.gd
extends Node

func _process(delta: float):
    # Update game logic
    update_game(delta)

func get_game_state() -> Dictionary:
    return {
        "time": get_time(),
        "score": get_score(),
        "health": get_health()
    }

func update_game(delta: float):
    # Game update logic
    pass
```

## Documentation Files

- [GDSCRIPT_INTEGRATION_GUIDE.md](GDSCRIPT_INTEGRATION_GUIDE.md) - Full documentation
- [include/GDScriptSystem.h](include/GDScriptSystem.h) - Header file
- [src/GDScriptSystem.cpp](src/GDScriptSystem.cpp) - Implementation

## See Also

- [IScriptSystem.h](include/IScriptSystem.h) - Script system interface
- [ScriptLanguageRegistry.h](include/ScriptLanguageRegistry.h) - Language registry
- [MULTI_LANGUAGE_SCRIPTING_GUIDE.md](MULTI_LANGUAGE_SCRIPTING_GUIDE.md) - Other languages
