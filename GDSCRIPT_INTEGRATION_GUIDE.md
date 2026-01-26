# GDScript Integration Guide

## Overview

This game engine now supports **GDScript**, the native scripting language from the Godot engine. GDScript brings powerful game development capabilities with a syntax designed specifically for game programming, featuring static typing with type inference, object-oriented programming, and fast interpretation.

## Why GDScript?

### Advantages
- **Godot Familiar**: If your team uses Godot, GDScript development transfers directly
- **Game-Centric Design**: Syntax optimized for game development (signals, nodes, physics)
- **Performance**: Comparable to Lua with optional JIT compilation support
- **Type Safety**: Static typing with type inference prevents many runtime errors
- **Visual Integration**: Works seamlessly with Godot editor workflows
- **Rich Standard Library**: Built-in functions for math, strings, and collections

### Comparison with Other Languages

| Feature | GDScript | Lua | Python | Wren |
|---------|----------|-----|--------|------|
| Type System | Static with inference | Dynamic | Dynamic | Static |
| Performance | ~3-5ms | ~5-10ms | ~20-50ms | ~1-3ms |
| Learning Curve | Moderate | Easy | Easy | Easy |
| Game Features | Excellent | Good | Good | Good |
| Godot Integration | Native | Via binding | Via binding | Via binding |
| Hot-Reload | Yes | Yes | Yes | No |

## Quick Start

### 1. Basic Setup

```cpp
#include "ScriptLanguageRegistry.h"
#include "GDScriptSystem.h"

// Initialize
auto& registry = ScriptLanguageRegistry::GetInstance();
registry.Init();

// GDScript is automatically registered and ready to use
```

### 2. Running a GDScript File

```cpp
// Load and execute a GDScript file
registry.ExecuteScript("scripts/player.gd", ScriptLanguage::GDScript);

// Or use file extension detection
registry.ExecuteScript("scripts/player.gd");  // Automatically detects .gd -> GDScript
```

### 3. Accessing the GDScript System

```cpp
#include "GDScriptSystem.h"

auto& gdscript = GDScriptSystem::GetInstance();

// Get execution time for profiling
double lastTime = gdscript.GetLastExecutionTime();

// Check for errors
if (gdscript.HasErrors()) {
    std::cout << "Error: " << gdscript.GetLastError() << std::endl;
}

// Get memory usage
uint64_t memory = gdscript.GetMemoryUsage();
```

## GDScript Scripting

### File Naming Convention

GDScript files use the `.gd` extension:
```
scripts/
├── player.gd
├── enemy.gd
├── game_manager.gd
└── utils/
    └── math_helper.gd
```

### Basic GDScript Syntax

```gdscript
# player.gd

extends Node

# Member variables
var health: float = 100.0
var speed: float = 5.0
var is_alive: bool = true

# Called when script is loaded
func _ready():
    print("Player initialized with health: ", health)

# Called every frame
func _process(delta: float):
    print("Delta time: ", delta)
    
    # Update player movement
    handle_movement()

# Custom function
func handle_movement():
    # Movement logic here
    pass

# Signal definition (event system)
signal health_changed(new_health: float)
signal died()

func take_damage(amount: float):
    health -= amount
    emit_signal("health_changed", health)
    
    if health <= 0:
        is_alive = false
        emit_signal("died")
```

### Type System

GDScript provides strong typing:

```gdscript
# Explicit typing
var position: Vector3 = Vector3(1, 2, 3)
var name: String = "Player"
var count: int = 42
var value: float = 3.14

# Type inference
var auto_position = Vector3(1, 2, 3)  # Inferred as Vector3

# Function signatures
func move(direction: Vector3, distance: float) -> bool:
    # Do movement
    return true

# Optional typing - dynamic if not specified
var value = get_some_value()  # Dynamic type
```

### Classes and Inheritance

```gdscript
# base_entity.gd
class_name BaseEntity
extends Node

var health: float = 100.0

func take_damage(amount: float):
    health -= amount

# enemy.gd
extends BaseEntity

var damage: float = 10.0

func _ready():
    health = 50.0  # Override base health

func attack(target: Node):
    target.take_damage(damage)
```

## C++ Integration

### Calling GDScript Functions from C++

```cpp
auto& gdscript = GDScriptSystem::GetInstance();

// Call a parameterless function
gdscript.CallFunction("initialize", {});

// Call with arguments
std::vector<std::any> args = {10.5f, 20.3f, 5};
std::any result = gdscript.CallFunction("calculate", args);

// Extract return value
if (result.type() == typeid(float)) {
    float value = std::any_cast<float>(result);
}
```

### Binding C++ Functions to GDScript

```cpp
auto& gdscript = GDScriptSystem::GetInstance();

// Bind a C++ function for use in GDScript
gdscript.BindFunction("get_delta_time", [](const std::vector<std::any>& args) -> std::any {
    return 0.016f;  // Return delta time
});

// Now accessible in GDScript as:
// var dt = get_delta_time()
```

### Registering C++ Classes

```cpp
auto& gdscript = GDScriptSystem::GetInstance();

// Register a custom C++ class for use in GDScript
gdscript.RegisterClass("Player", [](GDScriptObject obj) {
    // Initialize Player class with methods and properties
});

// Now usable in GDScript as:
// var player = Player.new()
```

### Signal System (Callback Connections)

```cpp
auto& gdscript = GDScriptSystem::GetInstance();

// Connect to a GDScript signal
gdscript.ConnectSignal("player", "health_changed", 
    [](const std::vector<std::any>& args) {
        if (args.size() > 0 && args[0].type() == typeid(float)) {
            float new_health = std::any_cast<float>(args[0]);
            std::cout << "Player health changed to: " << new_health << std::endl;
        }
    });

// Emit a signal from C++ to trigger GDScript callbacks
gdscript.EmitSignal("game", "on_level_start", {});
```

## Advanced Features

### Hot-Reload Support

Enable hot-reload for rapid development iteration:

```cpp
auto& gdscript = GDScriptSystem::GetInstance();

// Enable hot-reload
gdscript.SetHotReloadEnabled(true);

// Reload a script after editing
gdscript.ReloadScript("scripts/player.gd");
```

### Memory Management

Monitor GDScript system memory usage:

```cpp
auto& gdscript = GDScriptSystem::GetInstance();

uint64_t memory_used = gdscript.GetMemoryUsage();
std::cout << "GDScript memory: " << memory_used / 1024 << " KB" << std::endl;

// Get total across all scripting systems
auto& registry = ScriptLanguageRegistry::GetInstance();
uint64_t total_memory = registry.GetTotalMemoryUsage();
```

### Error Handling

```cpp
auto& gdscript = GDScriptSystem::GetInstance();

// Execute script and check for errors
bool success = gdscript.RunScript("scripts/game_logic.gd");

if (!success && gdscript.HasErrors()) {
    std::cerr << "Script error: " << gdscript.GetLastError() << std::endl;
}

// Check at registry level
auto& registry = ScriptLanguageRegistry::GetInstance();
auto errors = registry.GetAllErrors();
for (const auto& [lang, msg] : errors) {
    std::cerr << lang << ": " << msg << std::endl;
}
```

## Engine Type Bindings

The GDScript system automatically registers common engine types:

### Vec3
```gdscript
var position: Vector3 = Vector3(1, 2, 3)
var distance: float = position.length()
var normalized: Vector3 = position.normalized()
```

### Transform
```gdscript
var transform = Transform.new()
transform.position = Vector3(0, 0, 0)
transform.rotation = Quaternion()
```

### GameObject
```gdscript
var game_object = GameObject.new()
game_object.set_position(Vector3(0, 0, 0))
game_object.set_active(true)
```

## Example Game Scripts

### Player Controller

```gdscript
# scripts/player_controller.gd
extends Node

class_name PlayerController

var velocity: Vector3 = Vector3.ZERO
var move_speed: float = 5.0
var rotation_speed: float = 2.0

signal on_jump()
signal on_damage(amount: float)

func _ready():
    print("Player controller ready")

func _process(delta: float):
    handle_input(delta)
    update_position(delta)

func handle_input(delta: float):
    # Example: Handle movement (would use actual input system)
    var direction = Vector3.ZERO
    
    # Simulate input
    if true:  # Should be actual input check
        direction.z -= 1
    
    if direction.length() > 0:
        velocity = direction.normalized() * move_speed

func update_position(delta: float):
    # Update position based on velocity
    # position += velocity * delta
    pass

func take_damage(amount: float):
    emit_signal("on_damage", amount)
```

### Game Manager

```gdscript
# scripts/game_manager.gd
extends Node

class_name GameManager

var is_paused: bool = false
var current_level: int = 1
var player: Node

signal level_started(level: int)
signal level_finished(level: int)
signal game_paused()
signal game_resumed()

func _ready():
    emit_signal("level_started", current_level)

func pause_game():
    is_paused = true
    emit_signal("game_paused")

func resume_game():
    is_paused = false
    emit_signal("game_resumed")

func next_level():
    current_level += 1
    emit_signal("level_finished", current_level - 1)
    emit_signal("level_started", current_level)

func set_player(new_player: Node):
    player = new_player
```

### Enemy AI

```gdscript
# scripts/enemy_ai.gd
extends Node

class_name EnemyAI

var target: Node
var detection_range: float = 10.0
var patrol_speed: float = 2.0
var chase_speed: float = 5.0
var state: String = "idle"  # idle, patrol, chase, attack

func _ready():
    print("Enemy AI initialized")

func _process(delta: float):
    update_state(delta)
    match state:
        "idle":
            patrol(delta)
        "chase":
            chase_target(delta)
        "attack":
            attack_target(delta)

func update_state(delta: float):
    if target == null:
        state = "idle"
        return
    
    var distance = (target.position - position).length()
    
    if distance < detection_range:
        state = "chase"
    else:
        state = "idle"

func patrol(delta: float):
    # Simple patrol logic
    pass

func chase_target(delta: float):
    if target == null:
        return
    
    var direction = (target.position - position).normalized()
    # Move towards target with chase_speed
    pass

func attack_target(delta: float):
    if target == null:
        return
    
    # Attack logic
    pass

func take_damage(amount: float):
    print("Enemy took ", amount, " damage")
```

## Best Practices

### 1. Script Organization

```
scripts/
├── core/
│   ├── game_manager.gd
│   └── input_manager.gd
├── entities/
│   ├── player.gd
│   ├── enemy.gd
│   └── npc.gd
├── ui/
│   ├── hud.gd
│   └── menu.gd
└── utils/
    ├── math.gd
    └── helpers.gd
```

### 2. Use Type Hints

```gdscript
# Good - Clear types
func calculate_damage(attacker: Node, target: Node) -> float:
    pass

# Avoid - Unclear types
func calc_dmg(a, t):
    pass
```

### 3. Signal Usage for Loose Coupling

```gdscript
# Good - Signals decouple game systems
signal enemy_died(position: Vector3)

func die():
    emit_signal("enemy_died", global_position)

# Avoid - Direct function calls create tight coupling
game_manager.on_enemy_death(self)
```

### 4. Performance Considerations

```gdscript
# Cache expensive operations
var cached_transform: Transform

func _ready():
    cached_transform = global_transform

# Avoid expensive operations in _process
func _process(delta: float):
    # Good - O(1) lookup
    var pos = cached_transform.position
    
    # Avoid - Expensive computation every frame
    # var pos = some_complex_calculation()
```

## Integration with Game Loop

```cpp
// In your Application.cpp Update() loop
auto& registry = ScriptLanguageRegistry::GetInstance();

// Call scripted update functions
registry.CallFunction("_process", {delta_time});

// Or call GDScript system directly
auto& gdscript = GDScriptSystem::GetInstance();
gdscript.CallFunction("game_update", {delta_time});
```

## Performance Metrics

### Typical Execution Times
- Script load: 2-5ms per script
- Function call overhead: < 0.1ms
- Memory per instance: ~100-500 bytes
- Total system memory: ~500KB-2MB typical

### Optimization Tips
1. Use type hints for better performance
2. Cache transform and frequently accessed properties
3. Avoid expensive operations in _process()
4. Use signals instead of frequent function calls
5. Profile with execution time tracking

## Troubleshooting

### Script Not Loading
```cpp
if (!gdscript.RunScript("scripts/player.gd")) {
    std::cerr << "Error: " << gdscript.GetLastError() << std::endl;
}
```

### Function Not Found
```cpp
if (gdscript.HasFunction("my_function")) {
    gdscript.CallFunction("my_function", {});
} else {
    std::cout << "Function not found" << std::endl;
}
```

### Memory Leaks
Monitor memory usage regularly:
```cpp
uint64_t before = gdscript.GetMemoryUsage();
gdscript.RunScript("script.gd");
uint64_t after = gdscript.GetMemoryUsage();
std::cout << "Memory delta: " << (after - before) << " bytes" << std::endl;
```

## References

- [Official GDScript Documentation](https://docs.godotengine.org/en/stable/getting_started/scripting/gdscript/index.html)
- [Godot Engine Documentation](https://docs.godotengine.org/)
- [Game Engine Architecture](../ARCHITECTURE_DIAGRAM.md)
- [Multi-Language Scripting Guide](../MULTI_LANGUAGE_SCRIPTING_GUIDE.md)

## Files Reference

| File | Purpose |
|------|---------|
| [include/GDScriptSystem.h](../include/GDScriptSystem.h) | GDScript system class definition |
| [src/GDScriptSystem.cpp](../src/GDScriptSystem.cpp) | Implementation of GDScript system |
| [include/IScriptSystem.h](../include/IScriptSystem.h) | Base scripting interface (defines ScriptLanguage enum) |
| [src/ScriptLanguageRegistry.cpp](../src/ScriptLanguageRegistry.cpp) | Script language registration |

## Next Steps

1. Create your first GDScript file in `scripts/` directory
2. Load it using `registry.ExecuteScript("scripts/your_script.gd")`
3. Set up signal connections between C++ and GDScript
4. Implement game logic in GDScript
5. Use C++ for performance-critical systems (rendering, physics)
