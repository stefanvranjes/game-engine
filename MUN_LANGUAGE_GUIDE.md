# Mun Language Support - Implementation Guide

## Overview

Mun is a statically-typed, compiled scripting language specifically designed for game development with **built-in hot-reload capabilities**. Unlike interpreted languages, Mun scripts are compiled to native code while maintaining the iteration speed of interpreted languages through its unique hot-reload system.

### Why Mun?

| Feature | Advantage |
|---------|-----------|
| **Compiled** | Near C++ performance, no GC pauses |
| **Hot-Reload** | Recompile and reload scripts without engine restart |
| **Static Types** | Catch errors at compile-time, not runtime |
| **Safety** | Ownership system prevents common memory bugs |
| **First-Class Structs** | Perfect for game data and components |
| **Low Friction** | Minimal C++ interop boilerplate |

### Mun vs Other Languages

```
┌─────────────┬──────────┬──────────┬────────────┬─────────────┐
│ Language    │ Compiled │ Hot-Reload │ Performance│ Ease-of-Use │
├─────────────┼──────────┼──────────┼────────────┼─────────────┤
│ Mun         │ ✓        │ ✓ Native │ Excellent  │ Very Good   │
│ Lua         │ ✗        │ ✓ Reload │ Good       │ Easy        │
│ Python      │ ✗        │ ✓ Reload │ Fair       │ Easy        │
│ Rust (DLL)  │ ✓        │ ✗ Manual │ Excellent  │ Complex     │
│ C#/.NET     │ ✗        │ ✓ Reload │ Good       │ Good        │
│ Go          │ ✓        │ ✗        │ Good       │ Good        │
└─────────────┴──────────┴──────────┴────────────┴─────────────┘
```

## Installation

### Prerequisites

- **Mun Compiler**: Download from [https://mun-lang.org/](https://mun-lang.org/)
- **Platform Support**: Windows, macOS, Linux

### Install Mun

#### Windows
```powershell
# Install via Chocolatey (recommended)
choco install mun

# Or download from: https://github.com/mun-lang/mun/releases
```

#### macOS
```bash
# Install via Homebrew
brew install mun-lang/mun/mun

# Or download from releases
```

#### Linux
```bash
# Download from releases and add to PATH
wget https://github.com/mun-lang/mun/releases/download/v0.4/mun-linux-x64.zip
unzip mun-linux-x64.zip
sudo mv mun /usr/local/bin/
```

### Verify Installation
```bash
mun --version
```

## Integration Steps

### Step 1: Add to CMakeLists.txt

The Mun system is automatically compiled with the engine. Ensure your build includes:

```cmake
# In CMakeLists.txt
add_executable(GameEngine
    # ... existing sources
    src/MunScriptSystem.cpp
)

target_include_directories(GameEngine PRIVATE
    include/
)
```

### Step 2: Initialize in Application

```cpp
// In Application.cpp or your main game class
#include "MunScriptSystem.h"

class Application {
private:
    std::unique_ptr<MunScriptSystem> m_munSystem;

public:
    void Init() {
        m_munSystem = std::make_unique<MunScriptSystem>();
        m_munSystem->Init();  // Checks for Mun compiler
    }

    void Update(float deltaTime) {
        m_munSystem->Update(deltaTime);  // Checks for file changes
    }

    void Shutdown() {
        m_munSystem->Shutdown();
    }
};
```

## Usage

### Basic Script Loading

**C++ Code:**
```cpp
#include "MunScriptSystem.h"

auto& munSys = MunScriptSystem::GetInstance();
munSys.Init();

// Load and compile a Mun script
munSys.LoadScript("scripts/gameplay.mun");

// Script automatically reloads on file changes
munSys.Update(deltaTime);  // Call each frame

munSys.Shutdown();
```

### Writing Mun Scripts

**scripts/gameplay.mun:**
```mun
pub fn update_player(health: f32, damage: f32) -> f32 {
    let new_health = health - damage;
    if new_health < 0.0 {
        0.0
    } else {
        new_health
    }
}

pub struct Player {
    name: String,
    health: f32,
    speed: f32,
    is_alive: bool,
}

impl Player {
    pub fn new(name: String) -> Player {
        Player {
            name,
            health: 100.0,
            speed: 5.0,
            is_alive: true,
        }
    }

    pub fn take_damage(self: &mut Self, damage: f32) {
        self.health -= damage;
        if self.health <= 0.0 {
            self.health = 0.0;
            self.is_alive = false;
        }
    }

    pub fn heal(self: &mut Self, amount: f32) {
        self.health = (self.health + amount).min(100.0);
        self.is_alive = true;
    }

    pub fn is_healthy(self: &Self) -> bool {
        self.health > 50.0
    }
}

pub fn calculate_damage(base: f32, multiplier: f32, is_critical: bool) -> f32 {
    let mut final_damage = base * multiplier;
    if is_critical {
        final_damage *= 1.5;
    }
    final_damage
}
```

### Accessing Functions from C++

The compiled Mun library exposes functions that can be accessed via dynamic linking:

```cpp
// Advanced: Access compiled functions directly
#include "MunScriptSystem.h"

typedef float (*UpdatePlayerFunc)(float health, float damage);

void GameLogic::OnPlayerDamaged() {
    auto libPath = MunScriptSystem::GetInstance()
        .GetCompiledLibraryPath("gameplay");

    if (libPath.empty()) {
        std::cerr << "Script not loaded" << std::endl;
        return;
    }

    // The compiled Mun library can be accessed via dlopen/LoadLibraryA
    // and function pointers retrieved via dlsym/GetProcAddress
}
```

## Advanced Features

### Compilation Options

```cpp
MunScriptSystem::CompilationOptions opts;
opts.optimize = true;           // Enable release optimizations
opts.targetDir = "mun-target";  // Where to put compiled .dll/.so
opts.verbose = true;            // Show compiler output
opts.emitMetadata = true;       // Generate type metadata

auto& munSys = MunScriptSystem::GetInstance();
munSys.SetCompilationOptions(opts);
munSys.LoadScript("scripts/gameplay.mun");
```

### Hot-Reload Callbacks

```cpp
MunScriptSystem& munSys = MunScriptSystem::GetInstance();

// Set callback when script reloads
munSys.SetOnScriptReloaded([](const std::string& scriptName) {
    std::cout << "Script reloaded: " << scriptName << std::endl;
    // Re-bind functions, reset state, etc.
});

// Disable automatic hot-reload if needed
munSys.SetAutoHotReload(false);
```

### Directory Watching

```cpp
// Watch entire scripts directory for .mun files
MunScriptSystem::GetInstance()
    .WatchScriptDirectory("scripts/");

// Manually watch specific file
MunScriptSystem::GetInstance()
    .WatchScriptFile("scripts/gameplay.mun");

// Stop watching
MunScriptSystem::GetInstance()
    .UnwatchScriptFile("scripts/gameplay.mun");
```

### Statistics and Debugging

```cpp
auto& munSys = MunScriptSystem::GetInstance();

// Get compilation statistics
const auto& stats = munSys.GetCompilationStats();
std::cout << "Total compiles: " << stats.totalCompiles << std::endl;
std::cout << "Successful: " << stats.successfulCompiles << std::endl;
std::cout << "Failed: " << stats.failedCompiles << std::endl;
std::cout << "Hot-reloads: " << stats.totalReloads << std::endl;
std::cout << "Total compile time: " << stats.totalCompileTime << "s" << std::endl;
std::cout << "Last compile time: " << stats.lastCompileTime << "s" << std::endl;

// Reset statistics
munSys.ResetStats();

// Check for errors
if (munSys.HasErrors()) {
    std::cerr << munSys.GetLastError() << std::endl;
}

// Get list of loaded scripts
for (const auto& script : munSys.GetLoadedScripts()) {
    std::cout << "Loaded: " << script << std::endl;
}
```

## Mun Language Features

### Type System

```mun
// Primitives
pub fn primitives() {
    let i: i32 = 42;
    let f: f32 = 3.14;
    let b: bool = true;
    let s: String = "Hello";
}

// Structs
pub struct Position {
    x: f32,
    y: f32,
    z: f32,
}

// Enums
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

// Functions
pub fn distance(a: Position, b: Position) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}
```

### Methods and Impl Blocks

```mun
pub struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }

    pub fn magnitude(self: &Self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(self: &mut Self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            self.x /= mag;
            self.y /= mag;
            self.z /= mag;
        }
    }

    pub fn dot(self: &Self, other: &Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}
```

### Pattern Matching

```mun
pub enum Damage {
    Physical(f32),
    Fire(f32),
    Frost(f32),
    Holy(f32),
}

pub fn apply_damage(damage: Damage, armor: f32) -> f32 {
    match damage {
        Damage::Physical(amount) => (amount - armor).max(0.0),
        Damage::Fire(amount) => amount * 1.5,  // Armor doesn't protect
        Damage::Frost(amount) => amount * 0.8, // Slightly reduced
        Damage::Holy(amount) => amount,        // Full damage
    }
}
```

### Ownership and Borrowing

```mun
pub struct GameObject {
    name: String,
    components: Vec<(String, i32)>,
}

pub fn take_ownership(obj: GameObject) {
    // obj is now owned by this function
    // Original is moved, not accessible outside
}

pub fn borrow(obj: &GameObject) {
    // Read-only access, original still owned
}

pub fn borrow_mut(obj: &mut GameObject) {
    // Mutable access, original still owned
}
```

## Performance Considerations

### Compilation Time
- **First Load**: ~500ms - 5s depending on script size
- **Incremental Reload**: ~200ms - 1s with optimizations disabled
- **Release Build**: 1-3s with full optimizations

### Runtime Performance
- **Function Calls**: ~0% overhead (native code)
- **Memory**: Minimal overhead, no GC
- **Hot-Reload**: No frame time impact (compile on separate thread recommended)

### Optimization Tips

1. **Disable Optimizations During Development**
   ```cpp
   opts.optimize = false;  // Faster compilation, slower execution
   ```

2. **Use Release Mode for Shipping**
   ```cpp
   opts.optimize = true;  // Slower compilation, best performance
   ```

3. **Separate Compilation Thread**
   ```cpp
   // For production, compile on background thread
   // to avoid frame rate impact
   std::thread compilationThread([script] {
       MunScriptSystem::GetInstance().CompileScript(script);
   });
   ```

## File Structure

```
project-root/
├── include/
│   └── MunScriptSystem.h          # Header file
├── src/
│   └── MunScriptSystem.cpp        # Implementation
├── scripts/
│   ├── gameplay.mun               # Mun source files
│   ├── ai.mun
│   └── physics.mun
└── mun-target/                    # Compiled output (auto-created)
    ├── gameplay.dll/so            # Compiled binaries
    ├── ai.dll/so
    └── physics.dll/so
```

## Troubleshooting

### "Mun compiler not found"
- **Cause**: Mun not installed or not in system PATH
- **Solution**: Install Mun from [https://mun-lang.org/](https://mun-lang.org/)
- **Check**: Run `mun --version` in terminal

### Compilation Fails
- **Check Syntax**: Use `mun check scripts/file.mun`
- **View Errors**: Enable verbose mode: `opts.verbose = true`
- **Inspect Compiler Output**: Check console for detailed error messages

### Hot-Reload Not Working
- **Enable Watching**: Ensure file is watched via `WatchScriptFile()`
- **Check Updates**: Call `Update()` each frame in main loop
- **Verify Changes**: Ensure file is actually saved (IDE issue?)

### Function Not Found in Library
- **Mark Public**: Functions must be `pub fn` to be exported
- **Check Name Mangling**: Mun uses stable naming, verify in output

## Best Practices

### 1. Organize Scripts by System
```
scripts/
├── gameplay/
│   ├── player.mun
│   ├── enemy.mun
│   └── quest.mun
├── ui/
│   ├── menu.mun
│   └── hud.mun
└── physics/
    └── gravity.mun
```

### 2. Version Control
```bash
# Track source, ignore compiled output
scripts/**.mun  # Commit these
mun-target/     # Add to .gitignore
```

### 3. Error Handling
```cpp
MunScriptSystem& munSys = MunScriptSystem::GetInstance();
if (!munSys.LoadScript("scripts/gameplay.mun")) {
    std::cerr << "Load failed: " << munSys.GetLastError() << std::endl;
    // Fall back to default behavior
}
```

### 4. Development Workflow
```
Edit .mun → Save → Engine auto-detects → Recompiles → Hot-reloads
                      (via file watcher)
```

## Integration with Game Systems

### Behavior Trees with Mun

```mun
pub enum BehaviorResult {
    Success,
    Failure,
    Running,
}

pub struct BehaviorTree {
    name: String,
}

pub fn evaluate_behavior(entity_id: i32) -> BehaviorResult {
    if can_see_player(entity_id) {
        BehaviorResult::Running
    } else {
        BehaviorResult::Success
    }
}
```

### Component-Based Architecture

```mun
pub struct Position {
    x: f32,
    y: f32,
}

pub struct Velocity {
    vx: f32,
    vy: f32,
}

pub fn system_movement(
    positions: &Vec<Position>,
    velocities: &Vec<Velocity>,
    dt: f32,
) {
    // Update positions
}
```

## Migration from Other Languages

### From Lua
```lua
-- Lua
function update_player(health, damage)
    return health - damage
end
```

```mun
// Mun
pub fn update_player(health: f32, damage: f32) -> f32 {
    health - damage
}
```

### From Rust
```rust
// Rust (same syntax!)
pub fn update_player(health: f32, damage: f32) -> f32 {
    health - damage
}
```

## Additional Resources

- **Official Mun Book**: [https://docs.mun-lang.org/](https://docs.mun-lang.org/)
- **GitHub Repository**: [https://github.com/mun-lang/mun](https://github.com/mun-lang/mun)
- **Discord Community**: [https://discord.gg/mun-lang](https://discord.gg/mun-lang)
- **Playground**: [https://play.mun-lang.org/](https://play.mun-lang.org/)

## Summary

The Mun integration provides your game engine with:
- ✅ Compiled performance for critical gameplay scripts
- ✅ Automatic hot-reload for rapid iteration
- ✅ Static typing to catch errors early
- ✅ Memory safety without garbage collection
- ✅ Seamless C++ interoperability

This makes Mun ideal for performance-critical gameplay logic that needs fast development iteration.
