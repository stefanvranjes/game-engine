# Mun Language Quick Reference

## Installation

```bash
# Windows (Chocolatey)
choco install mun

# macOS (Homebrew)
brew install mun-lang/mun/mun

# Verify
mun --version
```

## Integration Summary

| Component | File | Purpose |
|-----------|------|---------|
| Header | [include/MunScriptSystem.h](include/MunScriptSystem.h) | System API |
| Implementation | [src/MunScriptSystem.cpp](src/MunScriptSystem.cpp) | Core logic |
| Guide | [MUN_LANGUAGE_GUIDE.md](MUN_LANGUAGE_GUIDE.md) | Full documentation |
| Example Script | [scripts/gameplay.mun](scripts/gameplay.mun) | Mun code examples |

## Quick Start

### C++ Initialization
```cpp
#include "MunScriptSystem.h"

MunScriptSystem& mun = MunScriptSystem::GetInstance();
mun.Init();                              // Check for compiler
mun.LoadScript("scripts/gameplay.mun");  // Compile & load
mun.Update(deltaTime);                   // Update each frame
mun.Shutdown();                          // Cleanup
```

### Mun Syntax
```mun
// Functions
pub fn greet(name: String) -> String {
    "Hello, " + name
}

// Structs
pub struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

// Methods
impl Vector3 {
    pub fn magnitude(self: &Self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

// Enums
pub enum Color {
    Red,
    Green,
    Blue,
}

// Pattern matching
match color {
    Color::Red => { /* code */ },
    Color::Green => { /* code */ },
    Color::Blue => { /* code */ },
}
```

## Hot-Reload Workflow

```
1. Edit scripts/gameplay.mun
2. Save file
3. Engine detects change (FileWatcher)
4. Auto-recompiles to mun-target/
5. Auto-reloads library
6. OnScriptReloaded callback triggered
```

## Core API

### Loading Scripts
```cpp
mun.LoadScript("scripts/gameplay.mun");
mun.CompileScript("scripts/gameplay.mun");
```

### Hot-Reload Control
```cpp
mun.SetAutoHotReload(true);           // Enable auto reload
mun.RecompileAndReload("gameplay");    // Manual reload
mun.SetOnScriptReloaded([](auto s) {   // Reload callback
    std::cout << "Reloaded: " << s << std::endl;
});
```

### File Watching
```cpp
mun.WatchScriptFile("scripts/gameplay.mun");
mun.WatchScriptDirectory("scripts/");
mun.UnwatchScriptFile("scripts/gameplay.mun");
```

### Statistics
```cpp
const auto& stats = mun.GetCompilationStats();
stats.totalCompiles;       // Total compilation count
stats.successfulCompiles;  // Successful count
stats.failedCompiles;      // Failed count
stats.totalReloads;        // Hot-reload count
stats.totalCompileTime;    // Total seconds spent compiling
stats.lastCompileTime;     // Last compile duration

mun.ResetStats();
```

### Debugging
```cpp
if (mun.HasErrors()) {
    std::cerr << mun.GetLastError() << std::endl;
}

for (const auto& script : mun.GetLoadedScripts()) {
    std::cout << "Loaded: " << script << std::endl;
}

for (const auto& file : mun.GetWatchedFiles()) {
    std::cout << "Watching: " << file << std::endl;
}
```

### Compilation Options
```cpp
MunScriptSystem::CompilationOptions opts;
opts.optimize = true;        // Release optimizations
opts.targetDir = "mun-target"; // Output directory
opts.verbose = true;         // Show compiler output
opts.emitMetadata = true;    // Type metadata

mun.SetCompilationOptions(opts);
```

## Mun Language Features

### Types
```mun
// Primitives
i32, i64           // Signed integers
u32, u64           // Unsigned integers
f32, f64           // Floating point
bool               // Boolean
String             // Dynamic string

// Collections
Vec<T>             // Vector/Dynamic array
(T1, T2, ...)      // Tuples

// Ownership
&T                 // Immutable reference
&mut T             // Mutable reference
```

### Structs and Methods
```mun
pub struct Player {
    name: String,
    health: f32,
}

impl Player {
    pub fn new(name: String) -> Player {
        Player { name, health: 100.0 }
    }

    pub fn take_damage(self: &mut Self, damage: f32) {
        self.health -= damage;
    }
}
```

### Enums and Pattern Matching
```mun
pub enum Weapon {
    Sword(f32),      // f32 = damage
    Bow(f32),
    Staff(f32),
}

pub fn get_damage(weapon: Weapon) -> f32 {
    match weapon {
        Weapon::Sword(damage) => damage,
        Weapon::Bow(damage) => damage * 0.8,
        Weapon::Staff(damage) => damage * 1.2,
    }
}
```

### Ownership Rules
```mun
// Owned: Only one owner
pub fn take_ownership(obj: MyStruct) {
    // obj is moved here, original no longer accessible
}

// Borrow: Read-only access
pub fn borrow_immut(obj: &MyStruct) {
    // Read obj, but can't modify
}

// Mutable borrow: Can modify
pub fn borrow_mut(obj: &mut MyStruct) {
    // Can read and modify obj
}
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| First Compile | 500ms - 5s | Depends on script size |
| Incremental Reload | 200ms - 1s | Unoptimized |
| Release Optimization | 1s - 3s | Full optimization pass |
| Function Call | ~0ns | Native code execution |
| Hot-Reload Overhead | ~0ms | No frame impact |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Mun compiler not found" | Install Mun, verify PATH |
| Compilation fails | Run `mun check script.mun` for details |
| Hot-reload not working | Call `mun.Update()` each frame |
| Function not exported | Mark with `pub fn` |
| Type mismatch errors | Check argument types, use type annotations |

## Directory Structure

```
project/
├── include/
│   └── MunScriptSystem.h
├── src/
│   └── MunScriptSystem.cpp
├── scripts/                   # Mun source files (.mun)
│   ├── gameplay.mun
│   ├── ai.mun
│   └── physics.mun
└── mun-target/               # Compiled output (auto)
    ├── gameplay.dll
    ├── ai.dll
    └── physics.dll
```

## Example: AI Behavior

**scripts/ai.mun:**
```mun
pub enum AIState {
    Idle,
    Patrol,
    Chase,
    Attack,
}

pub struct AIController {
    state: AIState,
    patrol_speed: f32,
    chase_speed: f32,
}

impl AIController {
    pub fn new() -> AIController {
        AIController {
            state: AIState::Idle,
            patrol_speed: 2.0,
            chase_speed: 5.0,
        }
    }

    pub fn update(self: &mut Self, player_in_range: bool) {
        if player_in_range {
            self.state = AIState::Chase;
        } else {
            self.state = AIState::Idle;
        }
    }

    pub fn get_speed(self: &Self) -> f32 {
        match self.state {
            AIState::Idle => 0.0,
            AIState::Patrol => self.patrol_speed,
            AIState::Chase => self.chase_speed,
            AIState::Attack => 0.0,
        }
    }
}
```

## Resources

- **Official Docs**: https://docs.mun-lang.org/
- **GitHub**: https://github.com/mun-lang/mun
- **Playground**: https://play.mun-lang.org/
- **Discord**: https://discord.gg/mun-lang
- **Book**: https://docs.mun-lang.org/book/

## Key Differences from Other Languages

### vs Lua
- ✅ Compiled (faster)
- ✅ Static typing (safer)
- ❌ Not as easy to learn
- ❌ Smaller ecosystem

### vs Rust
- ✅ Faster compilation
- ✅ Hot-reload native
- ✅ Simpler for scripting
- ❌ Less powerful type system
- ❌ Smaller community

### vs Python
- ✅ Much faster
- ✅ Static typing
- ✅ Better for games
- ❌ Less general purpose
- ❌ Smaller community

### vs C#/.NET
- ✅ No GC pauses
- ✅ Designed for games
- ✅ Native compilation
- ❌ Smaller standard library
- ❌ Newer ecosystem

## Summary

Mun provides the perfect balance for game development:
- **Performance**: Compiled to native code
- **Iteration**: Hot-reload without restart
- **Safety**: Static typing prevents errors
- **Ease of Use**: Simple, game-focused syntax

Ideal for: Gameplay logic, AI, physics, balancing parameters
