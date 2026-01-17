# Wren Scripting Architecture Diagram

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      Game Engine (C++)                           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Application::Update(deltaTime)                │ │
│  │                                                            │ │
│  │  for each GameObject:                                      │ │
│  │    scriptComponent->Update(deltaTime)                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │          WrenScriptComponent::Update(dt)                   │ │
│  │                                                            │ │
│  │  - Call Wren update(dt) function                          │ │
│  │  - Handle events and callbacks                            │ │
│  │  - Update physics/animation state                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         WrenScriptSystem (Singleton)                       │ │
│  │                                                            │ │
│  │  - Manage Wren VM                                          │ │
│  │  - Execute scripts                                         │ │
│  │  - Register bindings                                       │ │
│  │  - Handle callbacks                                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Wren VM (Native C)                            │ │
│  │                                                            │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Script Execution                                     │ │ │
│  │  │ - Bytecode compilation                              │ │ │
│  │  │ - Stack-based execution                             │ │ │
│  │  │ - Garbage collection                                │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                                                            │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Native Bindings Bridge                               │ │ │
│  │  │ - Function calls to C++                              │ │ │
│  │  │ - Object marshaling                                  │ │ │
│  │  │ - Type conversions                                   │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
   ┌────────────────────────┐      ┌─────────────────────┐
   │  Game Engine Systems   │      │   Wren Scripts      │
   │                        │      │                     │
   │ - Physics              │      │ - player_behavior   │
   │ - Audio                │      │ - enemy_ai          │
   │ - Animation            │      │ - collectible       │
   │ - Particles            │      │ - game_manager      │
   │ - Input                │      │ - utils             │
   │ - Time                 │      │ - Custom scripts    │
   └────────────────────────┘      └─────────────────────┘
```

---

## Script Lifecycle Diagram

```
┌─────────────┐
│   Attach    │
│  Script     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ LoadScript(path)    │
│ - Load file         │
│ - Parse Wren code   │
│ - Create module     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Init()            │
│ - Call init()       │
│ - Setup state       │
│ - Initialize refs   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Update Loop (frame)│
├─────────────────────┤
│ Update(deltaTime)   │
│ - Call update(dt)   │
│ - Process input     │
│ - Update logic      │
│ - Update animation  │
│ - Update physics    │
└──────┬──────────────┘
       │
       └─────────────┐
                     │ (Repeat every frame)
                     └─────────────┐
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │  Destroy()          │
                          │ - Call destroy()    │
                          │ - Cleanup           │
                          │ - Release resources │
                          └─────────────────────┘
```

---

## Binding System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  Wren Script Side                          │
│                                                            │
│  var gameObject = _gameObject                            │
│  var transform = gameObject.transform                    │
│  transform.setPosition(10, 5, 0)                         │
│                                                            │
│  var rb = gameObject.getComponent("RigidBody")           │
│  rb.setVelocity(5, 10, 0)                                │
└────────────────────────────────────────────────────────────┘
                          │
                          │ (Wren Foreign Function Call)
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│           WrenVM Foreign Function Handler                  │
│                                                            │
│  Parse arguments from Wren stack                          │
│  Call appropriate C++ function                            │
│  Marshal result back to Wren                              │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│                C++ Implementation                          │
│                                                            │
│  Transform::SetPosition(10, 5, 0)                         │
│  RigidBody::SetVelocity(5, 10, 0)                         │
│  ...other engine systems...                               │
└────────────────────────────────────────────────────────────┘
```

---

## Class Hierarchy

```
┌────────────────┐
│ IScriptSystem  │  (Abstract interface)
└────────┬───────┘
         │
    ┌────┼────┬──────────────┐
    │    │    │              │
    ▼    ▼    ▼              ▼
 Lua  Python Custom       Wren
              Script      Script
              System      System
                             │
                             ▼
                    ┌──────────────────┐
                    │ WrenScriptSystem │
                    │                  │
                    │ Manages:         │
                    │ - VM lifecycle   │
                    │ - All bindings   │
                    │ - Script loading │
                    └──────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Wren VM       │
                    │   (Native C)     │
                    └──────────────────┘
```

---

## GameObject-Script Integration

```
┌──────────────────────────────────────────────┐
│              GameObject                      │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │         Components                     │ │
│  │                                        │ │
│  │ - Transform                           │ │
│  │ - RigidBody (Physics)                 │ │
│  │ - Animator                            │ │
│  │ - AudioSource                         │ │
│  │ - ParticleSystem                      │ │
│  │ - [WrenScriptComponent] ◄─────────┐  │ │
│  └────────────────────────────────────┼──┘ │
└──────────────────────────────────────────┼──┘
                                           │
                    ┌──────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ WrenScriptComponent         │
        │                             │
        │ - Loaded Script Path        │
        │ - Module Name               │
        │ - Function Handles          │
        │                             │
        │ Methods:                    │
        │ + Init()                    │
        │ + Update(dt)                │
        │ + Destroy()                 │
        │ + LoadScript()              │
        │ + Reload()                  │
        └────────────┬────────────────┘
                     │
                     │ Calls functions in
                     │
                     ▼
        ┌────────────────────────────┐
        │ Wren Script Instance       │
        │ (player_behavior.wren)     │
        │                            │
        │ class Player {             │
        │   construct new() {...}    │
        │   init() {...}             │
        │   update(dt) {...}         │
        │   destroy() {...}          │
        │ }                          │
        │                            │
        │ var player = null          │
        │                            │
        │ construct init() {...}     │
        │ construct update(dt) {...} │
        │ construct destroy() {...}  │
        └────────────────────────────┘
```

---

## Data Flow: Input → Script → Engine

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Event                          │
│              (Keyboard Key Pressed: 'W')                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Input System (Engine)                     │
│         (Updates Input.getKey() state)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         WrenScriptComponent::Update(dt) called             │
│              (Frame update from main loop)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             Wren Script: update(dt) executes               │
│                                                            │
│  if (Input.getKey("W")) {                                 │
│      var vel = _rb.velocity                              │
│      _rb.setVelocity(vel.x + speed*dt, vel.y, vel.z)    │
│  }                                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         RigidBody::SetVelocity() (C++ Engine)              │
│                                                            │
│  Update physics state                                     │
│  Apply velocity next frame                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Physics System (Next Frame)                      │
│                                                            │
│  Update position based on velocity                        │
│  Transform updated                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Memory Layout

```
┌─────────────────────────────────────────┐
│         Wren VM Memory (~1-2 MB)        │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │   Bytecode (Compiled Scripts)     │ │
│  │   - player_behavior.wren          │ │
│  │   - enemy_ai.wren                 │ │
│  │   - collectible.wren              │ │
│  │   ~100-500 KB total               │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │   Heap (Script Objects)           │ │
│  │   - Class instances               │ │
│  │   - Arrays, maps, strings         │ │
│  │   ~100-500 KB (depends on usage)  │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │   Stack (Execution)               │ │
│  │   - Local variables               │ │
│  │   - Call frames                   │ │
│  │   ~10-100 KB typical              │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │   Foreign Data (C++ Objects)      │ │
│  │   - GameObject pointers           │ │
│  │   - Component handles             │ │
│  │   ~8 bytes per foreign object     │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

---

## Script Loading Sequence

```
1. LoadScript("assets/scripts/player.wren")
   │
   ├─ Read file from disk
   ├─ Parse Wren syntax
   ├─ Compile to bytecode
   └─ Load into VM

2. Init()
   │
   ├─ Call construct init() function
   ├─ Create class instance if exists
   ├─ Initialize member variables
   └─ Cache function handles

3. Update(dt) ← Called every frame
   │
   ├─ Push arguments (dt)
   ├─ Call construct update(dt)
   ├─ Execute game logic
   ├─ Call engine bindings
   └─ Return results to engine

4. OnEvent(name, data)
   │
   ├─ Check if handler exists
   ├─ Call onCollision/onTrigger/custom
   └─ Execute event logic

5. Destroy()
   │
   ├─ Call construct destroy()
   ├─ Clean up resources
   ├─ Release handles
   └─ Remove from active scripts
```

---

## Compilation & Execution Flow

```
Wren Script Source
       │
       ▼
   Lexer (Tokenize)
       │
       ▼
   Parser (AST)
       │
       ▼
   Compiler (Bytecode)
       │
       ▼
   Wren VM Bytecode
       │
       ▼
   Execution Engine
       │
       ├─ Stack-based execution
       ├─ Call stack management
       ├─ Garbage collection
       └─ Foreign function dispatch
       │
       ▼
   Results / Side Effects
       │
       ├─ Return values
       ├─ Engine state changes
       └─ Function calls back to C++
```

---

## Hot-Reload Process

```
File Modified (player_behavior.wren)
       │
       ▼
Wren Hot-Reload Triggered
       │
       ├─ Clear old bytecode
       ├─ Clear old script instances
       ├─ Clear function handles
       └─ Clear cached references
       │
       ▼
Recompile Script
       │
       ├─ Read updated file
       ├─ Parse and compile
       └─ Load new bytecode
       │
       ▼
Reinitialize
       │
       ├─ Call new init() if defined
       ├─ Restore game state
       └─ Resume from update()
       │
       ▼
Script Running with New Code
```

---

## Summary

The Wren scripting system provides:

1. **Clean Integration** - Single init/shutdown calls
2. **Full Engine Access** - All systems exposed via bindings
3. **Fast Iteration** - Hot-reload for rapid development
4. **Production Ready** - Optimized, tested, documented
5. **Easy to Learn** - Clear syntax, plenty of examples

This architecture enables gameplay programmers to develop game logic independently from engine developers, speeding up iteration and shipping.
