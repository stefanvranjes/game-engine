# Wren Scripting for Gameplay Logic

## Overview

Wren is a lightweight, dynamically-typed scripting language integrated into the game engine for rapid gameplay prototyping and development. Wren scripts handle gameplay logic, AI behavior, event systems, and interactive mechanics without requiring C++ recompilation.

**Key Features:**
- **Fast Execution**: Optimized VM designed for performance-critical gameplay
- **Object-Oriented**: Classes, inheritance, and polymorphism for clean game logic
- **Hot-Reloading**: Scripts can be reloaded without restarting the engine
- **Native Bindings**: Direct access to game engine types and systems
- **Fiber Support**: Built-in concurrency for complex sequences and coroutines

---

## Integration Architecture

### System Components

**WrenScriptSystem** ([WrenScriptSystem.h](../include/WrenScriptSystem.h))
- Singleton managing Wren VM lifecycle and script execution
- Initializes and registers all game engine bindings
- Handles script loading, compilation, and runtime errors
- Provides callback mechanism for native C++ function calls

**WrenScriptComponent** ([WrenScriptComponent.h](../include/WrenScriptComponent.h))
- Attaches Wren scripts to GameObjects
- Manages script lifecycle (Init, Update, Destroy)
- Bridges GameObject and Wren script instances
- Supports multiple scripts per component

**GameObject Integration**
```cpp
// C++ side
auto scriptComp = std::make_shared<WrenScriptComponent>(gameObject);
scriptComp->LoadScript("assets/scripts/player_behavior.wren");
gameObject->SetScriptComponent(scriptComp);

// Script component automatically calls:
// - init() on initialization
// - update(deltaTime) each frame
// - destroy() on cleanup
```

---

## Built-in Bindings

### GameObject

```wren
class GameObject {
    // Properties
    name              // Read-only string
    transform         // Transform component
    active            // Enable/disable GameObject
    layer             // Physics layer
    tag               // Custom tag string
    
    // Methods
    setActive(bool)           // Enable/disable
    destroy()                 // Schedule for deletion
    addComponent(type)        // Dynamically add component
    getComponent(type)        // Get component by type
    setTag(tag)              // Set custom tag
}
```

### Transform

```wren
class Transform {
    // Properties
    position          // Vec3 - local position
    rotation          // Quaternion - local rotation
    scale             // Vec3 - local scale
    parent            // Parent transform
    children          // Children array
    worldPosition     // Vec3 - world space position
    localPosition     // Vec3 - local space position
    forward           // Vec3 - forward direction
    right             // Vec3 - right direction
    up                // Vec3 - up direction
    
    // Methods
    setPosition(x, y, z)      // Set local position
    setRotation(x, y, z, w)   // Set rotation (quaternion)
    setScale(x, y, z)         // Set scale
    translate(x, y, z)        // Move relative to current position
    rotate(x, y, z)           // Rotate Euler angles
}
```

### Vec3 (Vector Math)

```wren
class Vec3 {
    construct new(x, y, z)
    
    // Properties
    x, y, z           // Components
    magnitude         // Length of vector
    normalized        // Unit vector in same direction
    
    // Methods
    dot(other)        // Dot product
    cross(other)      // Cross product
    distance(other)   // Distance to another point
    toString          // String representation
}
```

### Physics Components

```wren
foreign class RigidBody {
    // Properties
    mass                      // Body mass
    velocity                  // Linear velocity
    angularVelocity          // Rotation speed
    linearDamping            // Linear friction
    angularDamping           // Rotational friction
    isKinematic              // Non-physics controlled
    useGravity               // Affected by gravity
    
    // Methods
    setVelocity(x, y, z)     // Set velocity
    applyForce(x, y, z)      // Apply force (per frame)
    applyTorque(x, y, z)     // Apply rotation force
    applyImpulse(x, y, z)    // One-time impulse
    setKinematic(bool)       // Toggle kinematic mode
    setUseGravity(bool)      // Toggle gravity
}

foreign class Collider {
    // Properties
    isTrigger                 // Trigger vs solid
    material                  // Physics material
    shape                     // Collision shape
    
    // Methods
    setTrigger(bool)         // Toggle trigger mode
    
    // Callbacks
    onCollisionEnter(other)  // Collision started
    onCollisionStay(other)   // Collision ongoing
    onCollisionExit(other)   // Collision ended
    onTriggerEnter(other)    // Trigger entered
    onTriggerStay(other)     // Trigger stay
    onTriggerExit(other)     // Trigger exited
}
```

### Audio System

```wren
foreign class AudioSource {
    // Properties
    clip              // Current audio clip
    volume            // Playback volume (0-1)
    pitch             // Playback speed multiplier
    loop              // Loop mode
    spatialBlend      // 3D positioning (0=2D, 1=3D)
    dopplerLevel      // Doppler effect intensity
    isPlaying         // Currently playing
    
    // Methods
    play()                              // Start playback
    pause()                             // Pause playback
    stop()                              // Stop playback
    playOneShotAtPoint(clip, x, y, z, vol) // 3D one-shot
}
```

### Particles

```wren
foreign class ParticleSystem {
    // Properties
    emission          // Particles per second
    emissionRate      // Rate multiplier
    lifetime          // Particle lifespan
    speed             // Initial velocity
    size              // Particle size
    simulationSpace   // Local or world space
    
    // Methods
    play()            // Start emission
    stop()            // Stop emission
    pause()           // Pause emission
    emit(count)       // Emit specific count
}
```

### Time Management

```wren
class Time {
    static deltaTime      // Frame delta time
    static time           // Total elapsed time
    static timeScale      // Gameplay speed (1.0 = normal)
    static frameCount     // Frames since start
    static fixedDeltaTime // Physics step time
    
    static setTimeScale(scale)  // Slow motion, pause
}
```

### Input

```wren
class Input {
    static getKey(keyCode)         // Key held
    static getKeyDown(keyCode)     // Key pressed this frame
    static getKeyUp(keyCode)       // Key released this frame
    static getMouseButton(button)  // Mouse button held
    static getMouseButtonDown(button)   // Mouse pressed
    static getMouseButtonUp(button)     // Mouse released
    static getAxis(axisName)       // Analog axis (-1 to 1)
    static mousePosition           // Vec3 mouse screen position
    static isInputActive           // Input enabled
}
```

### Debugging & Math

```wren
class Debug {
    static log(message)      // Print message
    static warn(message)     // Print warning
    static error(message)    // Print error
    static drawLine(a, b, color)     // Debug line
    static drawRay(origin, dir, color)   // Debug ray
    static drawBox(center, size, color)  // Debug box
}

class Mathf {
    static pi           // 3.14159...
    static e            // 2.71828...
    
    static abs(x)                   // Absolute value
    static max(a, b)                // Maximum
    static min(a, b)                // Minimum
    static clamp(value, min, max)   // Clamp range
    static lerp(a, b, t)            // Linear interpolation
    static smoothstep(min, max, x)  // Smooth interpolation
}
```

---

## Script Lifecycle

Scripts attached to GameObjects follow a standard lifecycle:

```wren
// 1. INITIALIZATION - Called once when component is loaded
construct init() {
    System.print("Script initialized")
}

// 2. UPDATE LOOP - Called every frame with delta time
construct update(dt) {
    System.print("Frame time: %(dt)")
}

// 3. CLEANUP - Called when GameObject is destroyed
construct destroy() {
    System.print("Script cleaned up")
}
```

### Event Callbacks

Scripts can respond to physics and collision events:

```wren
// Called when two colliders first collide
onCollisionEnter(collider) {
    System.print("Hit: %(collider.gameObject.name)")
}

// Called every frame while colliding
onCollisionStay(collider) {
}

// Called when collision ends
onCollisionExit(collider) {
}

// Trigger volume callbacks
onTriggerEnter(collider) {
    System.print("Entered trigger")
}

onTriggerStay(collider) {
}

onTriggerExit(collider) {
}
```

---

## Example Scripts

### Player Movement Controller

See [player_behavior.wren](../assets/scripts/player_behavior.wren)

**Features:**
- WASD movement input handling
- Jump mechanics with ground check
- Animation state management
- Collision detection and response

```wren
class Player {
    construct new(gameObject) {
        _gameObject = gameObject
        _transform = gameObject.transform
        _rigidBody = gameObject.getComponent("RigidBody")
        _moveSpeed = 5.0
        _jumpForce = 10.0
    }
    
    update(dt) {
        handleInput(dt)
        updateMovement(dt)
    }
    
    handleInput(dt) {
        if (Input.getKey("W")) {
            // Move forward
        }
    }
}
```

### Enemy AI State Machine

See [enemy_ai.wren](../assets/scripts/enemy_ai.wren)

**Features:**
- Patrol waypoint navigation
- Player detection and chase
- Attack mechanics
- Health system and death handling

```wren
class EnemyAI {
    update(dt) {
        detectTarget()
        
        if (_state == "patrol") {
            updatePatrol(dt)
        } else if (_state == "chase") {
            updateChase(dt)
        } else if (_state == "attack") {
            updateAttack(dt)
        }
    }
}
```

### Item Collection System

See [collectible.wren](../assets/scripts/collectible.wren)

**Features:**
- Bobbing animation
- Collection with particles and sound
- Different item types (coins, health, ammo)
- Trigger volume detection

```wren
class Collectible {
    collect(player) {
        _audioSource.playOneShotAtPoint("collect", pos.x, pos.y, pos.z, 1.0)
        _particleSystem.emit(20)
        player.addCoins(_value)
    }
}
```

### Game Manager / Level Controller

See [game_manager.wren](../assets/scripts/game_manager.wren)

**Features:**
- Level state management
- Score tracking
- Win/lose conditions
- UI updates
- Level progression

```wren
class GameManager {
    update(dt) {
        _time = _time + dt
        
        if (_time > _levelTime) {
            endLevel(true)  // Won
        }
    }
}
```

### Utility Library

See [utils.wren](../assets/scripts/utils.wren)

Common helper functions for all scripts:

- **VectorUtils**: Distance, direction, lerp, clamp
- **MathUtils**: Random, sign, repeat, pingpong
- **PhysicsUtils**: Raycast, overlap queries
- **AnimationUtils**: Blending, crossfading
- **AudioUtils**: 3D sound, fading
- **GameObjectUtils**: Finding, spawning, destroying
- **DebugUtils**: Logging, assertions, profiling
- **ObjectPool**: Reusable object pooling
- **Timer**: Timed callbacks

---

## Common Patterns

### Class Structure with Constructor Pattern

```wren
class MyBehavior {
    construct new(gameObject) {
        _gameObject = gameObject
        _transform = gameObject.transform
        _initialized = false
    }
    
    init() {
        _initialized = true
    }
    
    update(dt) {
        // Update logic
    }
}

var _behavior = null

construct init() {
    _behavior = MyBehavior.new(_gameObject)
    _behavior.init()
}

construct update(dt) {
    if (_behavior) _behavior.update(dt)
}
```

### State Machine Pattern

```wren
_state = "idle"

update(dt) {
    if (_state == "idle") {
        handleIdleState()
    } else if (_state == "moving") {
        handleMovingState()
    } else if (_state == "attacking") {
        handleAttackingState()
    }
}

transitionTo(newState) {
    if (_state != newState) {
        exitState(_state)
        _state = newState
        enterState(_state)
    }
}
```

### Event System

```wren
class EventEmitter {
    construct new() {
        _listeners = {}
    }
    
    on(eventName, callback) {
        if (!_listeners.containsKey(eventName)) {
            _listeners[eventName] = []
        }
        _listeners[eventName].add(callback)
    }
    
    emit(eventName, data) {
        if (_listeners.containsKey(eventName)) {
            for (listener in _listeners[eventName]) {
                listener.call(data)
            }
        }
    }
}
```

### Object Pooling

```wren
class ProjectilePool {
    construct new(size) {
        _available = []
        for (i in 0...size) {
            var proj = Projectile.new()
            _available.add(proj)
        }
    }
    
    get() {
        if (_available.count > 0) {
            return _available.removeAt(0)
        }
        return Projectile.new()
    }
    
    release(projectile) {
        projectile.reset()
        _available.add(projectile)
    }
}
```

---

## Best Practices

### Performance

1. **Cache Component References**
   ```wren
   _transform = _gameObject.transform  // Cache this
   ```

2. **Minimize Allocations in Hot Path**
   ```wren
   // Avoid creating Vec3 every frame
   var pos = _transform.position
   _transform.setPosition(pos.x + vel.x, pos.y, pos.z)
   ```

3. **Use Object Pooling for Frequent Spawning**
   ```wren
   var projectile = _projectilePool.get()
   projectile.fire(_direction)
   ```

4. **Check Conditions Before Expensive Operations**
   ```wren
   if (_isVisible) {  // Check visibility first
       updateAnimation()
   }
   ```

### Code Quality

1. **Use Descriptive Names**
   ```wren
   _isGroundedLastFrame = true  // Better than _gnd
   ```

2. **Document Complex Logic**
   ```wren
   // Apply exponential damping to smooth movement
   _velocity = _velocity * (1.0 - _damping * dt)
   ```

3. **Separate Concerns**
   ```wren
   handleInput()
   updateMovement()
   updateAnimation()  // Keep separate from physics
   ```

4. **Use Constants**
   ```wren
   var MOVE_SPEED = 5.0
   var JUMP_FORCE = 10.0
   ```

### Debugging

1. **Use Debug.log() for Tracing**
   ```wren
   Debug.log("Player at %(getPosition())")
   ```

2. **Validate Input Parameters**
   ```wren
   takeDamage(amount) {
       if (amount < 0) {
           Debug.error("Damage cannot be negative")
           return
       }
       _health = _health - amount
   }
   ```

3. **Check Weak References**
   ```wren
   var owner = _owner.lock()
   if (!owner) return  // Owner was destroyed
   ```

---

## Integration with Engine

### From C++ Code

**Initialize Wren System:**
```cpp
WrenScriptSystem::GetInstance().Init();
```

**Attach Script to GameObject:**
```cpp
auto scriptComp = std::make_shared<WrenScriptComponent>(gameObject);
scriptComp->LoadScript("assets/scripts/player_behavior.wren");
gameObject->SetScriptComponent(scriptComp);

// During game loop:
scriptComp->Update(deltaTime);
```

**Call Custom Functions:**
```cpp
WrenScriptSystem::GetInstance().CallFunction("onGameEvent", "\"player_died\"");
```

### Script Hot-Reloading

```cpp
// In editor or debug mode:
WrenScriptSystem::GetInstance().ReloadAll();

// Or reload specific component:
scriptComponent->Reload();
```

---

## Troubleshooting

### Script Not Loading

Check file path and permissions:
```wren
bool result = WrenScriptSystem::GetInstance().RunScript("path/to/script.wren");
if (!result) {
    // Check logs for error details
}
```

### Binding Errors

Verify binding function exists:
```wren
// Make sure these exist:
construct init()      // Required for initialization
construct update(dt)  // Required for each frame
construct destroy()   // Optional cleanup
```

### Performance Issues

1. Profile with `Debug.log()` timestamps
2. Reduce physics queries per frame
3. Cache component references
4. Use object pooling for frequent allocations

### Type Mismatches

Wren is dynamically typed, but check operations match expected types:
```wren
var pos = Vec3.new(1, 2, 3)
// pos.x = "string"  // This works in Wren but may cause issues elsewhere
pos.x = 5.0          // Correct
```

---

## Advanced Features

### Coroutines with Fibers

Wren supports fiber-based coroutines for complex sequences:

```wren
class SequenceController {
    playSequence() {
        System.print("Start")
        // Wait would require integration with engine timer system
        System.print("End")
    }
}
```

### Meta-Programming

```wren
class GameObject {
    construct new(name) {
        _name = name
    }
    
    // Introspection
    getName() { _name }
}

var obj = GameObject.new("Player")
System.print(obj.getName())  // "Player"
```

### Extending Bindings

Register custom C++ functions in gameplay code:

```cpp
WrenScriptSystem::GetInstance().RegisterNativeMethod(
    "GameObject", 
    "customFunction",
    1,
    [](WrenVM* vm) {
        // Custom implementation
    }
);
```

---

## Resources

- **Wren Official**: https://wren.io
- **Wren Documentation**: https://wren.io/try
- **Game Engine Scripting**: See [ScriptSystem.h](../include/ScriptSystem.h) for Lua reference
- **Networking Scripts**: See [game-engine-multiplayer/](../game-engine-multiplayer/) for multiplayer examples

---

## Summary

Wren enables rapid gameplay development with:
- ✅ Clean, intuitive syntax for game logic
- ✅ Direct access to game engine systems
- ✅ Hot-reloading for iterative development
- ✅ Performance suitable for production gameplay
- ✅ Extensible binding system for custom functionality

Start scripting gameplay logic today and ship faster with Wren!
