# Wren Scripting Quick Reference

## Quick Start

### 1. Create a Script File
```wren
// assets/scripts/my_behavior.wren

class MyBehavior {
    construct new(gameObject) {
        _gameObject = gameObject
        _transform = gameObject.transform
    }
    
    update(dt) {
        // Game logic here
    }
}

var _behavior = null

construct init() {
    _behavior = MyBehavior.new(_gameObject)
}

construct update(dt) {
    if (_behavior) _behavior.update(dt)
}

construct destroy() {
    _behavior = null
}
```

### 2. Attach to GameObject (C++)
```cpp
auto scriptComp = std::make_shared<WrenScriptComponent>(gameObject);
scriptComp->LoadScript("assets/scripts/my_behavior.wren");
gameObject->SetScriptComponent(scriptComp);
```

### 3. Script Lifecycle
- `init()` - Called once when script loads
- `update(dt)` - Called every frame with delta time
- `destroy()` - Called when GameObject is destroyed

---

## Essential Bindings Cheat Sheet

### Transform & Position
```wren
var pos = _transform.position              // Get Vec3
_transform.setPosition(x, y, z)            // Set position
_transform.translate(dx, dy, dz)           // Move relative
var worldPos = _transform.worldPosition    // Get world coords
```

### Movement
```wren
var rb = _gameObject.getComponent("RigidBody")
rb.setVelocity(vx, vy, vz)                 // Direct velocity
rb.applyForce(fx, fy, fz)                  // Force per frame
rb.applyImpulse(ix, iy, iz)                // One-time impulse
rb.mass = 10.0                             // Set mass
```

### Input
```wren
if (Input.getKey("W")) { }                 // Key held
if (Input.getKeyDown("Space")) { }         // Key pressed
if (Input.getMouseButton(0)) { }           // Left click held
if (Input.getMouseButtonDown(0)) { }       // Left click
var axis = Input.getAxis("Horizontal")     // Analog input
```

### Animation
```wren
var anim = _gameObject.getComponent("Animator")
anim.play("walk")                          // Play animation
anim.play("attack", 1.5)                   // With speed
```

### Audio
```wren
var audio = _gameObject.getComponent("AudioSource")
audio.play()                               // Start playing
audio.stop()                               // Stop
audio.volume = 0.5                         // 0-1 range
audio.pitch = 1.2                          // Speed
audio.playOneShotAtPoint("sfx", x, y, z, 1.0)  // 3D sound
```

### Particles
```wren
var particles = _gameObject.getComponent("ParticleSystem")
particles.play()                           // Start emission
particles.emit(50)                         // Emit 50 particles
particles.stop()                           // Stop
```

### Physics Collision
```wren
onCollisionEnter(collider) {
    System.print("Hit: %(collider.gameObject.name)")
}

onTriggerEnter(collider) {
    System.print("Entered trigger zone")
}
```

### Math & Vectors
```wren
var v1 = Vec3.new(1, 2, 3)
var v2 = Vec3.new(4, 5, 6)
var dist = (v1 - v2).magnitude             // Distance
var dir = (v2 - v1).normalized             // Direction
var dot = v1.dot(v2)                       // Dot product
var cross = v1.cross(v2)                   // Cross product

var lerped = Vec3Utils.lerp(v1, v2, 0.5)   // Interpolate
```

### Time
```wren
var dt = Time.deltaTime                    // Frame time
var elapsed = Time.time                    // Total time
Time.setTimeScale(0.5)                     // Slow motion
var fps = 1.0 / Time.deltaTime             // Calculate FPS
```

### Debugging
```wren
Debug.log("Value: %(value)")               // Print to console
Debug.warn("Warning message")              // Yellow text
Debug.error("Error message")               // Red text
Debug.drawLine(from, to, "red")            // Debug visualization
```

---

## Common Script Patterns

### Player Controller
```wren
class PlayerController {
    construct new(gameObject) {
        _gameObject = gameObject
        _speed = 5.0
    }
    
    update(dt) {
        var moveX = 0.0
        if (Input.getKey("W")) moveX = moveX + 1
        if (Input.getKey("S")) moveX = moveX - 1
        
        var vel = _gameObject.getComponent("RigidBody").velocity
        _gameObject.getComponent("RigidBody").setVelocity(moveX * _speed, vel.y, 0)
    }
}
```

### Enemy AI
```wren
class SimpleAI {
    construct new(gameObject) {
        _gameObject = gameObject
        _state = "idle"
    }
    
    update(dt) {
        if (_state == "idle") {
            patrol()
        } else if (_state == "chase") {
            chasePlayer()
        }
    }
    
    patrol() {
        // Move in pattern
    }
    
    chasePlayer() {
        // Follow target
    }
}
```

### Item Pickup
```wren
onTriggerEnter(collider) {
    if (collider.gameObject.tag == "Player") {
        System.print("Collected!")
        _gameObject.destroy()
    }
}
```

### Event-Driven Logic
```wren
class EventSystem {
    construct new() {
        _events = {}
    }
    
    on(name, callback) {
        if (!_events.containsKey(name)) {
            _events[name] = []
        }
        _events[name].add(callback)
    }
    
    emit(name, data) {
        if (_events.containsKey(name)) {
            for (callback in _events[name]) {
                callback.call(data)
            }
        }
    }
}
```

---

## Syntax Quick Reference

### Variables
```wren
var x = 10
var name = "Player"
var position = Vec3.new(1, 2, 3)
```

### Control Flow
```wren
if (condition) {
    // ...
} else if (other) {
    // ...
} else {
    // ...
}

for (i in 0...10) {      // 0 to 9
    System.print(i)
}

for (i in 10..1) {       // 10 down to 1
    System.print(i)
}

while (condition) {
    // ...
}
```

### Functions
```wren
foo(x, y) {
    return x + y
}

doThing { |param|
    System.print(param)
}
```

### Classes
```wren
class MyClass {
    construct new(name) {
        _name = name
    }
    
    getName { _name }
    
    setName(name) {
        _name = name
    }
    
    static create() {
        return MyClass.new("Default")
    }
}
```

### Strings
```wren
var msg = "Hello"
System.print("Message: %(msg)")  // Interpolation
var upper = msg.toUpperCase
var count = msg.count
```

### Lists & Maps
```wren
var list = [1, 2, 3]
list.add(4)
var first = list[0]

var map = {"name": "Player", "health": 100}
var name = map["name"]
map["level"] = 5
```

---

## Performance Tips

1. **Cache References** - Don't call getComponent() every frame
   ```wren
   _rb = _gameObject.getComponent("RigidBody")  // Once
   _rb.setVelocity(...)                          // Many times
   ```

2. **Minimize Allocations** - Reuse vectors and objects
   ```wren
   // Bad: Creates new Vec3 every frame
   _transform.setPosition(x + 1, y, z)
   
   // Better: Modify existing
   var pos = _transform.position
   _transform.setPosition(pos.x + 1, pos.y, pos.z)
   ```

3. **Use Object Pooling** - For frequently spawned objects
   ```wren
   var projectile = _pool.get()  // Reuse
   projectile.fire(_direction)
   ```

4. **Batch Similar Operations** - Group related work
   ```wren
   // In update: check visibility first
   if (!isVisible()) return
   updateAnimation()  // Skip expensive operations
   ```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `null` reference error | Check if component/GameObject exists before use |
| Script not loading | Verify file path is correct and readable |
| Updates not running | Check `construct update(dt)` signature matches |
| Physics not working | Verify RigidBody exists and mass > 0 |
| Input not responding | Check Input.isInputActive and correct key names |
| No sound | Check AudioSource exists and volume > 0 |
| Low performance | Cache references, reduce allocations, profile with Debug.log |

---

## File Structure

```
assets/
├── scripts/
│   ├── player_behavior.wren       # Player controller
│   ├── enemy_ai.wren               # Enemy AI
│   ├── collectible.wren            # Item pickup
│   ├── game_manager.wren           # Level manager
│   └── utils.wren                  # Helper functions
├── models/
├── audio/
└── particles/
```

---

## Example: Complete Player Script

```wren
class Player {
    construct new(gameObject) {
        _gameObject = gameObject
        _transform = gameObject.transform
        _rb = gameObject.getComponent("RigidBody")
        _anim = gameObject.getComponent("Animator")
        
        _speed = 5.0
        _jumpPower = 5.0
        _isGrounded = true
    }
    
    init() {
        System.print("Player ready")
    }
    
    update(dt) {
        handleInput()
        updateAnimation()
    }
    
    handleInput() {
        var moveX = 0.0
        
        if (Input.getKey("W")) moveX = moveX + 1
        if (Input.getKey("S")) moveX = moveX - 1
        if (Input.getKey("A")) moveX = moveX - 1
        if (Input.getKey("D")) moveX = moveX + 1
        
        var vel = _rb.velocity
        _rb.setVelocity(moveX * _speed, vel.y, 0)
        
        if (Input.getKeyDown("Space") && _isGrounded) {
            _rb.applyImpulse(0, _jumpPower, 0)
            _isGrounded = false
        }
    }
    
    updateAnimation() {
        _anim.play(_isGrounded ? "walk" : "jump")
    }
    
    onCollisionEnter(collider) {
        if (collider.gameObject.tag == "Ground") {
            _isGrounded = true
        }
    }
}

var _player = null

construct init() {
    _player = Player.new(_gameObject)
    _player.init()
}

construct update(dt) {
    _player.update(dt)
}
```

---

## More Examples

- **Full Player**: [player_behavior.wren](../assets/scripts/player_behavior.wren)
- **Enemy AI**: [enemy_ai.wren](../assets/scripts/enemy_ai.wren)
- **Collectibles**: [collectible.wren](../assets/scripts/collectible.wren)
- **Game Manager**: [game_manager.wren](../assets/scripts/game_manager.wren)
- **Utilities**: [utils.wren](../assets/scripts/utils.wren)

For detailed documentation, see [WREN_SCRIPTING_GUIDE.md](./WREN_SCRIPTING_GUIDE.md).
