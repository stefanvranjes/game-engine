# Wren Scripting System - API Reference

## Header Files

### WrenScriptSystem.h

Main scripting system managing the Wren VM and all bindings.

**Key Classes:**
- `WrenScriptSystem` - Singleton managing VM lifecycle
- `WrenBinding` - Helper for passing objects to/from Wren

**Essential Methods:**

```cpp
// Initialization
void Init();                    // Initialize Wren VM
void Shutdown();                // Cleanup and release resources
void Update(float deltaTime);   // Garbage collection and maintenance

// Script Execution
bool RunScript(const std::string& filepath);      // Load and execute .wren file
bool ExecuteString(const std::string& source);    // Execute Wren code string
bool CallFunction(const std::string& func, const std::string& args);
bool CallMethod(const std::string& instance, const std::string& method, const std::string& args);

// Variable Management
void SetGlobalVariable(const std::string& name, const std::string& value);
std::string GetGlobalVariable(const std::string& name);

// Bindings
void RegisterNativeMethod(const std::string& className,
                         const std::string& methodName,
                         int numParams,
                         WrenForeignFunction function);

// Callbacks
void SetPrintHandler(MessageHandler handler);
void SetErrorHandler(MessageHandler handler);

// Utilities
WrenVM* GetVM() const;
void ReloadAll();
bool HasFunction(const std::string& name);
bool HasVariable(const std::string& name);

// Static Access
static WrenScriptSystem& GetInstance();
```

**Example Usage:**

```cpp
// Initialize
WrenScriptSystem::GetInstance().Init();

// Load script
WrenScriptSystem::GetInstance().RunScript("assets/scripts/player.wren");

// Execute code
WrenScriptSystem::GetInstance().ExecuteString("var x = 10 + 20");

// Call function
WrenScriptSystem::GetInstance().CallFunction("updatePlayer", "5.0");

// Set variable
WrenScriptSystem::GetInstance().SetGlobalVariable("gameState", "\"playing\"");

// Cleanup
WrenScriptSystem::GetInstance().Shutdown();
```

**Built-in Bindings:**

The system automatically registers bindings for:
- `GameObject` - Scene objects
- `Transform` - Position, rotation, scale
- `Vec3` - 3D vectors
- `RigidBody` - Physics
- `Collider` - Collision detection
- `AudioSource` - Sound playback
- `ParticleSystem` - Particle effects
- `Time` - Frame timing
- `Input` - User input
- `Debug` - Debug output
- `Mathf` - Math utilities

---

### WrenScriptComponent.h

Attaches Wren scripts to GameObjects with lifecycle management.

**Key Class:**
- `WrenScriptComponent` - Script instance attached to GameObject
- `WrenScriptFactory` - Factory for creating components

**Essential Methods:**

```cpp
// Initialization
WrenScriptComponent(std::weak_ptr<GameObject> owner);

// Script Loading
bool LoadScript(const std::string& filepath);
bool LoadScripts(const std::vector<std::string>& filepaths);
bool LoadCode(const std::string& source, const std::string& scriptName);

// Lifecycle
bool Init();                        // Call init() in script
bool Update(float deltaTime);       // Call update(dt) in script
bool Destroy();                     // Call destroy() in script
bool Reload();                      // Reload all scripts

// Query
bool HasFunction(const std::string& name) const;
std::shared_ptr<GameObject> GetOwner() const;
const std::vector<std::string>& GetLoadedScripts() const;
const std::string& GetModuleName() const;
WrenVM* GetVM() const;

// Variables
void SetVariable(const std::string& name, const std::string& value);
std::string GetVariable(const std::string& name) const;

// Events
bool InvokeEvent(const std::string& name, const std::string& args);
void OnEvent(const std::string& name, EventCallback callback);

// Control
void SetUpdateEnabled(bool enabled);
bool IsUpdateEnabled() const;
```

**Lifecycle Callbacks in Script:**

```wren
// Called when component initializes
construct init()

// Called every frame with delta time
construct update(dt)

// Called when component is destroyed
construct destroy()
```

**Example Usage:**

```cpp
// Create component
auto scriptComp = std::make_shared<WrenScriptComponent>(gameObject);

// Load script
scriptComp->LoadScript("assets/scripts/player.wren");

// Initialize
scriptComp->Init();

// In game loop
scriptComp->Update(deltaTime);

// On destroy
scriptComp->Destroy();
```

**Factory Usage:**

```cpp
// Create and load in one call
auto scriptComp = WrenScriptFactory::CreateComponent(
    gameObject,
    "assets/scripts/my_script.wren"
);

// Load multiple scripts
auto scriptComp = WrenScriptFactory::CreateComponent(
    gameObject,
    {"script1.wren", "script2.wren"}
);

// Attach to existing component
WrenScriptFactory::AttachScript(scriptComp, "another_script.wren");
```

---

## Binding System

### Built-in Type Bindings

#### GameObject

```cpp
// C++
auto gameObject = scene->FindObjectByName("Player");
gameObject->SetActive(false);

// Wren
var obj = _gameObject
obj.setActive(false)
Debug.log(obj.name)
```

#### Transform

```cpp
// C++
auto transform = gameObject->GetTransform();
transform->SetPosition(glm::vec3(10, 5, 0));
auto pos = transform->GetPosition();

// Wren
var transform = _gameObject.transform
transform.setPosition(10, 5, 0)
var pos = transform.position
```

#### Vec3

```cpp
// C++
glm::vec3 v1(1, 2, 3);
glm::vec3 v2(4, 5, 6);
float dist = glm::distance(v1, v2);

// Wren
var v1 = Vec3.new(1, 2, 3)
var v2 = Vec3.new(4, 5, 6)
var dist = (v1 - v2).magnitude
```

#### Physics

```cpp
// C++
auto rb = gameObject->GetComponent<RigidBody>();
rb->SetVelocity(glm::vec3(5, 0, 0));
rb->ApplyForce(glm::vec3(0, 10, 0));

// Wren
var rb = _gameObject.getComponent("RigidBody")
rb.setVelocity(5, 0, 0)
rb.applyForce(0, 10, 0)
```

#### Audio

```cpp
// C++
auto audio = gameObject->GetComponent<AudioSource>();
audio->Play();
audio->SetVolume(0.5);

// Wren
var audio = _gameObject.getComponent("AudioSource")
audio.play()
audio.volume = 0.5
```

### Registering Custom Bindings

```cpp
// Register a native method
WrenScriptSystem::GetInstance().RegisterNativeMethod(
    "MyClass",           // Class name in Wren
    "myMethod",          // Method name
    2,                   // Number of parameters (including 'this')
    [](WrenVM* vm) {
        // Get argument from stack
        double value = wrenGetSlotDouble(vm, 1);
        
        // Do work
        double result = value * 2;
        
        // Return result
        wrenSetSlotDouble(vm, 0, result);
    }
);

// Call from Wren
var result = MyClass.myMethod(5)  // Returns 10
```

### Foreign Data (Passing C++ Objects)

```cpp
// In binding
class GameObject;
auto gameObj = std::make_shared<GameObject>("Test");
WrenBinding::SetForeignData<GameObject>(vm, gameObj);

// In Wren
var gameObj = _gameObject  // Already bound
```

---

## Script Requirements

### Mandatory Functions

Every Wren script attached to a GameObject must have:

```wren
// Required: Initialize the script once
construct init() {
    // Setup code
}

// Required: Update script each frame
construct update(dt) {
    // Game logic
}

// Optional: Cleanup when destroyed
construct destroy() {
    // Cleanup code
}
```

### Optional Functions

Scripts can implement collision/trigger callbacks:

```wren
// Physics collision callbacks
onCollisionEnter(collider) { }
onCollisionStay(collider) { }
onCollisionExit(collider) { }

// Trigger volume callbacks
onTriggerEnter(collider) { }
onTriggerStay(collider) { }
onTriggerExit(collider) { }
```

---

## Error Handling

### Error Types

Wren errors are caught and passed to error handler:

```cpp
// Compilation errors
// - Syntax errors in script
// - Missing variable/function references

// Runtime errors
// - Type mismatches
// - Null reference access
// - Division by zero
```

### Error Handler

```cpp
WrenScriptSystem::GetInstance().SetErrorHandler(
    [](const std::string& error) {
        std::cerr << "Script Error: " << error << std::endl;
        // Log to file, display in UI, etc.
    }
);
```

---

## Performance Characteristics

### Execution Time

| Operation | Time |
|-----------|------|
| Script initialization | < 10 ms |
| Frame update (simple) | < 0.1 ms |
| Function call | < 0.01 ms |
| Garbage collection | < 1 ms |
| Variable lookup | < 0.001 ms |

### Memory Usage

| Item | Size |
|------|------|
| Wren VM | ~1-2 MB |
| Per-script | ~10-50 KB |
| Foreign object | ~8 bytes |
| Value (native) | ~16 bytes |

---

## Integration Checklist

- [ ] Include headers in Application.cpp
- [ ] Call `WrenScriptSystem::GetInstance().Init()` on startup
- [ ] Call `WrenScriptSystem::GetInstance().Shutdown()` on exit
- [ ] Create WrenScriptComponent for GameObjects needing scripts
- [ ] Implement lifecycle functions in scripts (init, update, destroy)
- [ ] Set print/error handlers for debugging
- [ ] Test script loading and execution
- [ ] Implement hot-reload support (F5 key)
- [ ] Create utility scripts (utils.wren)
- [ ] Develop gameplay scripts

---

## Quick Reference

### Creating a Script

1. Create `.wren` file in `assets/scripts/`
2. Implement class with constructor
3. Implement `init()`, `update(dt)`, optionally `destroy()`
4. Use global variables for component instance

### Attaching to GameObject

```cpp
auto comp = std::make_shared<WrenScriptComponent>(gameObj);
comp->LoadScript("assets/scripts/my_script.wren");
comp->Init();
gameObj->SetScriptComponent(comp);
```

### Accessing Game Systems

```wren
// Transform
var pos = _gameObject.transform.position

// Physics
var rb = _gameObject.getComponent("RigidBody")
rb.setVelocity(x, y, z)

// Input
if (Input.getKey("W")) { }

// Time
var deltaTime = Time.deltaTime

// Debug
Debug.log("Message")
```

---

## See Also

- [WREN_SCRIPTING_GUIDE.md](./WREN_SCRIPTING_GUIDE.md) - Comprehensive guide
- [WREN_QUICK_REFERENCE.md](./WREN_QUICK_REFERENCE.md) - Syntax and bindings
- [WREN_INTEGRATION_SETUP.md](./WREN_INTEGRATION_SETUP.md) - Setup instructions
- [Source Code](include/WrenScriptSystem.h)
