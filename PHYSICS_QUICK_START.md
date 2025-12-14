# Physics Engine Quick Start

## What's New

Your game engine now includes **Bullet3D Physics** with:
- ✅ Rigid body simulation (static, dynamic, kinematic)
- ✅ Kinematic character controller for player movement
- ✅ Multiple collision shapes (box, sphere, capsule, cylinder, compound)
- ✅ Force and impulse application
- ✅ Raycasting for hit detection
- ✅ Automatic GameObject synchronization
- ✅ Full profiling integration

## Files Added

**Headers (4):**
- `include/PhysicsSystem.h` - Main physics world
- `include/RigidBody.h` - Physics body component
- `include/KinematicController.h` - Character controller
- `include/PhysicsCollisionShape.h` - Collision shapes

**Implementations (3):**
- `src/PhysicsSystem.cpp`
- `src/RigidBody.cpp`
- `src/KinematicController.cpp`
- `src/PhysicsCollisionShape.cpp`

**Documentation:**
- `docs/PHYSICS_INTEGRATION_GUIDE.md` - Full guide
- `docs/PHYSICS_EXAMPLES.cpp` - 10 code examples
- `PHYSICS_INTEGRATION_STATUS.md` - Integration details

## 30-Second Quickstart

### 1. Dynamic Box
```cpp
auto shape = PhysicsCollisionShape::CreateBox(Vec3(0.5f, 0.5f, 0.5f));
auto body = std::make_shared<RigidBody>();
body->Initialize(BodyType::Dynamic, 1.0f, shape);
gameObject->SetRigidBody(body);
```

### 2. Player Character
```cpp
auto shape = PhysicsCollisionShape::CreateCapsule(0.3f, 1.8f);
auto controller = std::make_shared<KinematicController>();
controller->Initialize(shape, 80.0f, 0.35f);
gameObject->SetKinematicController(controller);
```

### 3. Player Movement
```cpp
controller->SetWalkDirection(Vec3(moveX, 0, moveZ) * 5.0f);
if (canJump) controller->Jump(Vec3(0, 10.0f, 0));
```

### 4. Apply Force
```cpp
body->ApplyForce(Vec3(100, 0, 0));
body->ApplyImpulse(Vec3(0, 500, 0));
```

### 5. Raycast
```cpp
RaycastHit hit;
if (PhysicsSystem::Get().Raycast(from, to, hit)) {
    Vec3 hitPoint = hit.point;
}
```

## Key Features

### Body Types
- **Static**: Fixed in place (terrain, buildings)
- **Dynamic**: Affected by gravity and forces (items, projectiles)
- **Kinematic**: Moved by code (platforms, doors)

### Shapes
- Box, Sphere, Capsule, Cylinder, Compound
- Collision detection with continuous collision checking
- Automatic margin handling for stability

### Physics Properties
- Mass (kg)
- Friction (0-∞)
- Restitution/Bounciness (0-1)
- Linear/Angular Damping (0-1)
- Gravity (per-body enable/disable)

### Character Controller
- Grounded detection
- Jump with vertical velocity
- Configurable step climbing
- Walk direction control
- Automatic gravity application

## How It Works

**Initialization:** Physics system initializes automatically in Application::Init()

**Update Loop:**
1. Application::Update() → Renderer::Update() (sprites/particles)
2. Application::Update() → Physics::Update() (simulate 1/60s)
3. Auto-sync: Physics positions → GameObject transforms
4. Renderer draws GameObjects with updated transforms

**Shutdown:** Physics cleanup automatic via Application destructor

## Example Workflow

```cpp
// Create floor (static)
auto floorShape = PhysicsCollisionShape::CreateBox(Vec3(10, 0.5f, 10));
auto floorBody = std::make_shared<RigidBody>();
floorBody->Initialize(BodyType::Static, 0.0f, floorShape);
floor->SetRigidBody(floorBody);

// Create player (kinematic)
auto playerShape = PhysicsCollisionShape::CreateCapsule(0.3f, 1.8f);
auto playerController = std::make_shared<KinematicController>();
playerController->Initialize(playerShape, 80.0f, 0.35f);
player->SetKinematicController(playerController);

// Create box (dynamic)
auto boxShape = PhysicsCollisionShape::CreateBox(Vec3(0.5f, 0.5f, 0.5f));
auto boxBody = std::make_shared<RigidBody>();
boxBody->Initialize(BodyType::Dynamic, 1.0f, boxShape);
box->SetRigidBody(boxBody);

// In update loop
playerController->SetWalkDirection(moveVec);
if (jumpPressed) playerController->Jump({0, 10, 0});

// Physics system auto-updates and syncs transforms
```

## Performance Tips

1. Use **Static** bodies for terrain/buildings (they're free to simulate)
2. Use **Kinematic** for characters (more stable than dynamic)
3. Use **Dynamic** only for moveable objects
4. Disable gravity on objects that don't need it
5. Put sleeping bodies in low-physics regions
6. Use simple shapes (box, sphere) instead of compound when possible
7. Limit active bodies in view

## Debugging

```cpp
// Check body properties
std::cout << body->GetMass() << std::endl;
std::cout << body->GetLinearVelocity() << std::endl;
std::cout << body->IsActive() << std::endl;

// Character controller
std::cout << controller->IsGrounded() << std::endl;
std::cout << controller->GetPosition() << std::endl;

// System stats
std::cout << PhysicsSystem::Get().GetNumRigidBodies() << std::endl;
std::cout << PhysicsSystem::Get().GetGravity() << std::endl;
```

## Common Patterns

### Falling Platform
```cpp
// Static by default, falls when triggered
body->SetMass(0.1f); // Convert to dynamic
body->SetLinearVelocity({0, -5, 0}); // Fall
```

### Moving Platform
```cpp
// Use kinematic, update position each frame
Vec3 newPos = currentPos + moveDir * deltaTime;
kinematicBody->SyncTransformToPhysics(newPos, rotation);
```

### Jump Pad
```cpp
// Static base, apply impulse on contact
if (playerTouching) {
    player->ApplyImpulse({0, 300, 0});
}
```

### Knockback
```cpp
// Apply force based on direction
Vec3 knockback = direction * force;
body->ApplyImpulse(knockback);
```

## API Cheat Sheet

### PhysicsSystem
```cpp
Initialize(Vec3 gravity)
Shutdown()
Update(float deltaTime)
SetGravity(Vec3)
GetGravity()
Raycast(Vec3 from, Vec3 to, RaycastHit&) → bool
GetNumRigidBodies() → int
```

### RigidBody
```cpp
Initialize(BodyType, float mass, PhysicsCollisionShape)
SetLinearVelocity(Vec3)
GetLinearVelocity() → Vec3
ApplyForce(Vec3)
ApplyImpulse(Vec3)
SetMass(float)
GetMass() → float
SetFriction(float)
SetRestitution(float)
SetLinearDamping(float)
SetAngularDamping(float)
SetGravityEnabled(bool)
IsActive() → bool
```

### KinematicController
```cpp
Initialize(PhysicsCollisionShape, float mass, float stepHeight)
Update(float deltaTime)
SetWalkDirection(Vec3)
Jump(Vec3 impulse)
IsGrounded() → bool
GetPosition() → Vec3
SetPosition(Vec3)
SetMaxWalkSpeed(float)
SetFallSpeed(float)
```

### PhysicsCollisionShape
```cpp
CreateBox(Vec3 halfExtents) → PhysicsCollisionShape
CreateSphere(float radius) → PhysicsCollisionShape
CreateCapsule(float radius, float height) → PhysicsCollisionShape
CreateCylinder(float radius, float height) → PhysicsCollisionShape
CreateCompound() → PhysicsCollisionShape
AddChildShape(shape, offset, rotation)
SetLocalScaling(Vec3)
SetMargin(float)
```

## Troubleshooting

**Bodies falling through floor?**
- Ensure floor has RigidBody(Static) attached
- Check collision margin (should be ~0.04)

**Character stuck on slopes?**
- Increase step height
- Check walk direction is normalized

**Jerky movement?**
- Reduce damping values
- Check if forces are being applied each frame

**Objects floating?**
- Verify gravity is enabled: `body->SetGravityEnabled(true)`
- Check mass is non-zero for dynamic bodies

**Memory leaks?**
- Ensure PhysicsSystem::Shutdown() is called
- Check RigidBody destructors clean up Bullet objects

## Next Steps

1. **Read** [docs/PHYSICS_INTEGRATION_GUIDE.md](docs/PHYSICS_INTEGRATION_GUIDE.md) for comprehensive guide
2. **Study** [docs/PHYSICS_EXAMPLES.cpp](docs/PHYSICS_EXAMPLES.cpp) for working code
3. **Experiment** with shapes and body types
4. **Profile** with `SCOPED_PROFILE("Physics::Update")` integration
5. **Extend** with constraints, soft bodies, vehicles as needed

## Build System

No manual setup needed! Bullet3D is automatically:
- Fetched from GitHub (v3.24)
- Built as static library
- Linked to game engine
- Ready to use

Just rebuild with `build.bat` or CMake - physics is included!

## License

- **Game Engine**: Your project license
- **Bullet3D**: zlib License (free for commercial use)

---

**Physics integration complete!** Your engine is ready for physics-based gameplay. 🎮
