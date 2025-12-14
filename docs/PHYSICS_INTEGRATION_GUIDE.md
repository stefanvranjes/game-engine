# Physics Engine Integration Guide

## Overview

The Game Engine now includes a fully integrated **Bullet3D Physics Engine** with comprehensive support for:

- **Rigid Body Dynamics**: Static, dynamic, and kinematic bodies
- **Collision Detection**: Multiple shape types with continuous collision detection
- **Kinematic Character Controller**: Optimized character movement with jumping, stepping, and slope handling
- **Force and Impulse**: Apply forces, torques, and impulses to bodies
- **Raycasting**: Cast rays into the physics world for hit detection

## Architecture

### Core Components

**PhysicsSystem** ([include/PhysicsSystem.h](../include/PhysicsSystem.h))
- Singleton manager for the Bullet3D dynamics world
- Handles initialization, simulation stepping, and cleanup
- Manages gravity and collision layers
- Provides raycast functionality

**RigidBody** ([include/RigidBody.h](../include/RigidBody.h))
- Component for physical simulation of GameObjects
- Supports Static, Dynamic, and Kinematic body types
- Applies forces, impulses, and torques
- Automatically syncs with GameObject transforms

**KinematicController** ([include/KinematicController.h](../include/KinematicController.h))
- Character-specific physics controller
- Handles grounded state, jumping, stepping
- More suitable for player/NPC movement than rigid bodies
- Prevents common physics issues with humanoid characters

**PhysicsCollisionShape** ([include/PhysicsCollisionShape.h](../include/PhysicsCollisionShape.h))
- Wrapper around Bullet3D collision shapes
- Supports: Box, Sphere, Capsule, Cylinder, Compound shapes
- Easy shape creation and modification

## Quick Start

### 1. Initialize Physics in Your Application

```cpp
// In Application::Init()
m_PhysicsSystem = std::make_unique<PhysicsSystem>();
m_PhysicsSystem->Initialize(Vec3(0, -9.81f, 0)); // Standard gravity
```

The physics system is already integrated into the main Application class.

### 2. Create a Dynamic Box

```cpp
#include "RigidBody.h"
#include "PhysicsCollisionShape.h"

// Create a GameObject
auto box = std::make_shared<GameObject>("PhysicsBox");
box->GetTransform().SetPosition(Vec3(0, 5, 0));

// Create collision shape (1 unit box, half-extents)
auto boxShape = PhysicsCollisionShape::CreateBox(Vec3(0.5f, 0.5f, 0.5f));

// Create and attach rigid body
auto rigidBody = std::make_shared<RigidBody>();
rigidBody->Initialize(BodyType::Dynamic, 1.0f, boxShape); // 1 kg mass
box->SetRigidBody(rigidBody);

// Add to scene
renderer->AddGameObject(box);
```

### 3. Create a Character Controller

```cpp
#include "KinematicController.h"

// Create character GameObject
auto character = std::make_shared<GameObject>("Player");
character->GetTransform().SetPosition(Vec3(0, 1, 0));

// Create capsule collision shape (radius, height)
auto capsuleShape = PhysicsCollisionShape::CreateCapsule(0.3f, 1.8f);

// Create and attach kinematic controller
auto controller = std::make_shared<KinematicController>();
controller->Initialize(capsuleShape, 80.0f, 0.35f); // 80 kg, 0.35m step height
character->SetKinematicController(controller);

renderer->AddGameObject(character);
```

### 4. Update Controller Each Frame

```cpp
// In your game loop or input handler:
void HandlePlayerInput(const KinematicController* controller, 
                       float moveX, float moveZ, bool jumpPressed) {
    // Set movement
    Vec3 moveDir = Vec3(moveX, 0, moveZ) * 5.0f; // 5 m/s walk speed
    controller->SetWalkDirection(moveDir);
    
    // Handle jumping
    if (jumpPressed && controller->IsGrounded()) {
        controller->Jump(Vec3(0, 10.0f, 0)); // Jump force
    }
}
```

### 5. Apply Forces to Dynamic Bodies

```cpp
auto rigidBody = gameObject->GetRigidBody();

// Apply force at center of mass
rigidBody->ApplyForce(Vec3(100, 0, 0)); // 100N force

// Apply impulse (instantaneous force)
rigidBody->ApplyImpulse(Vec3(0, 500, 0)); // Jump impulse

// Apply force at specific point
Vec3 contactPoint = gameObject->GetTransform().GetPosition() + Vec3(0.5f, 0, 0);
rigidBody->ApplyForceAtPoint(Vec3(0, 50, 0), contactPoint);
```

## Body Types

### Static Bodies
Non-moving bodies that other bodies collide with. Ideal for:
- Terrain
- Buildings
- Walls
- Static obstacles

```cpp
auto staticShape = PhysicsCollisionShape::CreateBox(Vec3(10, 0.5f, 10));
auto staticBody = std::make_shared<RigidBody>();
staticBody->Initialize(BodyType::Static, 0.0f, staticShape); // Mass is 0 for static
```

### Dynamic Bodies
Bodies affected by gravity and forces. Ideal for:
- Projectiles
- Destructible objects
- Objects affected by other bodies
- Anything that needs physics simulation

```cpp
auto dynamicShape = PhysicsCollisionShape::CreateSphere(0.5f);
auto dynamicBody = std::make_shared<RigidBody>();
dynamicBody->Initialize(BodyType::Dynamic, 2.0f, dynamicShape); // 2 kg
```

### Kinematic Bodies
Bodies moved by code but affecting dynamic bodies. Ideal for:
- Moving platforms
- Doors
- Manually controlled objects

```cpp
auto kinematicShape = PhysicsCollisionShape::CreateBox(Vec3(2, 0.5f, 5));
auto kinematicBody = std::make_shared<RigidBody>();
kinematicBody->Initialize(BodyType::Kinematic, 10.0f, kinematicShape);

// Move it manually each frame
Vec3 newPos = currentPos + Vec3(5, 0, 0) * deltaTime; // Move 5 m/s
kinematicBody->SyncTransformToPhysics(newPos, rotation);
```

## Collision Shapes

### Box
```cpp
auto boxShape = PhysicsCollisionShape::CreateBox(Vec3(1, 2, 0.5f));
// Half-extents in XYZ
```

### Sphere
```cpp
auto sphereShape = PhysicsCollisionShape::CreateSphere(1.0f);
// Radius only
```

### Capsule
```cpp
auto capsuleShape = PhysicsCollisionShape::CreateCapsule(0.5f, 2.0f);
// Ideal for character bodies
```

### Cylinder
```cpp
auto cylinderShape = PhysicsCollisionShape::CreateCylinder(0.5f, 2.0f);
// Radius and height
```

### Compound Shapes
Combine multiple shapes for complex geometries:

```cpp
auto compound = PhysicsCollisionShape::CreateCompound();

auto boxShape = PhysicsCollisionShape::CreateBox(Vec3(1, 0.5f, 1));
compound.AddChildShape(boxShape, Vec3(0, 0.5f, 0)); // Offset

auto sphereShape = PhysicsCollisionShape::CreateSphere(0.3f);
compound.AddChildShape(sphereShape, Vec3(0, 1.5f, 0)); // Head position

auto compoundBody = std::make_shared<RigidBody>();
compoundBody->Initialize(BodyType::Dynamic, 3.0f, compound);
```

## Physical Properties

### Friction
Controls how much objects slide on surfaces (0 = frictionless, 1+ = high friction):

```cpp
rigidBody->SetFriction(0.5f); // Medium friction
```

### Restitution (Bounciness)
Controls bounce intensity (0 = no bounce, 1 = perfect bounce):

```cpp
rigidBody->SetRestitution(0.8f); // Bouncy ball
```

### Damping
Simulates air/fluid resistance. Values: 0 to 1

```cpp
rigidBody->SetLinearDamping(0.1f);  // Linear air drag
rigidBody->SetAngularDamping(0.1f); // Rotational air drag
```

### Mass
Change mass at runtime. Affects inertia:

```cpp
rigidBody->SetMass(5.0f); // 5 kg
```

## Advanced Features

### Raycasting
Cast rays into the physics world for hit detection:

```cpp
Vec3 rayStart = camera->GetPosition();
Vec3 rayEnd = rayStart + camera->GetFront() * 100.0f; // 100m forward

RaycastHit hit;
if (PhysicsSystem::Get().Raycast(rayStart, rayEnd, hit)) {
    std::cout << "Hit at " << hit.point << " with normal " << hit.normal << std::endl;
    std::cout << "Distance: " << hit.distance << std::endl;
}
```

### Gravity Control
Set custom gravity or disable per-body:

```cpp
// Global gravity
PhysicsSystem::Get().SetGravity(Vec3(0, -20.0f, 0)); // Stronger gravity

// Per-body gravity
rigidBody->SetGravityEnabled(false); // Disable gravity for this body
```

### Query Active Bodies
Get simulation statistics:

```cpp
int numBodies = PhysicsSystem::Get().GetNumRigidBodies();
auto rigidBodies = PhysicsSystem::Get().GetRigidBodies();
```

### Collision Filtering
Control which bodies collide with each other:

```cpp
rigidBody->SetCollisionFilterGroup(1 << 0);  // Group 0
rigidBody->SetCollisionFilterMask(~(1 << 1)); // Ignore group 1
```

## Physics Synchronization

The engine automatically syncs physics bodies with GameObjects:

### Automatic Sync (Each Frame)
1. Physics simulation updates body positions/rotations
2. After `PhysicsSystem::Update()`, rigid bodies sync to GameObjects
3. The Renderer draws updated transforms

### Manual Sync (For Kinematic Bodies)
When you move kinematic bodies by code:

```cpp
// Move kinematic body
kinematicBody->SyncTransformToPhysics(newPos, rotation);

// Or for characters
kinematicController->SetPosition(newPos);
```

## Performance Tips

1. **Use Appropriate Body Types**
   - Use Static for fixed geometry (not Dynamic)
   - Use Kinematic for player characters (better than Dynamic for humanoids)
   - Use Dynamic only for objects that need full physics

2. **Optimize Collision Shapes**
   - Use simple shapes (box, sphere, capsule) when possible
   - Compound shapes are more expensive
   - Disable collision for objects far from player

3. **Control Sleep/Activation**
   ```cpp
   rigidBody->SetActive(false); // Put body to sleep
   ```

4. **Use Appropriate Collision Margins**
   ```cpp
   shape.SetMargin(0.04f); // Default is good for most cases
   ```

5. **Limit Active Bodies**
   - Only keep nearby bodies active
   - Disable physics for far-away objects

## Debugging

### Visualization
Enable debug drawing (planned for future):
```cpp
PhysicsSystem::Get().SetDebugDrawEnabled(true);
```

### Query Body State
```cpp
// Check if body is active
if (rigidBody->IsActive()) {
    // Body is simulating
}

// Get velocities
Vec3 linVel = rigidBody->GetLinearVelocity();
Vec3 angVel = rigidBody->GetAngularVelocity();

// Check body type
if (rigidBody->GetBodyType() == BodyType::Dynamic) {
    // Handle dynamic behavior
}

// Character-specific
if (controller->IsGrounded()) {
    // Can jump
}
```

## Common Patterns

### Jumping Platform
```cpp
auto platform = std::make_shared<GameObject>("JumpPad");
auto platformShape = PhysicsCollisionShape::CreateBox(Vec3(2, 0.5f, 2));
auto platformBody = std::make_shared<RigidBody>();
platformBody->Initialize(BodyType::Static, 0.0f, platformShape);
platform->SetRigidBody(platformBody);

// When player touches: apply upward impulse
if (playerColliding) {
    player->GetRigidBody()->ApplyImpulse(Vec3(0, 300, 0));
}
```

### Ragdoll (Multiple Bodies)
```cpp
// Create interconnected bodies for ragdoll effect
// Body 1: Head
auto headShape = PhysicsCollisionShape::CreateSphere(0.2f);
auto headBody = std::make_shared<RigidBody>();
headBody->Initialize(BodyType::Dynamic, 5.0f, headShape);

// Body 2: Torso
auto torsoShape = PhysicsCollisionShape::CreateBox(Vec3(0.3f, 0.5f, 0.2f));
auto torsoBody = std::make_shared<RigidBody>();
torsoBody->Initialize(BodyType::Dynamic, 15.0f, torsoShape);

// Connect with constraints (future feature)
// For now, you can create compound shapes
```

### Vehicle Physics
```cpp
// Simplified vehicle with kinematic body
auto vehicleShape = PhysicsCollisionShape::CreateBox(Vec3(1, 0.5f, 2));
auto vehicleBody = std::make_shared<RigidBody>();
vehicleBody->Initialize(BodyType::Kinematic, 1000.0f, vehicleShape);

// Move with input
Vec3 moveDir = Vec3(inputX, 0, inputZ) * vehicleSpeed;
vehicleBody->SyncTransformToPhysics(newPos, rotation);
```

## Troubleshooting

### Bodies Falling Through Floor
- Ensure floor collision margin is appropriate
- Check if floor has RigidBody attached
- Verify gravity direction

### Objects Floating
- Check mass/inertia values
- Verify gravity is enabled
- Check for force application

### Jerky Movement
- Reduce timestep or use more substeps
- Check for excessive forces
- Verify damping values

### Memory Leaks
- Always call `PhysicsSystem::Shutdown()` on exit
- Ensure RigidBody destructors are called
- Check for circular references with shared_ptrs

## API Reference

See headers for complete API:
- [PhysicsSystem.h](../include/PhysicsSystem.h)
- [RigidBody.h](../include/RigidBody.h)
- [KinematicController.h](../include/KinematicController.h)
- [PhysicsCollisionShape.h](../include/PhysicsCollisionShape.h)

## Dependencies

- **Bullet3D 3.24**: Physical simulation engine (MIT License)
- Integrated via CMake FetchContent

## Next Steps

Potential enhancements:
1. Constraints (hinges, springs, fixed joints)
2. Soft body dynamics (cloth, deformable meshes)
3. Fluid simulation
4. Particle physics
5. Vehicle controllers
6. Wheeled vehicle support
7. Debug visualization
8. Physics editor tools
