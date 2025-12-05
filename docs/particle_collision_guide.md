# Particle Collision System - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Collision Shapes](#collision-shapes)
4. [Particle Properties](#particle-properties)
5. [Usage Examples](#usage-examples)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## Introduction

The particle collision system adds realistic physics-based interactions to your particle effects. Particles can collide with various shapes (planes, spheres, boxes) and optionally with each other, creating dynamic and believable visual effects.

### Features
- ✅ **Three collision shapes**: Plane, Sphere, and Box
- ✅ **Physics-based response**: Realistic bounce and friction
- ✅ **Particle-to-particle collisions**: Optional with spatial grid optimization
- ✅ **Per-particle properties**: Customizable mass, restitution, and friction
- ✅ **Easy integration**: Simple API for adding collision shapes

## Quick Start

### Basic Ground Collision

```cpp
// Create a particle emitter
auto emitter = ParticleEmitter::CreateSparks(Vec3(0, 10, 0));

// Add a ground plane (normal pointing up, at y=0)
auto ground = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);
emitter->AddCollisionShape(ground);

// Adjust particle physics for better bounce
emitter->SetParticleRestitution(0.7f);  // Bouncy
emitter->SetParticleFriction(0.2f);     // Some friction

// Add to particle system
particleSystem->AddEmitter(emitter);
```

## Collision Shapes

### CollisionPlane

An infinite plane defined by a normal vector and distance from origin.

**Constructor:**
```cpp
CollisionPlane(const Vec3& normal, float distance);
```

**Parameters:**
- `normal`: Direction the plane faces (should be normalized)
- `distance`: Distance from origin along the normal

**Examples:**
```cpp
// Ground plane at y=0
auto ground = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);

// Ceiling at y=10
auto ceiling = std::make_shared<CollisionPlane>(Vec3(0, -1, 0), -10.0f);

// Wall facing +X at x=5
auto wall = std::make_shared<CollisionPlane>(Vec3(1, 0, 0), 5.0f);
```

### CollisionSphere

A spherical obstacle defined by center position and radius.

**Constructor:**
```cpp
CollisionSphere(const Vec3& center, float radius);
```

**Parameters:**
- `center`: Center position of the sphere
- `radius`: Radius of the sphere

**Methods:**
```cpp
void SetCenter(const Vec3& center);  // Update sphere position
```

**Example:**
```cpp
// Sphere obstacle at origin with radius 2
auto sphere = std::make_shared<CollisionSphere>(Vec3(0, 5, 0), 2.0f);
emitter->AddCollisionShape(sphere);

// Update sphere position dynamically
sphere->SetCenter(Vec3(2, 5, 0));
```

### CollisionBox

An axis-aligned bounding box defined by minimum and maximum corners.

**Constructor:**
```cpp
CollisionBox(const Vec3& min, const Vec3& max);
```

**Parameters:**
- `min`: Minimum corner (bottom-left-back)
- `max`: Maximum corner (top-right-front)

**Methods:**
```cpp
void SetBounds(const Vec3& min, const Vec3& max);  // Update box bounds
```

**Example:**
```cpp
// Box from (-5,-5,-5) to (5,5,5)
auto box = std::make_shared<CollisionBox>(Vec3(-5, -5, -5), Vec3(5, 5, 5));
emitter->AddCollisionShape(box);
```

## Particle Properties

Each particle has three collision-related properties:

### Mass
Controls how particles respond to collisions with each other.
- **Default**: 1.0
- **Range**: > 0.0
- **Effect**: Heavier particles push lighter ones more

```cpp
emitter->SetParticleMass(2.0f);  // Heavy particles
```

### Restitution (Bounciness)
Controls how much energy is retained after collision.
- **Default**: 0.5
- **Range**: 0.0 (no bounce) to 1.0 (perfect bounce)
- **Effect**: Higher values = more bouncy

```cpp
emitter->SetParticleRestitution(0.9f);  // Very bouncy
emitter->SetParticleRestitution(0.1f);  // Barely bounces
```

### Friction
Controls energy loss from surface friction.
- **Default**: 0.1
- **Range**: 0.0 (no friction) to 1.0 (high friction)
- **Effect**: Higher values = more energy loss

```cpp
emitter->SetParticleFriction(0.5f);   // High friction
emitter->SetParticleFriction(0.01f);  // Low friction (slippery)
```

## Usage Examples

### Example 1: Bouncing Sparks

Create sparks that bounce off the ground with realistic physics.

```cpp
auto sparks = ParticleEmitter::CreateSparks(Vec3(0, 5, 0));
sparks->SetGravity(Vec3(0, -9.8f, 0));
sparks->SetParticleRestitution(0.6f);
sparks->SetParticleFriction(0.3f);

// Ground plane
auto ground = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);
sparks->AddCollisionShape(ground);

particleSystem->AddEmitter(sparks);
```

### Example 2: Contained Fire Effect

Fire particles contained within a box.

```cpp
auto fire = ParticleEmitter::CreateFire(Vec3(0, 2, 0));

// Create a box container
auto container = std::make_shared<CollisionBox>(
    Vec3(-3, 0, -3),  // Min
    Vec3(3, 6, 3)     // Max
);
fire->AddCollisionShape(container);

particleSystem->AddEmitter(fire);
```

### Example 3: Obstacle Course

Particles navigate around multiple obstacles.

```cpp
auto emitter = std::make_shared<ParticleEmitter>(Vec3(-10, 10, 0), 500);
emitter->SetVelocityRange(Vec3(2, -2, -1), Vec3(4, 2, 1));
emitter->SetGravity(Vec3(0, -5, 0));
emitter->SetParticleRestitution(0.7f);

// Ground
auto ground = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);
emitter->AddCollisionShape(ground);

// Sphere obstacles
auto sphere1 = std::make_shared<CollisionSphere>(Vec3(-5, 3, 0), 1.5f);
auto sphere2 = std::make_shared<CollisionSphere>(Vec3(0, 4, 0), 1.5f);
auto sphere3 = std::make_shared<CollisionSphere>(Vec3(5, 3, 0), 1.5f);
emitter->AddCollisionShape(sphere1);
emitter->AddCollisionShape(sphere2);
emitter->AddCollisionShape(sphere3);

particleSystem->AddEmitter(emitter);
```

### Example 4: Particle-to-Particle Collisions

Enable particles to collide with each other for fluid-like behavior.

```cpp
auto emitter = std::make_shared<ParticleEmitter>(Vec3(0, 10, 0), 300);
emitter->SetSpawnRate(50.0f);
emitter->SetGravity(Vec3(0, -9.8f, 0));
emitter->SetParticleMass(1.0f);
emitter->SetParticleRestitution(0.8f);

// Enable particle-to-particle collisions
emitter->SetEnableParticleCollisions(true);
emitter->SetParticleCollisionRadius(1.0f);

// Ground to collect particles
auto ground = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);
emitter->AddCollisionShape(ground);

particleSystem->AddEmitter(emitter);
```

### Example 5: Dynamic Collision Shapes

Update collision shapes at runtime for animated obstacles.

```cpp
// Create sphere obstacle
auto movingSphere = std::make_shared<CollisionSphere>(Vec3(0, 5, 0), 2.0f);
emitter->AddCollisionShape(movingSphere);

// In your update loop:
void Update(float deltaTime) {
    static float time = 0.0f;
    time += deltaTime;
    
    // Move sphere in a circle
    float x = cos(time) * 5.0f;
    float z = sin(time) * 5.0f;
    movingSphere->SetCenter(Vec3(x, 5, z));
    
    particleSystem->Update(deltaTime);
}
```

## Performance Optimization

### Particle-to-Particle Collisions

Particle-to-particle collisions use spatial grid optimization but can still be expensive with many particles.

**Performance Tips:**

1. **Limit particle count** when using particle-to-particle collisions:
   ```cpp
   // Good: 200-500 particles with collisions
   auto emitter = std::make_shared<ParticleEmitter>(Vec3(0, 10, 0), 300);
   emitter->SetEnableParticleCollisions(true);
   ```

2. **Adjust collision radius** to reduce checks:
   ```cpp
   // Smaller radius = fewer collision checks
   emitter->SetParticleCollisionRadius(0.5f);
   ```

3. **Use selectively**: Enable only for key effects, not all emitters:
   ```cpp
   // Enable for main effect
   mainEmitter->SetEnableParticleCollisions(true);
   
   // Disable for background effects
   backgroundEmitter->SetEnableParticleCollisions(false);
   ```

### Collision Shape Count

Multiple collision shapes have minimal performance impact. Feel free to use as many as needed:

```cpp
// No problem having many shapes
for (int i = 0; i < 10; i++) {
    auto sphere = std::make_shared<CollisionSphere>(
        Vec3(i * 2, 3, 0), 1.0f
    );
    emitter->AddCollisionShape(sphere);
}
```

### Recommended Settings

| Particle Count | Particle Collisions | Notes |
|----------------|---------------------|-------|
| < 500 | ✅ Safe | Good performance |
| 500-1000 | ⚠️ Careful | May impact FPS |
| > 1000 | ❌ Avoid | Significant performance hit |

## Troubleshooting

### Particles Pass Through Shapes

**Problem**: Particles move through collision shapes without bouncing.

**Solutions**:
1. Check that collision shapes are added to the emitter:
   ```cpp
   emitter->AddCollisionShape(ground);  // Don't forget this!
   ```

2. Verify particle size vs collision radius:
   ```cpp
   // Particle size affects collision detection
   emitter->SetSizeRange(0.5f, 1.0f);
   ```

3. Ensure deltaTime is reasonable (not too large):
   ```cpp
   // Very large deltaTime can cause tunneling
   particleSystem->Update(deltaTime);  // Should be ~0.016 for 60fps
   ```

### Particles Don't Bounce

**Problem**: Particles stick to surfaces instead of bouncing.

**Solution**: Increase restitution:
```cpp
emitter->SetParticleRestitution(0.7f);  // Higher = more bounce
```

### Particles Bounce Too Much

**Problem**: Particles bounce endlessly without losing energy.

**Solutions**:
1. Reduce restitution:
   ```cpp
   emitter->SetParticleRestitution(0.3f);  // Less bouncy
   ```

2. Increase friction:
   ```cpp
   emitter->SetParticleFriction(0.5f);  // More energy loss
   ```

### Poor Performance with Particle Collisions

**Problem**: FPS drops when enabling particle-to-particle collisions.

**Solutions**:
1. Reduce particle count:
   ```cpp
   auto emitter = std::make_shared<ParticleEmitter>(Vec3(0, 10, 0), 200);
   ```

2. Decrease spawn rate:
   ```cpp
   emitter->SetSpawnRate(30.0f);  // Fewer particles per second
   ```

3. Reduce collision radius:
   ```cpp
   emitter->SetParticleCollisionRadius(0.5f);
   ```

### Particles Explode on Collision

**Problem**: Particles gain excessive velocity after collision.

**Solution**: This usually indicates very high restitution or mass imbalance. Adjust:
```cpp
emitter->SetParticleRestitution(0.5f);  // Moderate bounce
emitter->SetParticleMass(1.0f);         // Standard mass
```

## Advanced Topics

### Combining Multiple Collision Types

Create complex environments by combining different collision shapes:

```cpp
auto emitter = ParticleEmitter::CreateMagic(Vec3(0, 8, 0));

// Floor
auto floor = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);

// Walls (box without top)
auto wallLeft = std::make_shared<CollisionPlane>(Vec3(1, 0, 0), 10.0f);
auto wallRight = std::make_shared<CollisionPlane>(Vec3(-1, 0, 0), 10.0f);
auto wallFront = std::make_shared<CollisionPlane>(Vec3(0, 0, 1), 10.0f);
auto wallBack = std::make_shared<CollisionPlane>(Vec3(0, 0, -1), 10.0f);

// Central obstacle
auto centerSphere = std::make_shared<CollisionSphere>(Vec3(0, 3, 0), 2.0f);

emitter->AddCollisionShape(floor);
emitter->AddCollisionShape(wallLeft);
emitter->AddCollisionShape(wallRight);
emitter->AddCollisionShape(wallFront);
emitter->AddCollisionShape(wallBack);
emitter->AddCollisionShape(centerSphere);
```

### Custom Particle Properties

Set different physics properties for different particle types:

```cpp
// Heavy, non-bouncy particles (like rain)
auto rain = std::make_shared<ParticleEmitter>(Vec3(0, 15, 0), 1000);
rain->SetParticleMass(2.0f);
rain->SetParticleRestitution(0.1f);
rain->SetParticleFriction(0.1f);

// Light, bouncy particles (like bubbles)
auto bubbles = std::make_shared<ParticleEmitter>(Vec3(0, 1, 0), 300);
bubbles->SetParticleMass(0.5f);
bubbles->SetParticleRestitution(0.9f);
bubbles->SetParticleFriction(0.05f);
```

## API Quick Reference

### Collision Shape Management
```cpp
void AddCollisionShape(std::shared_ptr<CollisionShape> shape);
void RemoveCollisionShape(std::shared_ptr<CollisionShape> shape);
void ClearCollisionShapes();
```

### Particle Physics Properties
```cpp
void SetParticleMass(float mass);
void SetParticleRestitution(float restitution);
void SetParticleFriction(float friction);
```

### Particle-to-Particle Collisions
```cpp
void SetEnableParticleCollisions(bool enable);
void SetParticleCollisionRadius(float radius);
bool GetEnableParticleCollisions() const;
float GetParticleCollisionRadius() const;
```

### Collision Shape Constructors
```cpp
CollisionPlane(const Vec3& normal, float distance);
CollisionSphere(const Vec3& center, float radius);
CollisionBox(const Vec3& min, const Vec3& max);
```

---

**Need Help?** Check the implementation details in the source files:
- [CollisionShape.h](../include/CollisionShape.h) - Collision shape definitions
- [CollisionShape.cpp](../src/CollisionShape.cpp) - Collision detection implementation
- [ParticleEmitter.h](../include/ParticleEmitter.h) - Particle emitter with collision support
- [CollisionDemo.cpp](../src/CollisionDemo.cpp) - Visual demonstration examples
