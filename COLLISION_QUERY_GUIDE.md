# Character Controller & Collision Query Utilities - Complete Guide

## Overview

This guide covers the enhanced **Character Controller** and comprehensive **Collision Query Utilities** for advanced physics interactions in your game engine.

**Key Components:**
- `KinematicController` - Enhanced character controller with movement queries
- `CollisionQueryUtilities` - Advanced physics queries for raycasts, sweeps, overlaps, and more

---

## Character Controller (KinematicController)

### Basic Setup

```cpp
#include "KinematicController.h"

// Create a player character
auto player = std::make_shared<GameObject>("Player");
player->GetTransform().SetPosition(Vec3(0, 2, 0));

// Create capsule collision shape (0.3m radius, 1.8m height)
auto capsuleShape = PhysicsCollisionShape::CreateCapsule(0.3f, 1.8f);

// Create and initialize controller
auto controller = std::make_shared<KinematicController>();
controller->Initialize(capsuleShape, 80.0f, 0.35f); // 80kg, 0.35m step height

// Attach to game object
player->SetKinematicController(controller);
renderer->AddGameObject(player);
```

### Movement Control

```cpp
// In your input/update handler:
void UpdatePlayerMovement(float deltaTime, std::shared_ptr<KinematicController> controller) {
    // Gather input
    float moveX = 0, moveZ = 0;
    if (inputMgr->IsKeyPressed(GLFW_KEY_W)) moveZ -= 1;
    if (inputMgr->IsKeyPressed(GLFW_KEY_S)) moveZ += 1;
    if (inputMgr->IsKeyPressed(GLFW_KEY_A)) moveX -= 1;
    if (inputMgr->IsKeyPressed(GLFW_KEY_D)) moveX += 1;

    // Set walk direction (move at 5 m/s)
    Vec3 moveDir(moveX, 0, moveZ);
    if (moveDir.Length() > 0.001f) {
        moveDir.Normalize();
    }
    controller->SetWalkDirection(moveDir * 5.0f);

    // Handle jumping
    if (inputMgr->IsKeyPressed(GLFW_KEY_SPACE) && controller->IsGrounded()) {
        controller->Jump(Vec3(0, 10.0f, 0)); // Jump upward with 10 m/s
    }

    // Update controller each frame
    controller->Update(deltaTime);
}
```

### Advanced Movement Queries

#### Check if character can jump

```cpp
// Simple approach
if (controller->CanJump()) {
    // Safe to jump
    controller->Jump(Vec3(0, jumpForce, 0));
}

// Or directly check grounded state
if (controller->IsGrounded()) {
    // On solid ground
}
```

#### Detect walls ahead

```cpp
// Check if walking into a wall
Vec3 wallNormal;
if (controller->IsWallAhead(1.0f, &wallNormal)) {
    // Player is about to hit a wall
    // wallNormal points away from the wall
    std::cout << "Wall ahead! Normal: " << wallNormal << std::endl;
    
    // Prevent walking into wall
    controller->SetWalkDirection(Vec3(0, 0, 0));
}
```

#### Slope detection

```cpp
// Check if on a slope
float slopeAngle = 0;
if (controller->IsOnSlope(&slopeAngle)) {
    float angleDegrees = slopeAngle * 180.0f / 3.14159f;
    std::cout << "On slope: " << angleDegrees << " degrees" << std::endl;
    
    // Can reduce jump force on slopes
    if (angleDegrees > 30.0f) {
        // Steep slope - harder to jump
        controller->Jump(Vec3(0, 5.0f, 0));
    }
}
```

#### Get ground normal

```cpp
// Get the surface normal below the player
Vec3 groundNormal;
if (controller->GetGroundNormal(groundNormal, 0.5f)) {
    // groundNormal points away from the ground surface
    
    // Apply effects based on ground type
    if (groundNormal.y > 0.95f) {
        // Mostly flat ground
        std::cout << "Flat surface" << std::endl;
    } else if (groundNormal.y > 0.7f) {
        // Sloped ground
        std::cout << "On slope" << std::endl;
    }
}
```

#### Predict collisions

```cpp
// Check if a movement will collide
Vec3 moveDirection = Vec3(1, 0, 0) * 5.0f; // Try to move forward 5 units
Vec3 blockingDirection;
if (controller->WillMoveCollide(moveDirection, &blockingDirection)) {
    // Movement would be blocked
    // blockingDirection is normal away from obstacle
    
    // Try sliding along the obstacle
    Vec3 slideDir = moveDirection - blockingDirection * moveDirection.Dot(blockingDirection);
    controller->SetWalkDirection(slideDir);
}
```

#### Distance to collision

```cpp
// How far can we move forward before hitting something?
Vec3 forwardDir(0, 0, -1);
float distToWall = controller->GetDistanceToCollision(forwardDir);

if (distToWall < 2.0f) {
    std::cout << "Wall is " << distToWall << "m ahead" << std::endl;
}
```

#### Velocity queries

```cpp
// Get current velocity (including falling)
Vec3 velocity = controller->GetVelocity();
float horizontalSpeed = controller->GetMoveSpeed();
bool inAir = controller->IsInAir();

if (inAir) {
    std::cout << "Character is jumping/falling" << std::endl;
    std::cout << "Vertical velocity: " << controller->GetVerticalVelocity() << " m/s" << std::endl;
}
```

---

## Collision Query Utilities

### Raycast Queries

#### Single raycast

```cpp
#include "CollisionQueryUtilities.h"

// Simple single raycast
ExtendedRaycastHit hit;
Vec3 from(0, 2, 0);
Vec3 to(10, 2, 0);

if (CollisionQueryUtilities::Raycast(from, to, hit)) {
    std::cout << "Hit at: " << hit.point << std::endl;
    std::cout << "Surface normal: " << hit.normal << std::endl;
    std::cout << "Distance: " << hit.distance << std::endl;
    
    if (hit.hasRigidBody && hit.rigidBody) {
        std::cout << "Hit object mass: " << hit.rigidBody->GetMass() << std::endl;
    }
}
```

#### Raycast in direction

```cpp
// Cast ray in a specific direction with max distance
Vec3 origin(0, 1, 0);
Vec3 direction(0, 0, -1); // Forward
float maxDistance = 100.0f;

ExtendedRaycastHit hit;
if (CollisionQueryUtilities::RaycastDirection(origin, direction, maxDistance, hit)) {
    std::cout << "Found object at distance: " << hit.distance << std::endl;
}
```

#### Multi-hit raycast

```cpp
// Get all objects hit along a ray (sorted by distance)
Vec3 from(0, 5, 0);
Vec3 to(0, -5, 0);
int maxHits = 5; // Limit to 5 hits, 0 = unlimited

auto allHits = CollisionQueryUtilities::RaycastAll(from, to, maxHits);
std::cout << "Hit " << allHits.size() << " objects" << std::endl;

for (const auto& hit : allHits) {
    std::cout << "- Distance: " << hit.distance << std::endl;
}
```

### Sweep Tests (Collision-aware movement)

#### Sweep sphere

```cpp
// Check if sphere can move from A to B without collision
Vec3 startPos(0, 1, 0);
Vec3 endPos(10, 1, 0);
float sphereRadius = 0.5f;

SweepTestResult sweep = CollisionQueryUtilities::SweepSphere(startPos, endPos, sphereRadius);

if (sweep.hasHit) {
    std::cout << "Hit at distance: " << sweep.distance << std::endl;
    std::cout << "Hit point: " << sweep.hitPoint << std::endl;
    std::cout << "Fraction of movement completed: " << sweep.fraction << std::endl;
    
    // Move character up to the point of collision
    Vec3 safePos = startPos + (endPos - startPos) * sweep.fraction;
} else {
    std::cout << "Path is clear!" << std::endl;
}
```

#### Sweep capsule (for character movement)

```cpp
// Sweep capsule shape (useful for character controllers)
Vec3 currentPos = controller->GetPosition();
Vec3 desiredPos = currentPos + Vec3(5, 0, 0);
float capsuleRadius = 0.3f;
float capsuleHeight = 1.8f;

SweepTestResult sweep = CollisionQueryUtilities::SweepCapsule(
    currentPos, desiredPos, capsuleRadius, capsuleHeight
);

if (sweep.hasHit) {
    // Path is blocked
    // Move character to safe position before collision
    Vec3 safePos = currentPos + (desiredPos - currentPos) * (sweep.fraction - 0.01f);
    controller->SetPosition(safePos);
} else {
    // Clear to move
    controller->SetPosition(desiredPos);
}
```

#### Sweep box

```cpp
// Sweep axis-aligned box
Vec3 from(0, 0, 0);
Vec3 to(20, 0, 0);
Vec3 boxHalfExtents(1, 1, 1);

SweepTestResult sweep = CollisionQueryUtilities::SweepBox(from, to, boxHalfExtents);

if (sweep.hasHit) {
    std::cout << "Collision at " << (from + (to - from) * sweep.fraction) << std::endl;
}
```

### Overlap Tests (Find objects in region)

#### Overlap sphere

```cpp
// Find all objects touching a sphere
Vec3 center(0, 2, 0);
float radius = 5.0f;

OverlapTestResult overlap = CollisionQueryUtilities::OverlapSphere(center, radius);

std::cout << "Found " << overlap.count << " objects" << std::endl;
for (const auto& body : overlap.overlappingBodies) {
    std::cout << "- Object at: " << body->GetPosition() << std::endl;
    std::cout << "  Mass: " << body->GetMass() << std::endl;
}
```

#### Overlap box

```cpp
// Find all objects in an AABB region
Vec3 center(0, 2, 0);
Vec3 halfExtents(10, 5, 10);

OverlapTestResult overlap = CollisionQueryUtilities::OverlapBox(center, halfExtents);

// Useful for area-of-effect abilities
for (const auto& body : overlap.overlappingBodies) {
    // Apply damage or effect to each body
}
```

#### Overlap capsule

```cpp
// Useful for detecting enemies near player
Vec3 playerPos = player->GetTransform().GetPosition();
float checkRadius = 2.0f;
float checkHeight = 2.0f;

OverlapTestResult nearby = CollisionQueryUtilities::OverlapCapsule(
    playerPos, checkRadius, checkHeight
);

// Find enemies to attack
for (const auto& body : nearby.overlappingBodies) {
    // Check if this is an enemy
    // Apply attack effect
}
```

### Distance Queries

#### Sphere distance

```cpp
// Get distance between two objects
Vec3 pos1(0, 0, 0);
float radius1 = 1.0f;
Vec3 pos2(5, 0, 0);
float radius2 = 1.0f;

float distance = CollisionQueryUtilities::SphereDistance(pos1, radius1, pos2, radius2);

if (distance <= 0) {
    std::cout << "Spheres are overlapping!" << std::endl;
} else {
    std::cout << "Distance between surfaces: " << distance << std::endl;
}
```

#### Get bodies in radius

```cpp
// Find all rigid bodies within a distance
Vec3 center(0, 0, 0);
float maxDistance = 10.0f;

auto bodies = CollisionQueryUtilities::GetBodiesInRadius(center, maxDistance);

std::cout << "Found " << bodies.size() << " bodies within " << maxDistance << "m" << std::endl;

// Sort by distance for efficiency
std::sort(bodies.begin(), bodies.end(), [center](const auto& a, const auto& b) {
    return (a->GetPosition() - center).Length() < 
           (b->GetPosition() - center).Length();
});
```

### Character-Specific Queries

#### Ground detection for characters

```cpp
// Check if ground exists below character
float groundDistance = 0.0f;
if (CollisionQueryUtilities::IsGroundDetected(controller, 0.5f, &groundDistance)) {
    std::cout << "Ground is " << groundDistance << "m below" << std::endl;
    
    // Apply different effects based on distance
    if (groundDistance < 0.1f) {
        // Touching ground
    } else if (groundDistance < 1.0f) {
        // Close to ground but in air
    }
}
```

#### Check if movement is possible

```cpp
// Predict if a movement direction will collide
Vec3 moveDir = Vec3(1, 0, 0) * 5.0f;
float maxStepHeight = 0.5f;

if (CollisionQueryUtilities::CanMove(controller, moveDir, maxStepHeight)) {
    std::cout << "Movement is safe" << std::endl;
    controller->SetWalkDirection(moveDir);
} else {
    std::cout << "Blocked by obstacle" << std::endl;
    // Try alternative path
}
```

#### Find valid position (for unstucking)

```cpp
// Try to move character to target position
// If blocked, find nearest valid position
Vec3 targetPos(10, 1, 0);
float searchRadius = 5.0f;
Vec3 validPos;

if (CollisionQueryUtilities::FindValidPosition(controller, targetPos, searchRadius, validPos)) {
    std::cout << "Moving to valid position: " << validPos << std::endl;
    controller->SetPosition(validPos);
} else {
    std::cout << "No valid position found" << std::endl;
}
```

---

## Practical Examples

### Example 1: Player with Smart Movement

```cpp
class PlayerController {
    std::shared_ptr<KinematicController> m_Controller;

public:
    void Update(float deltaTime, const InputManager& input) {
        // Get input
        float moveX = 0, moveZ = 0;
        if (input.IsKeyPressed(GLFW_KEY_W)) moveZ -= 1;
        if (input.IsKeyPressed(GLFW_KEY_S)) moveZ += 1;
        if (input.IsKeyPressed(GLFW_KEY_A)) moveX -= 1;
        if (input.IsKeyPressed(GLFW_KEY_D)) moveX += 1;

        // Avoid walking into walls
        Vec3 desiredDir(moveX, 0, moveZ);
        desiredDir.Normalize();

        if (m_Controller->IsWallAhead(1.0f)) {
            // Wall ahead - try sliding along it
            Vec3 slideDir = desiredDir;
            slideDir.x *= 0.5f; // Reduce forward push
            m_Controller->SetWalkDirection(slideDir * 5.0f);
        } else {
            m_Controller->SetWalkDirection(desiredDir * 5.0f);
        }

        // Jumping
        if (input.IsKeyPressed(GLFW_KEY_SPACE) && m_Controller->CanJump()) {
            // Scale jump force based on slope
            float slopeAngle = 0;
            m_Controller->IsOnSlope(&slopeAngle);
            float slopeModifier = 1.0f - (slopeAngle / 3.14159f) * 0.5f;
            
            m_Controller->Jump(Vec3(0, 10.0f * slopeModifier, 0));
        }

        // Update physics
        m_Controller->Update(deltaTime);

        // Animation updates based on state
        if (m_Controller->IsInAir()) {
            PlayAnimation("jump");
        } else if (m_Controller->GetMoveSpeed() > 0.5f) {
            PlayAnimation("run");
        } else {
            PlayAnimation("idle");
        }
    }
};
```

### Example 2: Area-of-Effect Ability

```cpp
void CastAbility(Vec3 abilityPos, float abilityRadius) {
    // Find all enemies in radius
    OverlapTestResult result = CollisionQueryUtilities::OverlapSphere(abilityPos, abilityRadius);

    std::cout << "Ability hit " << result.count << " targets" << std::endl;

    for (const auto& body : result.overlappingBodies) {
        // Calculate distance for falloff
        Vec3 targetPos = body->GetPosition();
        float dist = (targetPos - abilityPos).Length();
        float normalizedDist = std::min(1.0f, dist / abilityRadius);

        // Falloff damage
        float damage = 100.0f * (1.0f - normalizedDist);

        // Apply knockback
        Vec3 knockbackDir = (targetPos - abilityPos).Normalize();
        body->ApplyImpulse(knockbackDir * 20.0f);

        std::cout << "- Damage: " << damage << std::endl;
    }
}
```

### Example 3: Line-of-Sight AI

```cpp
bool CanSeeTarget(Vec3 aiPos, Vec3 targetPos) {
    // Raycast from AI to target
    ExtendedRaycastHit hit;
    if (CollisionQueryUtilities::Raycast(aiPos, targetPos, hit)) {
        // Check if we hit the target
        // In real game, would compare hit.rigidBody to target's body
        float distToTarget = (targetPos - aiPos).Length();
        return hit.distance >= distToTarget * 0.95f; // Account for floating point errors
    }

    return true; // No obstructions
}
```

---

## Performance Considerations

### Query Optimization Tips

1. **Cache Results**: Don't query every frame if not needed
   ```cpp
   static float queryTimer = 0.0f;
   queryTimer += deltaTime;
   if (queryTimer >= 0.1f) { // Query 10x per second
       PerformQueries();
       queryTimer = 0.0f;
   }
   ```

2. **Use Appropriate Query Types**:
   - `Raycast` - Point-to-point line checks (fastest)
   - `SweepTest` - Collision-aware movement prediction
   - `OverlapTest` - Region queries (slowest, use sparingly)

3. **Limit Results**:
   ```cpp
   auto hits = CollisionQueryUtilities::RaycastAll(from, to, 5); // Max 5 hits
   ```

4. **Use Collision Filters** (when needed):
   ```cpp
   ExtendedRaycastHit hit;
   CollisionQueryUtilities::RaycastFiltered(from, to, groupA, maskA, hit);
   ```

### Typical Performance

- **Single Raycast**: < 0.1ms
- **RaycastAll (10 hits)**: < 1ms
- **SweepTest**: < 0.5ms
- **OverlapSphere**: < 2ms
- **OverlapBox**: < 2ms

---

## API Reference Summary

### KinematicController Queries
```cpp
bool IsGrounded() const;                              // On ground?
bool CanJump() const;                                 // Can jump?
bool IsInAir() const;                                 // Falling/jumping?
bool IsWallAhead(float dist, Vec3* normal = nullptr); // Wall ahead?
bool IsOnSlope(float* angle = nullptr) const;        // On slope?
bool GetGroundNormal(Vec3& n, float dist) const;     // Ground surface normal
bool WillMoveCollide(Vec3 dir, Vec3* block = nullptr); // Will move collide?
float GetDistanceToCollision(Vec3 dir) const;        // Distance to obstacle
Vec3 GetVelocity() const;                             // Current velocity
float GetMoveSpeed() const;                           // Horizontal speed
```

### CollisionQueryUtilities

**Raycasts:**
```cpp
bool Raycast(Vec3 from, Vec3 to, ExtendedRaycastHit& hit);
bool RaycastDirection(Vec3 origin, Vec3 dir, float dist, ExtendedRaycastHit& hit);
std::vector<ExtendedRaycastHit> RaycastAll(Vec3 from, Vec3 to, int maxHits = 0);
```

**Sweeps:**
```cpp
SweepTestResult SweepSphere(Vec3 from, Vec3 to, float radius);
SweepTestResult SweepCapsule(Vec3 from, Vec3 to, float radius, float height);
SweepTestResult SweepBox(Vec3 from, Vec3 to, Vec3 halfExtents);
```

**Overlaps:**
```cpp
OverlapTestResult OverlapSphere(Vec3 center, float radius);
OverlapTestResult OverlapBox(Vec3 center, Vec3 halfExtents);
OverlapTestResult OverlapCapsule(Vec3 center, float radius, float height);
```

**Utilities:**
```cpp
bool IsGroundDetected(controller, distance, outDistance);
bool CanMove(controller, direction, stepHeight);
bool FindValidPosition(controller, targetPos, searchRadius, outPos);
std::vector<RigidBody*> GetBodiesInRadius(Vec3 center, float maxDist);
float SphereDistance(Vec3 pos1, float r1, Vec3 pos2, float r2);
bool LineIntersect(Vec3 from, Vec3 to, Vec3& outPoint);
bool AABBOverlap(Vec3 min1, Vec3 max1, Vec3 min2, Vec3 max2);
```

---

## Next Steps

1. **Integrate** these utilities into your game logic
2. **Profile** queries to identify bottlenecks
3. **Cache** frequent queries where appropriate
4. **Experiment** with collision layer filtering for optimization
5. **Test** edge cases (corners, slopes, floating objects)

Good luck with your physics interactions! 🎮
