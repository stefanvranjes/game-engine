# Character Controller & Collision Queries - Quick Reference

## 30-Second Quickstart

### Basic Character Setup
```cpp
auto controller = std::make_shared<KinematicController>();
auto shape = PhysicsCollisionShape::CreateCapsule(0.3f, 1.8f);
controller->Initialize(shape, 80.0f, 0.35f);  // 80kg, 0.35m step height
gameObject->SetKinematicController(controller);
```

### Movement
```cpp
// Each frame:
controller->SetWalkDirection({inputX, 0, inputZ} * 5.0f);
if (canJump && controller->IsGrounded()) {
    controller->Jump({0, 10.0f, 0});
}
controller->Update(deltaTime);
```

### Common Queries
```cpp
// Check grounded
if (controller->IsGrounded()) { /* ok to jump */ }

// Check wall ahead
if (controller->IsWallAhead(1.0f)) { /* avoid walking */ }

// Get ground surface
Vec3 groundNormal;
controller->GetGroundNormal(groundNormal);

// Raycast
ExtendedRaycastHit hit;
if (CollisionQueryUtilities::Raycast(from, to, hit)) {
    std::cout << "Hit at: " << hit.point << std::endl;
}

// Sweep test
auto sweep = CollisionQueryUtilities::SweepSphere(start, end, radius);

// Find objects in area
auto overlap = CollisionQueryUtilities::OverlapSphere(center, radius);
```

---

## KinematicController Cheat Sheet

### Movement Control
| Method | Purpose |
|--------|---------|
| `SetWalkDirection(Vec3)` | Set movement direction & speed |
| `Jump(Vec3)` | Jump with impulse |
| `SetMaxWalkSpeed(float)` | Max walk speed (m/s) |
| `SetFallSpeed(float)` | Terminal velocity |
| `SetPosition(Vec3)` | Teleport character |
| `GetPosition()` → Vec3 | Current position |

### State Queries
| Method | Returns |
|--------|---------|
| `IsGrounded()` | bool - On solid ground |
| `IsInAir()` | bool - Falling or jumping |
| `CanJump()` | bool - Can jump (same as IsGrounded) |
| `GetVerticalVelocity()` | float - Vertical speed |
| `GetWalkDirection()` | Vec3 - Current walk direction |
| `GetMoveSpeed()` | float - Horizontal speed |

### Advanced Queries (New!)
| Method | Returns | Purpose |
|--------|---------|---------|
| `IsWallAhead(float, Vec3*)` | bool | Detect walls ahead |
| `IsOnSlope(float*)` | bool | Check if on slope |
| `GetGroundNormal(Vec3&, float)` | bool | Get ground surface normal |
| `WillMoveCollide(Vec3, Vec3*)` | bool | Predict collision |
| `GetDistanceToCollision(Vec3)` | float | Distance to obstacle |
| `GetVelocity()` | Vec3 | Total velocity |

### Configuration
| Method | Purpose |
|--------|---------|
| `SetStepHeight(float)` | Stair climbing height |
| `SetActive(bool)` | Enable/disable |
| `IsActive()` | Check if active |

---

## CollisionQueryUtilities Cheat Sheet

### Raycasts
```cpp
// Single hit (closest)
ExtendedRaycastHit hit;
CollisionQueryUtilities::Raycast(from, to, hit);

// All hits along ray
auto hits = CollisionQueryUtilities::RaycastAll(from, to, maxHits);

// In specific direction
ExtendedRaycastHit hit;
CollisionQueryUtilities::RaycastDirection(origin, direction, maxDist, hit);

// With filter
ExtendedRaycastHit hit;
CollisionQueryUtilities::RaycastFiltered(from, to, group, mask, hit);
```

### Sweep Tests (Movement Detection)
```cpp
// Sphere movement
auto result = CollisionQueryUtilities::SweepSphere(start, end, radius);

// Capsule movement (for characters)
auto result = CollisionQueryUtilities::SweepCapsule(start, end, radius, height);

// Box movement
auto result = CollisionQueryUtilities::SweepBox(start, end, halfExtents);

// Check result
if (result.hasHit) {
    Vec3 hitPoint = result.hitPoint;
    float dist = result.distance;
    float frac = result.fraction;  // 0-1 of movement completed
}
```

### Overlap Tests (Find Objects in Region)
```cpp
// Sphere overlap
auto result = CollisionQueryUtilities::OverlapSphere(center, radius);

// Box overlap
auto result = CollisionQueryUtilities::OverlapBox(center, halfExtents);

// Capsule overlap
auto result = CollisionQueryUtilities::OverlapCapsule(center, radius, height);

// Access results
for (const auto& body : result.overlappingBodies) {
    Vec3 pos = body->GetPosition();
    float mass = body->GetMass();
}
```

### Distance Queries
```cpp
// Distance between two spheres
float dist = CollisionQueryUtilities::SphereDistance(pos1, r1, pos2, r2);

// All bodies in radius
auto bodies = CollisionQueryUtilities::GetBodiesInRadius(center, maxDist);

// Line intersection
Vec3 intersectPoint;
if (CollisionQueryUtilities::LineIntersect(from, to, intersectPoint)) {
    // Found intersection
}

// AABB overlap
bool overlaps = CollisionQueryUtilities::AABBOverlap(min1, max1, min2, max2);
```

### Character Helpers
```cpp
// Ground detection
float groundDist;
if (CollisionQueryUtilities::IsGroundDetected(controller, checkDist, &groundDist)) {
    // Ground found at groundDist
}

// Can character move?
if (CollisionQueryUtilities::CanMove(controller, moveDir, maxStepHeight)) {
    // Safe to move
}

// Find valid position
Vec3 validPos;
if (CollisionQueryUtilities::FindValidPosition(controller, targetPos, searchRadius, validPos)) {
    controller->SetPosition(validPos);
}
```

---

## Hit Information Structure

### ExtendedRaycastHit
```cpp
struct ExtendedRaycastHit {
    Vec3 point;                         // World space impact point
    Vec3 normal;                        // Surface normal at impact
    float distance;                     // Distance from ray origin
    btRigidBody* body;                  // Raw Bullet3D pointer
    std::shared_ptr<RigidBody> rigidBody;  // Game engine wrapper
    bool hasRigidBody;                  // True if valid RigidBody
};
```

### SweepTestResult
```cpp
struct SweepTestResult {
    bool hasHit;                        // Was there a collision?
    Vec3 hitPoint;                      // Where collision occurred
    Vec3 hitNormal;                     // Surface normal
    float distance;                     // Distance to collision
    float fraction;                     // Percent of movement (0-1)
    std::shared_ptr<RigidBody> hitBody; // The object hit
};
```

### OverlapTestResult
```cpp
struct OverlapTestResult {
    std::vector<std::shared_ptr<RigidBody>> overlappingBodies;
    int count;  // Number of bodies
};
```

---

## Common Patterns

### Pattern 1: Smart Player Movement
```cpp
void UpdatePlayer(float dt, const Input& input) {
    Vec3 moveDir = GetInputDirection();  // From WASD
    
    // Avoid walls
    if (controller->IsWallAhead(1.0f)) {
        moveDir *= 0.5f;
    }
    
    controller->SetWalkDirection(moveDir * 5.0f);
    
    // Jump
    if (input.jump && controller->CanJump()) {
        controller->Jump({0, 10.0f, 0});
    }
    
    controller->Update(dt);
}
```

### Pattern 2: AOE Ability
```cpp
void CastAOE(Vec3 pos, float radius, float damage) {
    auto overlap = CollisionQueryUtilities::OverlapSphere(pos, radius);
    
    for (const auto& body : overlap.overlappingBodies) {
        float dist = (body->GetPosition() - pos).Length();
        float falloff = 1.0f - (dist / radius);
        ApplyDamage(body, damage * falloff);
    }
}
```

### Pattern 3: Line of Sight
```cpp
bool CanSeeTarget(Vec3 from, Vec3 to) {
    ExtendedRaycastHit hit;
    if (CollisionQueryUtilities::Raycast(from, to, hit)) {
        return hit.distance >= (to - from).Length() * 0.95f;
    }
    return true;
}
```

### Pattern 4: Predict Movement
```cpp
bool CanMoveTo(Vec3 from, Vec3 to) {
    auto sweep = CollisionQueryUtilities::SweepCapsule(from, to, 0.3f, 1.8f);
    return !sweep.hasHit || sweep.fraction > 0.95f;
}
```

### Pattern 5: Find Nearest Object
```cpp
std::shared_ptr<RigidBody> FindNearestBody(Vec3 pos, float maxRadius) {
    auto bodies = CollisionQueryUtilities::GetBodiesInRadius(pos, maxRadius);
    
    std::shared_ptr<RigidBody> nearest = nullptr;
    float minDist = maxRadius;
    
    for (const auto& body : bodies) {
        float dist = (body->GetPosition() - pos).Length();
        if (dist < minDist) {
            minDist = dist;
            nearest = body;
        }
    }
    
    return nearest;
}
```

---

## Performance Tips

✓ Cache query results (don't query every frame)
✓ Use RaycastAll with maxHits limit
✓ Prefer Raycast over Sweep for simple checks
✓ Use appropriate collision filters
✓ Limit OverlapTest usage (slower)
✓ Query only when needed

---

## Common Issues & Solutions

### Issue: Character stuck in corner
**Solution**: Use `FindValidPosition()` to unstuck
```cpp
Vec3 validPos;
if (CollisionQueryUtilities::FindValidPosition(controller, targetPos, 2.0f, validPos)) {
    controller->SetPosition(validPos);
}
```

### Issue: Raycast missing small objects
**Solution**: Expand query size or use sweep test
```cpp
// Instead of raycast
auto sweep = CollisionQueryUtilities::SweepSphere(from, to, 0.5f);
```

### Issue: Character sliding down slopes
**Solution**: Check slope angle and adjust gravity
```cpp
float angle;
if (controller->IsOnSlope(&angle) && angle > 0.5f) {
    // Steep slope - apply grip or slow movement
    controller->SetWalkDirection(controller->GetWalkDirection() * 0.5f);
}
```

### Issue: Wall clipping
**Solution**: Use sweep test for movement validation
```cpp
auto sweep = CollisionQueryUtilities::SweepCapsule(currentPos, desiredPos, 0.3f, 1.8f);
if (sweep.hasHit && sweep.fraction < 0.95f) {
    // Stop before wall
    controller->SetPosition(currentPos + (desiredPos - currentPos) * sweep.fraction * 0.9f);
}
```

---

## API at a Glance

```cpp
// KinematicController (movement + queries)
void Update(float dt);
void SetWalkDirection(Vec3);
void Jump(Vec3);
bool IsGrounded();
bool IsWallAhead(float, Vec3*);
bool IsOnSlope(float*);
bool GetGroundNormal(Vec3&, float);
bool WillMoveCollide(Vec3, Vec3*);
float GetDistanceToCollision(Vec3);
Vec3 GetVelocity();
float GetMoveSpeed();

// CollisionQueryUtilities (advanced queries)
bool Raycast(Vec3, Vec3, ExtendedRaycastHit&);
std::vector<ExtendedRaycastHit> RaycastAll(Vec3, Vec3, int);
SweepTestResult SweepSphere(Vec3, Vec3, float);
SweepTestResult SweepCapsule(Vec3, Vec3, float, float);
SweepTestResult SweepBox(Vec3, Vec3, Vec3);
OverlapTestResult OverlapSphere(Vec3, float);
OverlapTestResult OverlapBox(Vec3, Vec3);
OverlapTestResult OverlapCapsule(Vec3, float, float);
bool IsGroundDetected(controller, float, float*);
bool CanMove(controller, Vec3, float);
bool FindValidPosition(controller, Vec3, float, Vec3&);
std::vector<RigidBody*> GetBodiesInRadius(Vec3, float);
float SphereDistance(Vec3, float, Vec3, float);
bool LineIntersect(Vec3, Vec3, Vec3&);
bool AABBOverlap(Vec3, Vec3, Vec3, Vec3);
```

See [COLLISION_QUERY_GUIDE.md](COLLISION_QUERY_GUIDE.md) for detailed examples!
