#include "Box2DBackend.h"
#include "Box2DRigidBody.h"
#include <iostream>

#ifdef USE_BOX2D

Box2DBackend::Box2DBackend()
    : m_World(nullptr)
    , m_VelocityIterations(6)
    , m_PositionIterations(2)
    , m_Initialized(false)
    , m_DebugDrawEnabled(false)
{
}

Box2DBackend::~Box2DBackend() {
    if (m_Initialized) {
        Shutdown();
    }
}

void Box2DBackend::Initialize(const Vec3& gravity) {
    if (m_Initialized) return;

    b2Vec2 gravity2D = ToBox2D(gravity);
    m_World = new b2World(gravity2D);

    m_Initialized = true;
    std::cout << "Box2D Backend initialized with gravity (" << gravity2D.x << ", " << gravity2D.y << ")" << std::endl;
}

void Box2DBackend::Shutdown() {
    if (m_World) {
        delete m_World;
        m_World = nullptr;
    }
    m_Initialized = false;
    std::cout << "Box2D Backend shutdown." << std::endl;
}

void Box2DBackend::Update(float deltaTime, int subSteps) {
    if (!m_Initialized || !m_World) return;

    // Box2D doesn't support sub-stepping in the same way as Bullet/PhysX via param, 
    // but we can loop here if needed.
    // Usually standard step is enough.
    // subSteps is ignored for now, or we can divide deltaTime.
    
    m_World->Step(deltaTime, m_VelocityIterations, m_PositionIterations);
}

void Box2DBackend::SetGravity(const Vec3& gravity) {
    if (m_World) {
        m_World->SetGravity(ToBox2D(gravity));
    }
}

Vec3 Box2DBackend::GetGravity() const {
    if (m_World) {
        return ToVec3(m_World->GetGravity());
    }
    return Vec3(0, 0, 0);
}

// Raycast Callback
class RycastCallback : public b2RayCastCallback {
public:
    b2Fixture* m_fixture = nullptr;
    b2Vec2 m_point;
    b2Vec2 m_normal;
    float m_fraction = 1.0f;
    uint32_t m_filter = ~0u;

    float ReportFixture(b2Fixture* fixture, const b2Vec2& point, const b2Vec2& normal, float fraction) override {
        // Filter check can go here if we used collision filtering
        // For now, accept closest
        m_fixture = fixture;
        m_point = point;
        m_normal = normal;
        m_fraction = fraction;
        return fraction; // Returns fraction to clip the ray to this hit
    }
};

bool Box2DBackend::Raycast(const Vec3& from, const Vec3& to, RaycastHit& hit, uint32_t filter) {
    if (!m_World) return false;

    RycastCallback callback;
    callback.m_filter = filter;
    
    b2Vec2 p1 = ToBox2D(from);
    b2Vec2 p2 = ToBox2D(to);

    m_World->RayCast(&callback, p1, p2);

    if (callback.m_fixture) {
        hit.point = Vec3(callback.m_point.x, callback.m_point.y, from.z); // Keep Z from ray start? Or 0?
        hit.normal = Vec3(callback.m_normal.x, callback.m_normal.y, 0.0f);
        hit.distance = (hit.point - from).Length();
        // hit.userData = callback.m_fixture->GetBody()->GetUserData().pointer; 
        // Box2D 2.4.1 uses uintptr_t (GetUserData().pointer)
        // We need to verify if user data is set.
        hit.userData = reinterpret_cast<void*>(callback.m_fixture->GetBody()->GetUserData().pointer);
        return true;
    }
    
    return false;
}

int Box2DBackend::GetNumRigidBodies() const {
    if (m_World) {
        return m_World->GetBodyCount();
    }
    return 0;
}

void Box2DBackend::SetDebugDrawEnabled(bool enabled) {
    m_DebugDrawEnabled = enabled;
}

bool Box2DBackend::IsDebugDrawEnabled() const {
    return m_DebugDrawEnabled;
}

void* Box2DBackend::GetNativeWorld() {
    return m_World;
}

void Box2DBackend::ApplyImpulse(void* userData, const Vec3& impulse, const Vec3& point) {
    if (!userData) return;
    b2Body* body = static_cast<b2Body*>(userData);
    body->ApplyLinearImpulse(ToBox2D(impulse), ToBox2D(point), true);
}

void Box2DBackend::GetActiveRigidBodies(std::vector<IPhysicsRigidBody*>& outBodies) {
    outBodies.clear();
    if (!m_World) return;

    for (b2Body* body = m_World->GetBodyList(); body; body = body->GetNext()) {
        if (body->GetType() == b2_dynamicBody && body->IsAwake()) {
            Box2DRigidBody* rb = reinterpret_cast<Box2DRigidBody*>(body->GetUserData().pointer);
            if (rb) {
                outBodies.push_back(rb);
            }
        }
    }
}

// Overlap queries - simplified implementation for Box2D
// Box2D AABB query
class OverlapCallback : public b2QueryCallback {
public:
    std::vector<void*>& m_Results;
    OverlapCallback(std::vector<void*>& results) : m_Results(results) {}

    bool ReportFixture(b2Fixture* fixture) override {
        void* bodyPtr = reinterpret_cast<void*>(fixture->GetBody()->GetUserData().pointer);
        if (bodyPtr) {
            // Avoid duplicates
            bool found = false;
            for(auto ptr : m_Results) if(ptr == bodyPtr) found = true;
            if(!found) m_Results.push_back(bodyPtr);
        }
        return true;
    }
};

int Box2DBackend::OverlapSphere(const Vec3& center, float radius, std::vector<void*>& results, uint32_t filter) {
    if (!m_World) return 0;
    
    b2AABB aabb;
    b2Vec2 c = ToBox2D(center);
    aabb.lowerBound = c - b2Vec2(radius, radius);
    aabb.upperBound = c + b2Vec2(radius, radius);
    
    OverlapCallback callback(results);
    m_World->QueryAABB(&callback, aabb);
    
    return (int)results.size();
}

int Box2DBackend::OverlapBox(const Vec3& center, const Vec3& halfExtents, const Quat& rotation, std::vector<void*>& results, uint32_t filter) {
    // Basic AABB approximation ignoring rotation for this backend implementation
    if (!m_World) return 0;
    
    b2AABB aabb;
    b2Vec2 c = ToBox2D(center);
    b2Vec2 h = ToBox2D(halfExtents);
    aabb.lowerBound = c - h;
    aabb.upperBound = c + h;
    
    OverlapCallback callback(results);
    m_World->QueryAABB(&callback, aabb);
    
    return (int)results.size();
}

int Box2DBackend::OverlapCapsule(const Vec3& center, float radius, float halfHeight, const Quat& rotation, std::vector<void*>& results, uint32_t filter) {
    // Approximate with box
    if (!m_World) return 0;
    
    b2AABB aabb;
    b2Vec2 c = ToBox2D(center);
    // Capsule height is usually Y or X axis. AABB covering max extent.
    float extent = halfHeight + radius;
    aabb.lowerBound = c - b2Vec2(extent, extent);
    aabb.upperBound = c + b2Vec2(extent, extent);
     
    OverlapCallback callback(results);
    m_World->QueryAABB(&callback, aabb);
    
    return (int)results.size();
}

#endif // USE_BOX2D
