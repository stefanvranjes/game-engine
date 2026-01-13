#include "Box2DBackend.h"
#include "Box2DRigidBody.h"
#include "Box2DJoint.h"
#include "Box2DCharacterController.h"
#include "Gizmo.h"
#include "Renderer.h"
#include <iostream>
#include <GL/gl.h> // Basic GL for debug draw lines

#ifdef USE_BOX2D

Box2DBackend::Box2DBackend()
    : m_World(nullptr)
    , m_VelocityIterations(6)
    , m_PositionIterations(2)
    , m_Initialized(false)
    , m_DebugDrawEnabled(false)
    , m_DebugDraw(nullptr)
    , m_ContactListener(nullptr)
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

    // Setup Contact Listener
    m_ContactListener = new Box2DContactListener();
    m_World->SetContactListener(m_ContactListener);

    // Setup Debug Draw
    m_DebugDraw = new Box2DDebugDraw();
    m_DebugDraw->SetFlags(b2Draw::e_shapeBit | b2Draw::e_jointBit | b2Draw::e_centerOfMassBit);
    m_World->SetDebugDraw(m_DebugDraw);

    m_Initialized = true;
    std::cout << "Box2D Backend initialized with gravity (" << gravity2D.x << ", " << gravity2D.y << ")" << std::endl;
}

void Box2DBackend::Shutdown() {
    for (auto joint : m_Joints) {
        if (m_World) {
            b2Joint* native = static_cast<Box2DJoint*>(joint)->GetNativeJoint();
            if (native) m_World->DestroyJoint(native);
        }
        delete joint;
    }
    m_Joints.clear();

    for (auto controller : m_Controllers) {
        // Body destroyed when world destroyed? 
        // Box2DCharacterController destructor destroys body.
        // But if world is deleted first, body destruction might crash if not careful?
        // Box2D World destructor destroys all bodies.
        // So we just delete the wrapper.
        // Ensure wrapper doesn't try to destroy body if world is gone?
        // Box2DCharacterController checks m_Backend->GetWorld().
        // If we set m_World to nullptr AFTER this loop, it might be okay.
        delete controller; 
    }
    m_Controllers.clear();

    if (m_World) {
        delete m_World;
        m_World = nullptr;
    }
    if (m_ContactListener) {
        delete m_ContactListener;
        m_ContactListener = nullptr;
    }
    if (m_DebugDraw) {
        delete m_DebugDraw;
        m_DebugDraw = nullptr;
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
    
    // Use a fixed time step for Box2D for stability
    const float fixedDeltaTime = 1.0f / 60.0f; // 60 Hz
    static float accumulator = 0.0f;
    accumulator += deltaTime;

    while (accumulator >= fixedDeltaTime) {
        m_World->Step(fixedDeltaTime, m_VelocityIterations, m_PositionIterations);
        
        // Update Transform Sync here if needed, or in RigidBodies
        // We sync FROM physics TO game objects usually.
        // Box2DRigidBody doesn't have an Update(). 
        // Application::Update loops game objects and asks them to sync? 
        // Yes, Application::Update -> SyncTransformFromPhysics.
        
        // Update Character Controllers
        Vec3 gravity = GetGravity();
        for (auto controller : m_Controllers) {
            controller->Update(fixedDeltaTime, gravity);
        }

        accumulator -= fixedDeltaTime;
    }
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

bool Box2DBackend::Raycast(const Vec3& from, const Vec3& to, PhysicsRaycastHit& hit, uint32_t filter) {
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

class Box2DContactListener : public b2ContactListener {
    void BeginContact(b2Contact* contact) override {
        b2Fixture* fa = contact->GetFixtureA();
        b2Fixture* fb = contact->GetFixtureB();
        
        Box2DRigidBody* ba = reinterpret_cast<Box2DRigidBody*>(fa->GetBody()->GetUserData().pointer);
        Box2DRigidBody* bb = reinterpret_cast<Box2DRigidBody*>(fb->GetBody()->GetUserData().pointer);
        
        if (ba) {
            IPhysicsRigidBody::CollisionInfo info;
            info.point = Box2DBackend::ToVec3(contact->GetManifold()->localPoint); // Approximation
            info.normal = Box2DBackend::ToVec3(contact->GetManifold()->localNormal); 
            info.otherBody = bb;
            ba->OnCollision(info); // Assuming accessor or friend access to internal OnCollision if protected, or public
            // However, IPhysicsRigidBody declares SetOnCollisionCallback. The implementation should call the callback.
            // But INative rigid bodies usually have a Dispatch method or we invoke the callback directly if we had access.
            // Since we can't access m_CollisionCallback of interface easily without friend, let's assume we cast to Box2DRigidBody 
            // and Box2DRigidBody has a method to trigger it.
        }
        if (bb) {
            IPhysicsRigidBody::CollisionInfo info;
            info.otherBody = ba;
             // Inverse normal? Box2D manifold is one way.
            bb->OnCollision(info);
        }
    }
    void EndContact(b2Contact* contact) override {}
};

class Box2DDebugDraw : public b2Draw {
public:
    Renderer* m_Renderer = nullptr; 

    // Simple immediate mode line drawer using Legacy GL for debug or construct vertices
    // Since we don't have a dynamic line flush system exposed easily, we use direct GL calls for debug.
    // This assumes an OpenGL context is active (which it is during Render).
    // Note: If using strict Core Profile this might fail, but Gizmo.cpp implies VAO usage.
    // Let's try to use a minimal VAO approach or just lines if compatibility profile is used.
    // Actually engine uses 4.x Core. glBegin/End won't work.
    // We MUST use a helper. 
    // Let's implement a quick VBO streaming or just individual draw calls.
    // For simplicity of this task, I'll assume we can use a very inefficient but working DrawLine helper I'll add locally here.

    void DrawPolygon(const b2Vec2* vertices, int32 vertexCount, const b2Color& color) override {
        if (!m_Renderer || !m_Renderer->GetGizmoShader()) return;
        Vec3 c(color.r, color.g, color.b);
        for (int32 i = 0; i < vertexCount; ++i) {
             Vec3 p1 = m_Backend->ToVec3(vertices[i]);
             Vec3 p2 = m_Backend->ToVec3(vertices[(i + 1) % vertexCount]);
             Gizmo::DrawLine(m_Renderer->GetGizmoShader(), p1, p2, c);
        }
    }
    void DrawSolidPolygon(const b2Vec2* vertices, int32 vertexCount, const b2Color& color) override {
        DrawPolygon(vertices, vertexCount, color);
    }
    void DrawCircle(const b2Vec2& center, float radius, const b2Color& color) override {
        if (!m_Renderer || !m_Renderer->GetGizmoShader()) return;
        Vec3 c(color.r, color.g, color.b);
        Vec3 cnt = m_Backend->ToVec3(center);
        const int segments = 16;
        for(int i=0; i<segments; ++i) {
            float t1 = (float)i / segments * 6.28318f;
            float t2 = (float)(i+1) / segments * 6.28318f;
            Vec3 p1 = cnt + Vec3(cos(t1)*radius, sin(t1)*radius, 0);
            Vec3 p2 = cnt + Vec3(cos(t2)*radius, sin(t2)*radius, 0);
            Gizmo::DrawLine(m_Renderer->GetGizmoShader(), p1, p2, c);
        }
    }
    void DrawSolidCircle(const b2Vec2& center, float radius, const b2Vec2& axis, const b2Color& color) override {
        DrawCircle(center, radius, color);
    }
    void DrawSegment(const b2Vec2& p1, const b2Vec2& p2, const b2Color& color) override {
        if (!m_Renderer || !m_Renderer->GetGizmoShader()) return;
        Vec3 c(color.r, color.g, color.b);
        Gizmo::DrawLine(m_Renderer->GetGizmoShader(), m_Backend->ToVec3(p1), m_Backend->ToVec3(p2), c);
    }
    void DrawTransform(const b2Transform& xf) override {
        if (!m_Renderer || !m_Renderer->GetGizmoShader()) return;
        float axisScale = 0.5f;
        Vec3 p = m_Backend->ToVec3(xf.p);
        Vec3 px = p + m_Backend->ToVec3(xf.q.GetXAxis()) * axisScale;
        Vec3 py = p + m_Backend->ToVec3(xf.q.GetYAxis()) * axisScale;
        
        Gizmo::DrawLine(m_Renderer->GetGizmoShader(), p, px, Vec3(1,0,0));
        Gizmo::DrawLine(m_Renderer->GetGizmoShader(), p, py, Vec3(0,1,0));
    }
    void DrawPoint(const b2Vec2& p, float size, const b2Color& color) override {}
};

void Box2DBackend::DebugDraw(Renderer* renderer) {
    if (!m_DebugDraw) return;
    static_cast<Box2DDebugDraw*>(m_DebugDraw)->m_Renderer = renderer;
    m_World->DebugDraw();
}

IPhysicsJoint* Box2DBackend::CreateJoint(const JointDef& def) {
    if (!m_World) return nullptr;

    b2Body* bodyA = static_cast<b2Body*>(def.bodyA->GetNativeBody());
    b2Body* bodyB = static_cast<b2Body*>(def.bodyB->GetNativeBody());

    if (!bodyA || !bodyB) {
        std::cerr << "CreateJoint: Invalid bodies" << std::endl;
        return nullptr;
    }

    Box2DJoint* wrapper = nullptr;
    b2Joint* joint = nullptr;

    if (def.type == JointType::Revolute) {
        b2RevoluteJointDef jDef;
        // Box2D anchors are local?? No, Initialize uses world.
        jDef.Initialize(bodyA, bodyB, ToBox2D(def.anchorA)); // AnchorA is usually the common anchor in world space
        jDef.collideConnected = def.collideConnected;
        jDef.enableLimit = def.enableLimit;
        jDef.lowerAngle = def.lowerAngle;
        jDef.upperAngle = def.upperAngle;
        jDef.enableMotor = def.enableMotor;
        jDef.motorSpeed = def.motorSpeed;
        jDef.maxMotorTorque = def.maxMotorTorque;

        joint = m_World->CreateJoint(&jDef);
        if (joint) {
            wrapper = new Box2DRevoluteJoint(this, static_cast<b2RevoluteJoint*>(joint));
        }
    }
    else if (def.type == JointType::Distance) {
        b2DistanceJointDef jDef;
        jDef.Initialize(bodyA, bodyB, ToBox2D(def.anchorA), ToBox2D(def.anchorB));
        jDef.collideConnected = def.collideConnected;
        // jDef.length is set by Initialize to current distance
        if (def.length > 0) jDef.length = def.length; 
        
        // Box2D 2.4.1 params
        // jDef.frequencyHz
        // jDef.dampingRatio
        if (def.stiffness > 0) {
              b2LinearStiffness(jDef.stiffness, jDef.damping, jDef.length, bodyA, bodyB);
              // Or custom if older Box2D
        }

        joint = m_World->CreateJoint(&jDef);
        if (joint) {
            wrapper = new Box2DJoint(this, joint, JointType::Distance);
        }
    }
    
    if (wrapper) {
        m_Joints.push_back(wrapper);
    }
    return wrapper;
}

void Box2DBackend::DestroyJoint(IPhysicsJoint* joint) {
    if (!joint || !m_World) return;
    
    auto it = std::find(m_Joints.begin(), m_Joints.end(), joint);
    if (it != m_Joints.end()) {
        Box2DJoint* bJoint = static_cast<Box2DJoint*>(joint);
        m_World->DestroyJoint(bJoint->GetNativeJoint());
        delete bJoint;
        m_Joints.erase(it);
    }
}

void Box2DBackend::SetPlane(Plane2D plane) {
    m_Plane = plane;
    // Re-initialize or wake up bodies if needed?
    // Changing plane at runtime might be weird for existing bodies if they are not updated.
    // Ideally set before initialization.
}

b2Vec2 Box2DBackend::ToBox2D(const Vec3& v) const {
    if (m_Plane == Plane2D::XZ) {
        return b2Vec2(v.x, v.z);
    }
    return b2Vec2(v.x, v.y);
}

Vec3 Box2DBackend::ToVec3(const b2Vec2& v) const {
    if (m_Plane == Plane2D::XZ) {
        return Vec3(v.x, 0.0f, v.y);
    }
    return Vec3(v.x, v.y, 0.0f);
}

std::shared_ptr<IPhysicsCharacterController> Box2DBackend::CreateCharacterController() {
    Box2DCharacterController* controller = new Box2DCharacterController(this);
    m_Controllers.push_back(controller);
    
    return std::shared_ptr<IPhysicsCharacterController>(controller, [this](IPhysicsCharacterController* c) {
        this->DestroyCharacterController(c);
    });
}

void Box2DBackend::DestroyCharacterController(IPhysicsCharacterController* controller) {
    if (!controller) return;
    auto it = std::find(m_Controllers.begin(), m_Controllers.end(), controller);
    if (it != m_Controllers.end()) {
        delete *it;
        m_Controllers.erase(it);
    }
}

#endif // USE_BOX2D
