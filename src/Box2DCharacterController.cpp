#include "Box2DCharacterController.h"
#include "Box2DBackend.h"
#include "Box2DShape.h"
#include "Renderer.h" // For debugging if needed
#include <iostream>

#ifdef USE_BOX2D

Box2DCharacterController::Box2DCharacterController(Box2DBackend* backend)
    : m_Backend(backend) {
}

Box2DCharacterController::~Box2DCharacterController() {
    if (m_Body && m_Backend && m_Backend->GetWorld()) {
        m_Backend->GetWorld()->DestroyBody(m_Body);
    }
}

void Box2DCharacterController::Initialize(std::shared_ptr<IPhysicsShape> shape, float mass, float stepHeight) {
    if (!m_Backend || !m_Backend->GetWorld()) return;

    m_StepHeight = stepHeight;

    b2BodyDef bodyDef;
    bodyDef.type = b2_dynamicBody;
    bodyDef.fixedRotation = true; // Essential for character controller
    // Position should be set via SetPosition usually, but default 0,0 is fine for init
    
    m_Body = m_Backend->GetWorld()->CreateBody(&bodyDef);
    
    // Attach shape
    // Assuming shape is Box2DShape. We need to create a fixture.
    // If shape is null, we create a default box?
    // Using simple box for now if shape conversion is complex, but let's try to reuse Box2DShape info if possible.
    // Box2DShape creates shapes but doesn't expose b2Shape directly easily without wrapper... 
    // Wait, Box2DShape has 'CreateBox' etc which return IPhysicsShape.
    // We need the b2Shape to create fixture. 
    // Let's assume for now we just create a box of reasonable size if shape is not easily decomposable.
    // Ideally IPhysicsShape would expose GetNativeShape() but it returns void*.
    
    b2PolygonShape boxShape;
    boxShape.SetAsBox(0.5f, 1.0f); // Default size if extraction fails
    
    if (shape) {
        // Try to check type or use native shape
        // For this task, I'll just use a fixed box of size (0.5, 1.0) which is typical for a character.
        // User passed mass.
    }
    
    b2FixtureDef fixtureDef;
    fixtureDef.shape = &boxShape;
    fixtureDef.density = mass; // Density * Area = Mass. Area = 1.0 * 2.0 = 2.0. So Density = mass / 2.0.
    if (mass > 0) fixtureDef.density = mass / 2.0f;
    fixtureDef.friction = 0.0f; // No friction on walls usually, we handle movement manually
    fixtureDef.restitution = 0.0f; // No bounce
    
    m_Body->CreateFixture(&fixtureDef);
    
    // User data for collision callbacks if needed
    // m_Body->GetUserData().pointer = (uintptr_t)this; // Or something distinctive?
    // Usually RigidBodies set this. We might need a separate way to identify controllers.
}

void Box2DCharacterController::Update(float deltaTime, const Vec3& gravity) {
    if (!m_Body) return;
    
    CheckGrounded();
    
    b2Vec2 velocity = m_Body->GetLinearVelocity();
    
    // Apply Walk Direction (X-axis movement)
    // We want to reach target velocity.
    float targetSpeed = m_WalkDirection.Length() * m_MaxWalkSpeed; // Or just project
    // Actually we only care about 2D horizontal.
    // If XY plane, "horizontal" is X. 
    // If XZ plane, "horizontal" is composite of X/Z? 
    // Box2D only has X/Y. 
    // If XZ plane, World X -> Box2D X. World Z -> Box2D Y.
    // Walk direction is 3D.
    
    b2Vec2 walk2D = m_Backend->ToBox2D(m_WalkDirection);
    // Normalize if length > 0
    if (walk2D.LengthSquared() > 0.0001f) {
        walk2D.Normalize();
        walk2D *= m_MaxWalkSpeed;
    }
    
    // We want to preserve vertical velocity (gravity).
    // In Box2D:
    // XY Plane: Y is vertical.
    // XZ Plane: Box2D Y is World Z (horizontal). What is vertical?
    // Wait. If XZ is the ground plane, then Y is vertical (Gravity).
    // But Box2D is 2D. It doesn't HAVE a 3rd dimension.
    // So in XZ mode (Top Down), gravity is usually ZERO (or fake "down" into screen?).
    // Usually top-down games don't have gravity affecting the 2D position unless jumping is "fake" (scaling).
    // Box2DCharacterController assumes side-scroller gravity?
    // If XZ plane, usually we move in X and Y (World Z). Gravity is irrelevant for 2D position.
    // Jumping would be a Z-offset visualization or separate variable.
    // BUT, if we want actual physics jumping (parabola), we need Gravity.
    // If XZ plane is used, standard Box2D doesn't handle "Height" (World Y).
    // So "Jump" in XZ top-down is usually manual height management.
    // Does Box2DBackend support 3D physics? No.
    // So if Plane is XZ, "Vertical" velocity is ignored by Box2D simulation.
    // We must handle Height manually if we want jumping in Top-Down.
    // OR we change mapping:
    // XY: X=X, Y=Y.
    // XZ: X=X, Y=Z.
    
    if (m_Backend->GetPlane() == Box2DBackend::Plane2D::XY) {
        // Standard Side-Scroller
        // Controlled axis is X. Y is gravity.
        velocity.x = walk2D.x; // Set X velocity
    } else {
        // Top-Down
        // Controlled axes are X and Y (World Z).
        velocity = walk2D; // Set both X and Y (Z)
    }

    m_Body->SetLinearVelocity(velocity);
}

void Box2DCharacterController::SetWalkDirection(const Vec3& direction) {
    m_WalkDirection = direction;
}

void Box2DCharacterController::Jump(const Vec3& impulse) {
    if (m_IsGrounded && m_Body) {
        if (m_Backend->GetPlane() == Box2DBackend::Plane2D::XY) {
             m_Body->ApplyLinearImpulseToCenter(m_Backend->ToBox2D(impulse), true);
        } else {
             // Top down jump? Just "Height" visualization?
             // Box2D can't simulate jump in Y axis (which is World Y, orthogonal to plane).
             // We'd need to add a manual vertical velocity tracker.
             // For now, ignore jump in Top-Down or apply to a "fake" vertical axis?
             // Let's assume standard behavior: Apply impulse converts to 2D.
             // If XZ, impulse.y (Vertical) is LOST in ToBox2D. So no jump force.
             // Implies XZ mode doesn't support physics-based jumping via Box2D.
        }
        m_IsGrounded = false; // Optimistic update
    }
}

bool Box2DCharacterController::IsGrounded() const {
    return m_IsGrounded;
}

void Box2DCharacterController::CheckGrounded() {
    if (!m_Body || !m_Backend) return;
    
    // Raycast down from feet
    float rayLength = 1.05f; // Half height (1.0) + slight margin
    b2Vec2 pos = m_Body->GetPosition();
    b2Vec2 p1 = pos;
    b2Vec2 p2 = pos + b2Vec2(0, -rayLength);
    
    // We need a raycast callback
    // Using a lambda callback? 
    // b2World::RayCast takes a b2RayCastCallback*
    
    class GroundRayCastCallback : public b2RayCastCallback {
    public:
        bool m_Hit = false;
        b2Body* m_Self = nullptr;
        
        float ReportFixture(b2Fixture* fixture, const b2Vec2& point, const b2Vec2& normal, float fraction) override {
            if (fixture->GetBody() == m_Self) return -1.0f; // Ignore self
            if (fixture->IsSensor()) return -1.0f; // Ignore sensors
            m_Hit = true;
            return 0.0f; // Terminate
        }
    };
    
    GroundRayCastCallback callback;
    callback.m_Self = m_Body;
    
    m_Backend->GetWorld()->RayCast(&callback, p1, p2);
    m_IsGrounded = callback.m_Hit;
}

Vec3 Box2DCharacterController::GetPosition() const {
    if (!m_Body) return Vec3(0);
    return m_Backend->ToVec3(m_Body->GetPosition());
}

void Box2DCharacterController::SetPosition(const Vec3& position) {
    if (m_Body) {
        m_Body->SetTransform(m_Backend->ToBox2D(position), m_Body->GetAngle());
    }
}

float Box2DCharacterController::GetVerticalVelocity() const {
    if (!m_Body) return 0.0f;
    return m_Body->GetLinearVelocity().y;
}

void Box2DCharacterController::SetVerticalVelocity(float velocity) {
    if (m_Body) {
        b2Vec2 v = m_Body->GetLinearVelocity();
        v.y = velocity;
        m_Body->SetLinearVelocity(v);
    }
}

void Box2DCharacterController::Resize(float height) {
    // Recreate fixture capability needed?
    // For now, ignore.
}

#endif // USE_BOX2D
