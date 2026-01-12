#include "PhysXBackend.h"
#include "PhysXAggregate.h"
#include "PhysXRigidBody.h"
#include "PhysXShape.h"
#include "Math/Vec3.h"

#ifdef USE_PHYSX

void RunAggregateDemo() {
    PhysXBackend backend;
    backend.Initialize(Vec3(0, -9.81f, 0));

    // Create Aggregate (max 10 bodies, self collision enabled)
    PhysXAggregate aggregate(&backend, 10, true);
    aggregate.AddToScene();

    // Create Bodies
    for (int i = 0; i < 5; i++) {
        auto body = std::make_shared<PhysXRigidBody>(&backend);
        auto shape = PhysXShape::CreateBox(&backend, Vec3(0.5f));
        body->Initialize(BodyType::Dynamic, 1.0f, shape);
        
        // Add to aggregate (automatically manages scene addition)
        aggregate.AddBody(body);
        
        body->SyncTransformToPhysics(Vec3(0, 5 + i * 2.0f, 0), Quat::Identity());
    }

    // Step physics
    for (int i = 0; i < 100; i++) {
        backend.Update(0.016f);
    }

    // Cleanup
    aggregate.RemoveFromScene(); // Optional, destructor handles it
    backend.Shutdown();
}

#endif
