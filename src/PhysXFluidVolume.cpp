#include "PhysXFluidVolume.h"
#include "PhysXRigidBody.h"
#include "PhysXShape.h" // If we need shape access
#include <algorithm>
#include <cmath>

#ifdef USE_PHYSX
#include "PxPhysicsAPI.h"
using namespace physx;
#endif

PhysXFluidVolume::PhysXFluidVolume(PhysXRigidBody* triggerBody)
    : m_TriggerBody(triggerBody)
{
}

PhysXFluidVolume::~PhysXFluidVolume()
{
}

float PhysXFluidVolume::CalculateSubmergedVolume(PhysXRigidBody* body, Vec3& outBuoyancyCenter, Vec3& outFlowVelocity)
{
    if (!body || !m_TriggerBody) return 0.0f;

#ifdef USE_PHYSX
    // Get Bounds of body
    PxRigidActor* actor = static_cast<PxRigidActor*>(body->GetNativeBody());
    if (!actor) return 0.0f;

    PxBounds3 bodyBounds = actor->getWorldBounds();
    
    // Get Bounds of fluid
    PxRigidActor* fluidActor = static_cast<PxRigidActor*>(m_TriggerBody->GetNativeBody());
    if (!fluidActor) return 0.0f;

    PxBounds3 fluidBounds = fluidActor->getWorldBounds();

    // Check intersection
    if (!bodyBounds.intersects(fluidBounds)) {
        return 0.0f;
    }

    // Determine Fluid Surface Height
    // If wave height is used, we might query it. For now, assume top of fluid bounds or BaseHeight
    float surfaceHeight = fluidBounds.maximum.y;
    if (UseWaveHeight) {
        // TODO: Query wave height at body position
        surfaceHeight = BaseHeight;
    }

    // Clip body bounds to fluid bounds
    PxBounds3 submergedBounds;
    submergedBounds.minimum = bodyBounds.minimum.maximum(fluidBounds.minimum);
    submergedBounds.maximum = bodyBounds.maximum.minimum(fluidBounds.maximum);

    // Further clip top by surface height (in case fluid bounds are huge but water level is lower?)
    // Actually fluidBounds should define the volume.
    // If it's an infinite ocean, usually the volume is defined large enough.
    
    // Calculate submerged volume (AABB approximation)
    PxVec3 dims = submergedBounds.getDimensions();
    if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) return 0.0f;
    
    float submergedVol = dims.x * dims.y * dims.z;

    // Calculate total volume approximation
    PxVec3 bodyDims = bodyBounds.getDimensions();
    float totalVol = bodyDims.x * bodyDims.y * bodyDims.z;
    
    if (totalVol <= 1e-6f) return 0.0f;

    float fraction = submergedVol / totalVol;

    // Center of buoyancy is center of submerged volume
    PxVec3 center = submergedBounds.getCenter();
    outBuoyancyCenter = Vec3(center.x, center.y, center.z);
    
    // Flow
    outFlowVelocity = FlowDirection.Normalized() * FlowSpeed;
    if (FlowDirection.LengthSquared() < 0.0001f) {
         outFlowVelocity = Vec3(0,0,0);
    }

    return fraction;
#else
    return 0.0f;
#endif
}
