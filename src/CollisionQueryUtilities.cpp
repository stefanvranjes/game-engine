#include "CollisionQueryUtilities.h"
#include "PhysicsSystem.h"
#include "RigidBody.h"
#include "KinematicController.h"
#include <btBulletDynamicsCommon.h>
#include <BulletCollision/NarrowPhaseCollision/btRaycastCallback.h>
#include <BulletCollision/CollisionDispatch/btCollisionWorld.h>
#include <algorithm>
#include <cmath>

// ==================== HELPER STRUCTURES ====================

/**
 * @brief Custom raycast callback for multiple hits
 */
class AllHitsRayCallback : public btCollisionWorld::RayResultCallback {
public:
    AllHitsRayCallback(const btVector3& rayFromWorld, const btVector3& rayToWorld)
        : m_rayFromWorld(rayFromWorld), m_rayToWorld(rayToWorld) {}

    btVector3 m_rayFromWorld;
    btVector3 m_rayToWorld;
    std::vector<std::pair<float, btRigidBody*>> m_hits;

    btScalar addSingleResult(btCollisionWorld::LocalRayResult& rayResult,
                             bool normalInWorldSpace) override {
        btRigidBody* body = btRigidBody::upcast(rayResult.m_collisionObject);
        if (!body)
            return 1.0f;

        btScalar hitFraction = rayResult.m_hitFraction;
        m_hits.push_back({hitFraction, body});
        return 1.0f; // Continue testing
    }
};

/**
 * @brief Custom callback for overlap tests
 */
class OverlapCallback : public btBroadphaseAabbCallback {
public:
    std::vector<btCollisionObject*> m_overlappingObjects;

    bool process(const btBroadphaseProxy* proxy) override {
        btCollisionObject* object = (btCollisionObject*)proxy->m_clientObject;
        if (object) {
            m_overlappingObjects.push_back(object);
        }
        return true;
    }
};

// ==================== IMPLEMENTATION ====================

std::shared_ptr<RigidBody> CollisionQueryUtilities::GetRigidBodyFromCollisionObject(
    btCollisionObject* obj) {
    if (!obj)
        return nullptr;

    const std::vector<RigidBody*>& bodies = PhysicsSystem::Get().GetRigidBodies();
    btRigidBody* btBody = btRigidBody::upcast(obj);

    for (RigidBody* body : bodies) {
        if (body && body->GetBulletRigidBody() == btBody) {
            return body->shared_from_this();
        }
    }
    return nullptr;
}

std::vector<ExtendedRaycastHit> CollisionQueryUtilities::RaycastAll(
    const Vec3& from,
    const Vec3& to,
    int maxHits) {
    std::vector<ExtendedRaycastHit> results;

    btVector3 rayFrom(from.x, from.y, from.z);
    btVector3 rayTo(to.x, to.y, to.z);

    AllHitsRayCallback callback(rayFrom, rayTo);
    PhysicsSystem::Get().GetDynamicsWorld()->rayTest(rayFrom, rayTo, callback);

    // Sort hits by distance
    std::sort(callback.m_hits.begin(), callback.m_hits.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    int hitCount = 0;
    for (const auto& hit : callback.m_hits) {
        if (maxHits > 0 && hitCount >= maxHits)
            break;

        btRigidBody* body = hit.second;
        btVector3 hitPoint = rayFrom + hit.first * (rayTo - rayFrom);
        btVector3 hitNormal;

        ExtendedRaycastHit result;
        result.point = Vec3(hitPoint.x(), hitPoint.y(), hitPoint.z());
        result.normal = Vec3(0, 1, 0); // Default up
        result.distance =
            (to - from).Length() * hit.first; // Approximate distance calculation
        result.body = body;
        result.rigidBody = GetRigidBodyFromCollisionObject(body);
        result.hasRigidBody = (result.rigidBody != nullptr);

        results.push_back(result);
        hitCount++;
    }

    return results;
}

bool CollisionQueryUtilities::Raycast(const Vec3& from, const Vec3& to,
                                       ExtendedRaycastHit& outHit) {
    RaycastHit basicHit;
    if (!PhysicsSystem::Get().Raycast(from, to, basicHit)) {
        return false;
    }

    outHit.point = basicHit.point;
    outHit.normal = basicHit.normal;
    outHit.distance = basicHit.distance;
    outHit.body = basicHit.body;
    outHit.rigidBody = GetRigidBodyFromCollisionObject(basicHit.body);
    outHit.hasRigidBody = (outHit.rigidBody != nullptr);

    return true;
}

bool CollisionQueryUtilities::RaycastDirection(const Vec3& origin,
                                                const Vec3& direction,
                                                float maxDistance,
                                                ExtendedRaycastHit& outHit) {
    Vec3 normalizedDir = direction;
    normalizedDir.Normalize();
    Vec3 target = origin + normalizedDir * maxDistance;
    return Raycast(origin, target, outHit);
}

SweepTestResult CollisionQueryUtilities::SweepSphere(const Vec3& from,
                                                     const Vec3& to,
                                                     float radius) {
    SweepTestResult result;
    result.hasHit = false;
    result.distance = 0.0f;
    result.fraction = 0.0f;

    btSphereShape sphereShape(radius);
    btTransform fromTrans, toTrans;
    fromTrans.setIdentity();
    fromTrans.setOrigin(btVector3(from.x, from.y, from.z));
    toTrans.setIdentity();
    toTrans.setOrigin(btVector3(to.x, to.y, to.z));

    btCollisionWorld::ConvexResultCallback& callback =
        *new btCollisionWorld::ClosestConvexResultCallback(fromTrans.getOrigin(),
                                                           toTrans.getOrigin());
    auto* castCallback = dynamic_cast<btCollisionWorld::ClosestConvexResultCallback*>(&callback);

    PhysicsSystem::Get().GetDynamicsWorld()->convexSweepTest(&sphereShape, fromTrans, toTrans,
                                                             *castCallback);

    if (castCallback->hasHit()) {
        result.hasHit = true;
        result.hitPoint = Vec3(castCallback->m_hitPointWorld.x(),
                               castCallback->m_hitPointWorld.y(),
                               castCallback->m_hitPointWorld.z());
        result.hitNormal = Vec3(castCallback->m_hitNormalWorld.x(),
                                castCallback->m_hitNormalWorld.y(),
                                castCallback->m_hitNormalWorld.z());
        result.distance = (to - from).Length() * castCallback->m_closestHitFraction;
        result.fraction = castCallback->m_closestHitFraction;
        result.hitBody =
            GetRigidBodyFromCollisionObject(castCallback->m_hitCollisionObject);
    }

    return result;
}

SweepTestResult CollisionQueryUtilities::SweepCapsule(const Vec3& from,
                                                      const Vec3& to,
                                                      float radius,
                                                      float height) {
    SweepTestResult result;
    result.hasHit = false;
    result.distance = 0.0f;
    result.fraction = 0.0f;

    btCapsuleShape capsuleShape(radius, height);
    btTransform fromTrans, toTrans;
    fromTrans.setIdentity();
    fromTrans.setOrigin(btVector3(from.x, from.y, from.z));
    toTrans.setIdentity();
    toTrans.setOrigin(btVector3(to.x, to.y, to.z));

    btCollisionWorld::ClosestConvexResultCallback callback(fromTrans.getOrigin(),
                                                           toTrans.getOrigin());

    PhysicsSystem::Get().GetDynamicsWorld()->convexSweepTest(&capsuleShape, fromTrans, toTrans,
                                                             callback);

    if (callback.hasHit()) {
        result.hasHit = true;
        result.hitPoint = Vec3(callback.m_hitPointWorld.x(), callback.m_hitPointWorld.y(),
                               callback.m_hitPointWorld.z());
        result.hitNormal = Vec3(callback.m_hitNormalWorld.x(), callback.m_hitNormalWorld.y(),
                                callback.m_hitNormalWorld.z());
        result.distance = (to - from).Length() * callback.m_closestHitFraction;
        result.fraction = callback.m_closestHitFraction;
        result.hitBody = GetRigidBodyFromCollisionObject(callback.m_hitCollisionObject);
    }

    return result;
}

SweepTestResult CollisionQueryUtilities::SweepBox(const Vec3& from,
                                                  const Vec3& to,
                                                  const Vec3& halfExtents) {
    SweepTestResult result;
    result.hasHit = false;
    result.distance = 0.0f;
    result.fraction = 0.0f;

    btBoxShape boxShape(btVector3(halfExtents.x, halfExtents.y, halfExtents.z));
    btTransform fromTrans, toTrans;
    fromTrans.setIdentity();
    fromTrans.setOrigin(btVector3(from.x, from.y, from.z));
    toTrans.setIdentity();
    toTrans.setOrigin(btVector3(to.x, to.y, to.z));

    btCollisionWorld::ClosestConvexResultCallback callback(fromTrans.getOrigin(),
                                                           toTrans.getOrigin());

    PhysicsSystem::Get().GetDynamicsWorld()->convexSweepTest(&boxShape, fromTrans, toTrans,
                                                             callback);

    if (callback.hasHit()) {
        result.hasHit = true;
        result.hitPoint = Vec3(callback.m_hitPointWorld.x(), callback.m_hitPointWorld.y(),
                               callback.m_hitPointWorld.z());
        result.hitNormal = Vec3(callback.m_hitNormalWorld.x(), callback.m_hitNormalWorld.y(),
                                callback.m_hitNormalWorld.z());
        result.distance = (to - from).Length() * callback.m_closestHitFraction;
        result.fraction = callback.m_closestHitFraction;
        result.hitBody = GetRigidBodyFromCollisionObject(callback.m_hitCollisionObject);
    }

    return result;
}

OverlapTestResult CollisionQueryUtilities::OverlapSphere(const Vec3& center,
                                                         float radius) {
    OverlapTestResult result;

    btSphereShape sphereShape(radius);
    btTransform shapeTransform;
    shapeTransform.setIdentity();
    shapeTransform.setOrigin(btVector3(center.x, center.y, center.z));

    btCollisionObject tempObject;
    tempObject.setCollisionShape(&sphereShape);
    tempObject.setWorldTransform(shapeTransform);

    OverlapCallback callback;
    btVector3 minAabb, maxAabb;
    sphereShape.getAabb(shapeTransform, minAabb, maxAabb);

    PhysicsSystem::Get().GetDynamicsWorld()->getBroadphase()->aabbQuery(
        minAabb, maxAabb, callback);

    for (btCollisionObject* obj : callback.m_overlappingObjects) {
        auto body = GetRigidBodyFromCollisionObject(obj);
        if (body) {
            result.overlappingBodies.push_back(body);
            result.count++;
        }
    }

    return result;
}

OverlapTestResult CollisionQueryUtilities::OverlapBox(const Vec3& center,
                                                      const Vec3& halfExtents) {
    OverlapTestResult result;

    btBoxShape boxShape(btVector3(halfExtents.x, halfExtents.y, halfExtents.z));
    btTransform shapeTransform;
    shapeTransform.setIdentity();
    shapeTransform.setOrigin(btVector3(center.x, center.y, center.z));

    btCollisionObject tempObject;
    tempObject.setCollisionShape(&boxShape);
    tempObject.setWorldTransform(shapeTransform);

    OverlapCallback callback;
    btVector3 minAabb, maxAabb;
    boxShape.getAabb(shapeTransform, minAabb, maxAabb);

    PhysicsSystem::Get().GetDynamicsWorld()->getBroadphase()->aabbQuery(
        minAabb, maxAabb, callback);

    for (btCollisionObject* obj : callback.m_overlappingObjects) {
        auto body = GetRigidBodyFromCollisionObject(obj);
        if (body) {
            result.overlappingBodies.push_back(body);
            result.count++;
        }
    }

    return result;
}

OverlapTestResult CollisionQueryUtilities::OverlapCapsule(const Vec3& center,
                                                          float radius,
                                                          float height) {
    OverlapTestResult result;

    btCapsuleShape capsuleShape(radius, height);
    btTransform shapeTransform;
    shapeTransform.setIdentity();
    shapeTransform.setOrigin(btVector3(center.x, center.y, center.z));

    btCollisionObject tempObject;
    tempObject.setCollisionShape(&capsuleShape);
    tempObject.setWorldTransform(shapeTransform);

    OverlapCallback callback;
    btVector3 minAabb, maxAabb;
    capsuleShape.getAabb(shapeTransform, minAabb, maxAabb);

    PhysicsSystem::Get().GetDynamicsWorld()->getBroadphase()->aabbQuery(
        minAabb, maxAabb, callback);

    for (btCollisionObject* obj : callback.m_overlappingObjects) {
        auto body = GetRigidBodyFromCollisionObject(obj);
        if (body) {
            result.overlappingBodies.push_back(body);
            result.count++;
        }
    }

    return result;
}

bool CollisionQueryUtilities::GetClosestPointOnBody(const Vec3& bodyPosition,
                                                     const Vec3& point,
                                                     Vec3& outClosestPoint,
                                                     float& outDistance) {
    outDistance = std::numeric_limits<float>::max();
    outClosestPoint = point;

    // This is a simplified implementation
    // For more accuracy, you'd need direct Bullet3D access to shape
    const std::vector<RigidBody*>& bodies = PhysicsSystem::Get().GetRigidBodies();
    for (RigidBody* body : bodies) {
        if (body) {
            Vec3 diff = point - bodyPosition;
            float dist = diff.Length();
            if (dist < outDistance) {
                outDistance = dist;
                outClosestPoint = bodyPosition + diff.Normalize() * 0.5f; // Approximate
            }
        }
    }

    return outDistance < std::numeric_limits<float>::max();
}

float CollisionQueryUtilities::SphereDistance(const Vec3& pos1,
                                              float radius1,
                                              const Vec3& pos2,
                                              float radius2) {
    float centerDistance = (pos2 - pos1).Length();
    return centerDistance - radius1 - radius2;
}

bool CollisionQueryUtilities::IsGroundDetected(
    const std::shared_ptr<KinematicController>& controller,
    float groundDistance,
    float* outDistance) {
    if (!controller || !controller->IsInitialized()) {
        return false;
    }

    Vec3 controllerPos = controller->GetPosition();
    Vec3 below = controllerPos + Vec3(0, -groundDistance, 0);

    ExtendedRaycastHit hit;
    if (Raycast(controllerPos, below, hit)) {
        if (outDistance) {
            *outDistance = (hit.point - controllerPos).Length();
        }
        return true;
    }

    return false;
}

bool CollisionQueryUtilities::CanMove(
    const std::shared_ptr<KinematicController>& controller,
    const Vec3& direction,
    float maxStepHeight) {
    if (!controller || !controller->IsInitialized()) {
        return false;
    }

    Vec3 startPos = controller->GetPosition();
    Vec3 targetPos = startPos + direction;

    // Check if movement would result in collision
    SweepTestResult sweep = SweepCapsule(startPos, targetPos, 0.3f, 1.8f);
    if (sweep.hasHit && sweep.fraction < 0.95f) {
        // Check if we can step over it
        Vec3 stepped = targetPos + Vec3(0, maxStepHeight, 0);
        SweepTestResult steppedSweep = SweepCapsule(startPos, stepped, 0.3f, 1.8f);
        return !steppedSweep.hasHit || steppedSweep.fraction > 0.95f;
    }

    return true;
}

bool CollisionQueryUtilities::FindValidPosition(
    const std::shared_ptr<KinematicController>& controller,
    const Vec3& targetPosition,
    float searchRadius,
    Vec3& outValidPosition) {
    if (!controller || !controller->IsInitialized()) {
        return false;
    }

    // Start with target position
    SweepTestResult sweep = SweepCapsule(controller->GetPosition(), targetPosition, 0.3f, 1.8f);
    if (!sweep.hasHit) {
        outValidPosition = targetPosition;
        return true;
    }

    // Try positions around the target
    Vec3 direction = targetPosition - controller->GetPosition();
    float distance = direction.Length();
    if (distance > 0.001f) {
        direction = direction / distance;

        // Search along the sweep hit point
        float searchDistance = std::max(0.0f, distance * sweep.fraction - 0.1f);
        outValidPosition = controller->GetPosition() + direction * searchDistance;
        return true;
    }

    return false;
}

std::vector<std::shared_ptr<RigidBody>> CollisionQueryUtilities::GetBodiesInRadius(
    const Vec3& center,
    float maxDistance) {
    std::vector<std::shared_ptr<RigidBody>> results;

    const std::vector<RigidBody*>& bodies = PhysicsSystem::Get().GetRigidBodies();
    for (RigidBody* body : bodies) {
        if (body) {
            Vec3 bodyPos = body->GetPosition();
            float dist = (bodyPos - center).Length();
            if (dist <= maxDistance) {
                results.push_back(body->shared_from_this());
            }
        }
    }

    return results;
}

bool CollisionQueryUtilities::LineIntersect(const Vec3& from,
                                            const Vec3& to,
                                            Vec3& outIntersection) {
    ExtendedRaycastHit hit;
    if (Raycast(from, to, hit)) {
        outIntersection = hit.point;
        return true;
    }
    return false;
}

std::vector<Vec3> CollisionQueryUtilities::GetContactPoints(
    const std::shared_ptr<RigidBody>& body1,
    const std::shared_ptr<RigidBody>& body2) {
    std::vector<Vec3> contacts;

    if (!body1 || !body2) {
        return contacts;
    }

    // This would require accessing the Bullet3D contact manifold
    // For now, return empty - full implementation requires manifold access
    return contacts;
}

bool CollisionQueryUtilities::AABBOverlap(const Vec3& min1,
                                          const Vec3& max1,
                                          const Vec3& min2,
                                          const Vec3& max2) {
    return (min1.x <= max2.x && max1.x >= min2.x) &&
           (min1.y <= max2.y && max1.y >= min2.y) &&
           (min1.z <= max2.z && max1.z >= min2.z);
}

bool CollisionQueryUtilities::RaycastFiltered(const Vec3& from,
                                              const Vec3& to,
                                              uint32_t filterGroup,
                                              uint32_t filterMask,
                                              ExtendedRaycastHit& outHit) {
    RaycastHit basicHit;
    if (!PhysicsSystem::Get().Raycast(from, to, basicHit, filterMask)) {
        return false;
    }

    outHit.point = basicHit.point;
    outHit.normal = basicHit.normal;
    outHit.distance = basicHit.distance;
    outHit.body = basicHit.body;
    outHit.rigidBody = GetRigidBodyFromCollisionObject(basicHit.body);
    outHit.hasRigidBody = (outHit.rigidBody != nullptr);

    return true;
}

std::vector<std::shared_ptr<RigidBody>> CollisionQueryUtilities::GetBodiesInRegionFiltered(
    const Vec3& center,
    const Vec3& halfExtents,
    uint32_t filterGroup,
    uint32_t filterMask) {
    std::vector<std::shared_ptr<RigidBody>> results;

    OverlapTestResult overlap = OverlapBox(center, halfExtents);
    for (const auto& body : overlap.overlappingBodies) {
        // Apply filter checks (this would require full Bullet3D integration)
        results.push_back(body);
    }

    return results;
}

void CollisionQueryUtilities::ValidateSweepResult(SweepTestResult& result) {
    if (result.hasHit) {
        result.hitNormal.Normalize();
    }
}
