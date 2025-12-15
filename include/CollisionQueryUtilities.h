#pragma once

#include "Math/Vec3.h"
#include "PhysicsSystem.h"
#include <vector>
#include <memory>
#include <optional>

// Forward declarations
class RigidBody;
class KinematicController;
class btCollisionObject;

/**
 * @brief Extended raycast hit information
 * 
 * Includes the hit object reference for easier access to game logic
 */
struct ExtendedRaycastHit : public RaycastHit {
    std::shared_ptr<RigidBody> rigidBody;  // Shared pointer to the hit rigid body (if any)
    bool hasRigidBody = false;              // True if hit object is a rigid body
};

/**
 * @brief Result of a sphere/capsule sweep test
 */
struct SweepTestResult {
    bool hasHit;                            // Was an object hit?
    Vec3 hitPoint;                          // World space point of impact
    Vec3 hitNormal;                         // Surface normal at impact
    float distance;                         // Distance traveled before hit
    float fraction;                         // Fraction of sweep distance (0-1)
    std::shared_ptr<RigidBody> hitBody;   // The rigid body that was hit
};

/**
 * @brief Result of an overlap/aabb test
 */
struct OverlapTestResult {
    std::vector<std::shared_ptr<RigidBody>> overlappingBodies;
    int count = 0;
};

/**
 * @brief Collision query utilities for physics queries
 * 
 * Provides advanced collision detection queries beyond basic raycasting:
 * - Sweep tests (sphere/capsule movement detection)
 * - Overlap tests (AABB and sphere overlap)
 * - Multi-hit raycasts
 * - Distance queries
 * - Ground detection for characters
 * 
 * Usage:
 *   auto result = CollisionQueryUtilities::RaycastAll(from, to);
 *   auto sweep = CollisionQueryUtilities::SweepSphere(start, end, radius);
 *   auto overlaps = CollisionQueryUtilities::OverlapSphere(center, radius);
 */
class CollisionQueryUtilities {
public:
    // ==================== RAYCAST QUERIES ====================
    
    /**
     * @brief Perform a raycast and get all hits along the ray
     * @param from Ray start position
     * @param to Ray end position
     * @param maxHits Maximum number of hits to return (0 = unlimited)
     * @return Vector of all hits along the ray
     */
    static std::vector<ExtendedRaycastHit> RaycastAll(
        const Vec3& from,
        const Vec3& to,
        int maxHits = 0
    );

    /**
     * @brief Perform a single raycast (standard raycast)
     * @param from Ray start position
     * @param to Ray end position
     * @param outHit Output hit information
     * @return True if a hit was detected
     */
    static bool Raycast(
        const Vec3& from,
        const Vec3& to,
        ExtendedRaycastHit& outHit
    );

    /**
     * @brief Raycast in a direction with a maximum distance
     * @param origin Ray origin
     * @param direction Normalized direction vector
     * @param maxDistance Maximum ray distance
     * @param outHit Output hit information
     * @return True if a hit was detected
     */
    static bool RaycastDirection(
        const Vec3& origin,
        const Vec3& direction,
        float maxDistance,
        ExtendedRaycastHit& outHit
    );

    // ==================== SWEEP TESTS ====================

    /**
     * @brief Sweep a sphere along a path and detect collisions
     * @param from Start position
     * @param to End position
     * @param radius Sphere radius
     * @return Result containing first hit or empty result if no hit
     */
    static SweepTestResult SweepSphere(
        const Vec3& from,
        const Vec3& to,
        float radius
    );

    /**
     * @brief Sweep a capsule (character shape) along a path
     * @param from Start position
     * @param to End position
     * @param radius Capsule radius
     * @param height Capsule height
     * @return Result containing first hit or empty result if no hit
     */
    static SweepTestResult SweepCapsule(
        const Vec3& from,
        const Vec3& to,
        float radius,
        float height
    );

    /**
     * @brief Sweep an axis-aligned box along a path
     * @param from Start position
     * @param to End position
     * @param halfExtents Half-size of the box (width/2, height/2, depth/2)
     * @return Result containing first hit or empty result if no hit
     */
    static SweepTestResult SweepBox(
        const Vec3& from,
        const Vec3& to,
        const Vec3& halfExtents
    );

    // ==================== OVERLAP QUERIES ====================

    /**
     * @brief Test if a sphere overlaps with any physics objects
     * @param center Sphere center position
     * @param radius Sphere radius
     * @return Result containing all overlapping bodies
     */
    static OverlapTestResult OverlapSphere(
        const Vec3& center,
        float radius
    );

    /**
     * @brief Test if an AABB overlaps with any physics objects
     * @param center AABB center position
     * @param halfExtents Half-size of the box
     * @return Result containing all overlapping bodies
     */
    static OverlapTestResult OverlapBox(
        const Vec3& center,
        const Vec3& halfExtents
    );

    /**
     * @brief Test if a capsule overlaps with any physics objects
     * @param center Capsule center position
     * @param radius Capsule radius
     * @param height Capsule height
     * @return Result containing all overlapping bodies
     */
    static OverlapTestResult OverlapCapsule(
        const Vec3& center,
        float radius,
        float height
    );

    // ==================== DISTANCE QUERIES ====================

    /**
     * @brief Get the closest point on a physics body to a given point
     * @param bodyPosition Position of the physics body
     * @param point Query point
     * @param outClosestPoint Closest point on the body
     * @param outDistance Distance to closest point
     * @return True if query succeeded
     */
    static bool GetClosestPointOnBody(
        const Vec3& bodyPosition,
        const Vec3& point,
        Vec3& outClosestPoint,
        float& outDistance
    );

    /**
     * @brief Calculate distance between two spheres
     * @param pos1 Center of first sphere
     * @param radius1 Radius of first sphere
     * @param pos2 Center of second sphere
     * @param radius2 Radius of second sphere
     * @return Distance between sphere surfaces (0 if overlapping, negative if overlapping)
     */
    static float SphereDistance(
        const Vec3& pos1,
        float radius1,
        const Vec3& pos2,
        float radius2
    );

    // ==================== CHARACTER QUERIES ====================

    /**
     * @brief Check if there's ground below a character controller
     * @param controller The kinematic controller to test
     * @param groundDistance How far below to check for ground
     * @param outDistance Distance to ground (0 if already on ground)
     * @return True if ground was detected
     */
    static bool IsGroundDetected(
        const std::shared_ptr<KinematicController>& controller,
        float groundDistance = 0.5f,
        float* outDistance = nullptr
    );

    /**
     * @brief Check if a character controller can move in a direction
     * @param controller The kinematic controller
     * @param direction Desired movement direction (normalized or with magnitude for distance)
     * @param maxStepHeight Maximum step height to allow
     * @return True if movement is possible (no collision blocking)
     */
    static bool CanMove(
        const std::shared_ptr<KinematicController>& controller,
        const Vec3& direction,
        float maxStepHeight = 0.5f
    );

    /**
     * @brief Find a valid movement position for a character controller
     * Useful for unstuck operations or AI pathfinding adjustments
     * @param controller The kinematic controller
     * @param targetPosition Desired position
     * @param searchRadius How far to search for a valid position
     * @param outValidPosition A valid position that can be reached
     * @return True if a valid position was found
     */
    static bool FindValidPosition(
        const std::shared_ptr<KinematicController>& controller,
        const Vec3& targetPosition,
        float searchRadius,
        Vec3& outValidPosition
    );

    // ==================== UTILITY QUERIES ====================

    /**
     * @brief Get all physics bodies within a certain distance of a point
     * @param center Center point
     * @param maxDistance Maximum distance to search
     * @return Vector of bodies within distance
     */
    static std::vector<std::shared_ptr<RigidBody>> GetBodiesInRadius(
        const Vec3& center,
        float maxDistance
    );

    /**
     * @brief Check if a line segment intersects with any physics geometry
     * @param from Start of line segment
     * @param to End of line segment
     * @param outIntersection Point of first intersection
     * @return True if intersection found
     */
    static bool LineIntersect(
        const Vec3& from,
        const Vec3& to,
        Vec3& outIntersection
    );

    /**
     * @brief Get contact points between two bodies
     * @param body1 First rigid body
     * @param body2 Second rigid body
     * @return Vector of contact point positions
     */
    static std::vector<Vec3> GetContactPoints(
        const std::shared_ptr<RigidBody>& body1,
        const std::shared_ptr<RigidBody>& body2
    );

    /**
     * @brief Check if two AABBs overlap
     * @param min1 Minimum corner of first AABB
     * @param max1 Maximum corner of first AABB
     * @param min2 Minimum corner of second AABB
     * @param max2 Maximum corner of second AABB
     * @return True if AABBs overlap
     */
    static bool AABBOverlap(
        const Vec3& min1,
        const Vec3& max1,
        const Vec3& min2,
        const Vec3& max2
    );

    // ==================== FILTER & LAYER QUERIES ====================

    /**
     * @brief Perform a raycast with collision filtering
     * @param from Ray start
     * @param to Ray end
     * @param filterGroup Collision group to test against
     * @param filterMask Collision mask to test against
     * @param outHit Output hit information
     * @return True if hit found matching filter criteria
     */
    static bool RaycastFiltered(
        const Vec3& from,
        const Vec3& to,
        uint32_t filterGroup,
        uint32_t filterMask,
        ExtendedRaycastHit& outHit
    );

    /**
     * @brief Get all bodies in a region matching collision filters
     * @param center Center of query region
     * @param halfExtents Half-extents of AABB region
     * @param filterGroup Collision group to test
     * @param filterMask Collision mask to test
     * @return Vector of matching bodies
     */
    static std::vector<std::shared_ptr<RigidBody>> GetBodiesInRegionFiltered(
        const Vec3& center,
        const Vec3& halfExtents,
        uint32_t filterGroup,
        uint32_t filterMask
    );

private:
    /**
     * @brief Helper to convert Bullet3D collision object to RigidBody
     */
    static std::shared_ptr<RigidBody> GetRigidBodyFromCollisionObject(btCollisionObject* obj);

    /**
     * @brief Helper to validate sweep test results
     */
    static void ValidateSweepResult(SweepTestResult& result);
};
