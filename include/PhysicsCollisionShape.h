#pragma once

#include "Math/Vec3.h"
#include <memory>

// Forward declarations
class btCollisionShape;
class btBoxShape;
class btSphereShape;
class btCapsuleShape;
class btCylinderShape;
class btCompoundShape;

/**
 * @brief Collision shape types supported by the physics engine
 */
enum class PhysicsShapeType {
    Box,
    Sphere,
    Capsule,
    Cylinder,
    Compound,
    Mesh,  // For future use with btBvhTriangleMeshShape
    Plane   // For future use with btStaticPlaneShape
};

/**
 * @brief Wrapper around Bullet3D collision shapes
 * Provides a unified interface for creating and managing various collision shape types
 */
class PhysicsCollisionShape {
public:
    PhysicsCollisionShape();
    ~PhysicsCollisionShape();

    // No copy semantics (unique ownership of btCollisionShape)
    PhysicsCollisionShape(const PhysicsCollisionShape&) = delete;
    PhysicsCollisionShape& operator=(const PhysicsCollisionShape&) = delete;

    // Move semantics allowed
    PhysicsCollisionShape(PhysicsCollisionShape&&) noexcept;
    PhysicsCollisionShape& operator=(PhysicsCollisionShape&&) noexcept;

    /**
     * @brief Create a box collision shape
     * @param halfExtents Half-size of the box in each dimension
     */
    static PhysicsCollisionShape CreateBox(const Vec3& halfExtents);

    /**
     * @brief Create a sphere collision shape
     * @param radius Radius of the sphere
     */
    static PhysicsCollisionShape CreateSphere(float radius);

    /**
     * @brief Create a capsule collision shape
     * @param radius Radius of the capsule
     * @param height Height of the capsule (distance between sphere centers)
     */
    static PhysicsCollisionShape CreateCapsule(float radius, float height);

    /**
     * @brief Create a cylinder collision shape
     * @param radius Radius of the cylinder
     * @param height Height of the cylinder
     */
    static PhysicsCollisionShape CreateCylinder(float radius, float height);

    /**
     * @brief Create a compound shape (combination of multiple shapes)
     * Useful for complex object geometries
     */
    static PhysicsCollisionShape CreateCompound();

    /**
     * @brief Add a child shape to a compound shape
     * @param childShape The shape to add
     * @param offset Local position offset of the child
     * @param rotation Local rotation of the child (as Vec3 in radians: roll, pitch, yaw)
     * @note Only valid for compound shapes
     */
    void AddChildShape(const PhysicsCollisionShape& childShape, const Vec3& offset, const Vec3& rotation = Vec3(0, 0, 0));

    /**
     * @brief Get the underlying Bullet3D collision shape
     */
    btCollisionShape* GetShape() const { return m_Shape; }

    /**
     * @brief Get the shape type
     */
    PhysicsShapeType GetType() const { return m_Type; }

    /**
     * @brief Get the local scaling of the shape
     */
    Vec3 GetLocalScaling() const;

    /**
     * @brief Set the local scaling of the shape
     */
    void SetLocalScaling(const Vec3& scale);

    /**
     * @brief Get the margin around the shape (for collision detection)
     */
    float GetMargin() const;

    /**
     * @brief Set the margin around the shape
     */
    void SetMargin(float margin);

private:
    btCollisionShape* m_Shape;
    PhysicsShapeType m_Type;

    // Private constructor for internal use
    PhysicsCollisionShape(btCollisionShape* shape, PhysicsShapeType type);
};
