#pragma once

#include "Math/Vec3.h"
#include <memory>

/**
 * @brief Physics shape type enumeration
 */
enum class PhysicsShapeType {
    Box,
    Sphere,
    Capsule,
    Cylinder,
    Compound,
    ConvexMesh,
    TriangleMesh,
    Plane
};

/**
 * @brief Abstract interface for collision shapes
 * 
 * Provides a common API for collision shape creation and management
 * across different physics backends.
 */
class IPhysicsShape {
public:
    virtual ~IPhysicsShape() = default;

    /**
     * @brief Get shape type
     * @return Shape type enumeration
     */
    virtual PhysicsShapeType GetType() const = 0;

    /**
     * @brief Get local scaling
     * @return Scale vector
     */
    virtual Vec3 GetLocalScaling() const = 0;

    /**
     * @brief Set local scaling
     * @param scale Scale vector
     */
    virtual void SetLocalScaling(const Vec3& scale) = 0;

    /**
     * @brief Get collision margin
     * @return Margin in meters
     */
    virtual float GetMargin() const = 0;

    /**
     * @brief Set collision margin
     * @param margin Margin in meters
     */
    virtual void SetMargin(float margin) = 0;

    /**
     * @brief Set if shape is a trigger
     * @param isTrigger True to make this a trigger volume
     */
    virtual void SetTrigger(bool isTrigger) = 0;

    /**
     * @brief Check if shape is a trigger
     * @return True if trigger
     */
    virtual bool IsTrigger() const = 0;

    /**
     * @brief Get backend-specific shape pointer
     * @return Opaque pointer to native shape
     */
    virtual void* GetNativeShape() = 0;

    /**
     * @brief Add child shape (for compound shapes)
     * @param child Child shape
     * @param position Local position offset
     * @param rotation Local rotation offset
     */
    virtual void AddChildShape(std::shared_ptr<IPhysicsShape> child, const Vec3& position, const Vec3& rotation) = 0;
};
