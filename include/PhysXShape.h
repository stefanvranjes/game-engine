#pragma once

#include "IPhysicsShape.h"
#include "Math/Vec3.h"
#include <memory>
#include <vector>

#ifdef USE_PHYSX

namespace physx {
    class PxShape;
    class PxGeometry;
    class PxMaterial;
}

class PhysXBackend;

/**
 * @brief PhysX implementation of collision shapes
 */
class PhysXShape : public IPhysicsShape {
public:
    PhysXShape(PhysXBackend* backend, PhysicsShapeType type);
    ~PhysXShape() override;

    // IPhysicsShape implementation
    PhysicsShapeType GetType() const override { return m_Type; }
    Vec3 GetLocalScaling() const override;
    void SetLocalScaling(const Vec3& scale) override;
    float GetMargin() const override;
    void SetMargin(float margin) override;
    void SetMargin(float margin) override;
    void SetTrigger(bool isTrigger) override;
    bool IsTrigger() const override;
    void* GetNativeShape() override;
    void AddChildShape(std::shared_ptr<IPhysicsShape> child, const Vec3& position, const Vec3& rotation) override;

    // Factory methods
    static std::shared_ptr<PhysXShape> CreateBox(PhysXBackend* backend, const Vec3& halfExtents);
    static std::shared_ptr<PhysXShape> CreateSphere(PhysXBackend* backend, float radius);
    static std::shared_ptr<PhysXShape> CreateCapsule(PhysXBackend* backend, float radius, float height);
    static std::shared_ptr<PhysXShape> CreateCylinder(PhysXBackend* backend, float radius, float height);
    static std::shared_ptr<PhysXShape> CreateCompound(PhysXBackend* backend);

private:
    PhysXBackend* m_Backend;
    physx::PxShape* m_Shape;
    PhysicsShapeType m_Type;
    Vec3 m_LocalScaling;

    // For compound shapes
    std::vector<std::shared_ptr<IPhysicsShape>> m_ChildShapes;
};

#endif // USE_PHYSX
