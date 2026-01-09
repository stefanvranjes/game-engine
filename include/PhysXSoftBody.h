#pragma once

#include "IPhysicsSoftBody.h"
#include "Math/Vec3.h"
#include <memory>
#include <vector>
#include <functional>

class SoftBodyTearSystem;
class SoftBodyPieceManager;
class SoftBodyTearPattern;
class TearResistanceMap;

#ifdef USE_PHYSX

namespace physx {
    class PxSoftBody;
    class PxTetrahedronMesh;
    class PxSoftBodyMesh;
    class PxShape;
}

class PhysXBackend;
class IPhysicsRigidBody;

/**
 * @brief PhysX implementation of soft body physics
 * 
 * Uses PhysX 5.x GPU-accelerated soft body simulation with Position-Based Dynamics.
 * Supports volume preservation, shape matching, deformation limits, and collision.
 */
class PhysXSoftBody : public IPhysicsSoftBody {
public:
    PhysXSoftBody(PhysXBackend* backend);
    ~PhysXSoftBody() override;

    // IPhysicsSoftBody implementation
    void Initialize(const SoftBodyDesc& desc) override;
    void Update(float deltaTime) override;
    void SetEnabled(bool enabled) override;
    bool IsEnabled() const override;

    // Vertex/Particle Access
    int GetVertexCount() const override;
    void GetVertexPositions(Vec3* positions) const override;
    void SetVertexPositions(const Vec3* positions) override;
    void GetVertexNormals(Vec3* normals) const override;
    void GetVertexVelocities(Vec3* velocities) const override;

    // Material Properties
    void SetVolumeStiffness(float stiffness) override;
    float GetVolumeStiffness() const override;
    void SetShapeStiffness(float stiffness) override;
    float GetShapeStiffness() const override;
    void SetDeformationStiffness(float stiffness) override;
    float GetDeformationStiffness() const override;
    void SetMaxStretch(float maxStretch) override;
    void SetMaxCompress(float maxCompress) override;
    void SetDamping(float linear, float angular) override;

    // Forces
    void AddForce(const Vec3& force) override;
    void AddForceAtVertex(int vertexIndex, const Vec3& force) override;
    void AddImpulse(const Vec3& impulse) override;
    void AddImpulseAtVertex(int vertexIndex, const Vec3& impulse) override;

    // Attachments
    void AttachVertexToRigidBody(int vertexIndex, IPhysicsRigidBody* rigidBody, const Vec3& localPos) override;
    void DetachVertex(int vertexIndex) override;
    void FixVertex(int vertexIndex, const Vec3& worldPos) override;
    void UnfixVertex(int vertexIndex) override;

    // Collision
    void SetSceneCollision(bool enabled) override;
    void SetSelfCollision(bool enabled) override;
    void SetCollisionMargin(float margin) override;
    void AddCollisionSphere(const Vec3& center, float radius) override;
    void AddCollisionCapsule(const Vec3& p0, const Vec3& p1, float radius) override;

    // Tearing/Fracture
    void SetTearable(bool tearable) override;
    bool TearAtVertex(int vertexIndex) override;

    // State Queries
    float GetTotalMass() const override;
    float GetVolume() const override;
    Vec3 GetCenterOfMass() const override;
    bool IsActive() const override;
    void SetActive(bool active) override;
    void* GetNativeSoftBody() override;

    // PhysX specific
    physx::PxSoftBody* GetPxSoftBody() const { return m_SoftBody; }

    // Tearing
    virtual void SetTearThreshold(float threshold) override;
    virtual float GetTearThreshold() const override;
    
    /**
     * @brief Set callback for tear events
     */
    void SetTearCallback(std::function<void(int tetrahedronIndex, float stress)> callback);
    
    /**
     * @brief Set callback for piece creation
     */
    void SetPieceCreatedCallback(std::function<void(std::shared_ptr<PhysXSoftBody>)> callback);
    
    /**
     * @brief Tear along pattern
     */
    void TearAlongPattern(
        const SoftBodyTearPattern& pattern,
        const Vec3& startPoint,
        const Vec3& endPoint
    );
    
    /**
     * @brief Tear with straight line
     */
    void TearStraightLine(const Vec3& start, const Vec3& end, float width = 0.1f);
    
    /**
     * @brief Tear with curved path
     */
    void TearCurvedPath(const Vec3& start, const Vec3& end, float curvature = 0.5f);
    
    /**
     * @brief Tear with radial burst
     */
    void TearRadialBurst(const Vec3& center, int rayCount = 8, float radius = 1.0f);
    
    /**
     * @brief Get tear resistance map
     */
    TearResistanceMap& GetResistanceMap() { return m_ResistanceMap; }
    const TearResistanceMap& GetResistanceMap() const { return m_ResistanceMap; }
    
    /**
     * @brief Set resistance for region (sphere)
     */
    void SetRegionResistance(const Vec3& center, float radius, float resistance);
    
    /**
     * @brief Set resistance gradient
     */
    void SetResistanceGradient(
        const Vec3& start, const Vec3& end,
        float startResistance, float endResistance
    );
    
    /**
     * @brief Enable tear healing
     */
    void SetHealingEnabled(bool enabled);
    
    /**
     * @brief Set healing rate (progress per second)
     */
    void SetHealingRate(float progressPerSecond);
    
    /**
     * @brief Set delay before healing starts
     */
    void SetHealingDelay(float seconds);
    
    /**
     * @brief Get number of healing tears
     */
    int GetHealingTearCount() const;

private:
    PhysXBackend* m_Backend;
    physx::PxSoftBody* m_SoftBody;
    physx::PxTetrahedronMesh* m_TetraMesh;
    physx::PxSoftBodyMesh* m_SoftBodyMesh;

    // Configuration
    bool m_Enabled;
    bool m_Tearable;
    float m_VolumeStiffness;
    float m_ShapeStiffness;
    float m_DeformationStiffness;
    float m_MaxStretch;
    float m_MaxCompress;
    float m_LinearDamping;
    float m_AngularDamping;
    float m_CollisionMargin;
    bool m_SceneCollisionEnabled;
    bool m_SelfCollisionEnabled;

    // Mesh data
    int m_VertexCount;
    int m_TetrahedronCount;
    std::vector<Vec3> m_InitialPositions;
    std::vector<int> m_FixedVertices;

    // Collision shapes
    struct CollisionShape {
        int type; // 0 = sphere, 1 = capsule
        Vec3 pos0;
        Vec3 pos1; // Only for capsule
        float radius;
        physx::PxShape* shape;
    };
    std::vector<CollisionShape> m_CollisionShapes;
    
    // Tearing system
    std::unique_ptr<SoftBodyTearSystem> m_TearSystem;
    TearResistanceMap m_ResistanceMap;
    std::vector<Vec3> m_RestPositions;  // Original vertex positions
    std::vector<int> m_TetrahedronIndices;  // Tetrahedral indices
    float m_TearThreshold;
    bool m_CheckTearing;
    float m_LastTearCheckTime;
    float m_TearCheckInterval;  // Check every N seconds
    
    // Callbacks
    std::function<void(int, float)> m_TearCallback;
    std::function<void(std::shared_ptr<PhysXSoftBody>)> m_PieceCreatedCallback;

    // Helper methods
    void CreateTetrahedralMesh(const SoftBodyDesc&amp; desc);
    void RecalculateNormals(Vec3* normals) const;
    void UpdateCollisionShapes();
};

#endif // USE_PHYSX
