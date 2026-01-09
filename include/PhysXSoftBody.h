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
class FractureLine;

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

    // Debug Rendering
    virtual void DebugRender(Shader* shader) override;

    // Helpers
    void CreateDebugResources();
    void UpdateDebugBuffers();

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
    
    /**
     * @brief Enable plasticity
     */
    void SetPlasticityEnabled(bool enabled);
    
    /**
     * @brief Set plastic threshold (stress ratio)
     */
    void SetPlasticThreshold(float threshold);
    
    /**
     * @brief Set plasticity rate
     */
    void SetPlasticityRate(float rate);
    
    /**
     * @brief Reset to original rest shape
     */
    void ResetRestShape();
    
    /**
     * @brief Add fracture line
     */
    void AddFractureLine(const FractureLine& fractureLine);
    
    /**
     * @brief Clear all fracture lines
     */
    void ClearFractureLines();
    
    // Cloth Collision
    /**
     * @brief Get collision spheres representing soft body surface for cloth collision
     * @param positions Output array of sphere positions
     * @param radii Output array of sphere radii
     * @param maxSpheres Maximum number of spheres to generate (PhysX cloth limit is 32)
     * @return Number of spheres generated
     */
    int GetCollisionSpheres(std::vector<Vec3>& positions, std::vector<float>& radii, int maxSpheres = 32) const;
    
    /**
     * @brief Enable/disable collision with cloth
     */
    void SetClothCollisionEnabled(bool enabled);
    
    /**
     * @brief Check if cloth collision is enabled
     */
    bool IsClothCollisionEnabled() const { return m_ClothCollisionEnabled; }
    
    /**
     * @brief Set collision sphere radius scale
     */
    void SetCollisionSphereRadius(float radius) { m_CollisionSphereRadius = radius; }
    
    /**
     * @brief Get collision sphere radius
     */
    float GetCollisionSphereRadius() const { return m_CollisionSphereRadius; }
    
    /**
     * @brief Calculate optimal sphere count based on soft body complexity
     * @return Recommended number of collision spheres
     */
    int CalculateOptimalSphereCount() const;
    
    /**
     * @brief Set adaptive sphere count parameters
     * @param minSpheres Minimum spheres (default: 4)
     * @param maxSpheres Maximum spheres (default: 32)
     * @param verticesPerSphere Vertices needed per additional sphere (default: 50)
     */
    void SetAdaptiveSphereParams(int minSpheres, int maxSpheres, int verticesPerSphere);
    
    /**
     * @brief Get current adaptive parameters
     */
    void GetAdaptiveSphereParams(int& minSpheres, int& maxSpheres, int& verticesPerSphere) const;
    
    /**
     * @brief Enable/disable adaptive sphere count
     */
    void SetUseAdaptiveSphereCount(bool enabled) { m_UseAdaptiveSphereCount = enabled; }
    
    /**
     * @brief Check if adaptive sphere count is enabled
     */
    bool GetUseAdaptiveSphereCount() const { return m_UseAdaptiveSphereCount; }
    
    /**
     * @brief Surface area calculation mode
     */
    enum class SurfaceAreaMode {
        BoundingBox,  // Fast approximation using bounding box
        Triangles,    // Accurate calculation using surface triangles
        ConvexHull    // Collision envelope using convex hull
    };
    
    /**
     * @brief Calculate approximate surface area of soft body
     * @return Surface area in square meters
     */
    float CalculateSurfaceArea() const;
    
    /**
     * @brief Set surface area calculation mode
     * @param mode Calculation mode (BoundingBox or Triangles)
     */
    void SetSurfaceAreaMode(SurfaceAreaMode mode);
    
    /**
     * @brief Get current surface area calculation mode
     */
    SurfaceAreaMode GetSurfaceAreaMode() const { return m_SurfaceAreaMode; }

    enum class ConvexHullAlgorithm {
        QuickHull,
        GiftWrapping,
        Incremental,
        DivideAndConquer
    };

    /**
     * @brief Set the algorithm used for convex hull calculation
     * @param algo The algorithm efficiency/robustness trade-off
     */
    void SetConvexHullAlgorithm(ConvexHullAlgorithm algo);

    /**
     * @brief Get the current convex hull algorithm
     */
    ConvexHullAlgorithm GetConvexHullAlgorithm() const { return m_HullAlgorithm; }
    
    /**
     * @brief Set adaptive weights for sphere count calculation
     * @param vertexWeight Weight for vertex count component (0-1, default: 0.5)
     * @param areaWeight Weight for surface area component (0-1, default: 0.5)
     * @param areaPerSphere Surface area per sphere in m² (default: 1.0)
     */
    void SetAdaptiveWeights(float vertexWeight, float areaWeight, float areaPerSphere);
    
    /**
     * @brief Get current adaptive weights
     */
    void GetAdaptiveWeights(float& vertexWeight, float& areaWeight, float& areaPerSphere) const;

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
    
    // Cloth collision
    bool m_ClothCollisionEnabled;
    float m_CollisionSphereRadius;
    mutable std::vector<Vec3> m_CachedCollisionSpherePositions;
    mutable std::vector<float> m_CachedCollisionSphereRadii;
    mutable bool m_CollisionSpheresNeedUpdate;
    
    // Adaptive sphere count
    bool m_UseAdaptiveSphereCount;
    int m_MinCollisionSpheres;
    int m_MaxCollisionSpheres;
    int m_VerticesPerSphere;
    float m_AdaptiveVertexWeight;
    float m_AdaptiveAreaWeight;
    float m_AreaPerSphere;
    mutable float m_CachedSurfaceArea;
    mutable bool m_SurfaceAreaNeedsUpdate;
    SurfaceAreaMode m_SurfaceAreaMode;
    ConvexHullAlgorithm m_HullAlgorithm;

    // Helper methods
    void CreateTetrahedralMesh(const SoftBodyDesc& desc);
    void RecalculateNormals(Vec3* normals) const;
    void UpdateCollisionShapes();
    
    // Surface area calculation helpers
    float CalculateBoundingBoxArea() const;
    float CalculateTriangleArea() const;
    float CalculateConvexHullArea() const;
};

#endif // USE_PHYSX
