#pragma once

#include "IPhysicsCloth.h"
#include "Math/Vec3.h"
#include "ClothLOD.h"
#include "SpatialGrid.h"
#include <memory>
#include <vector>
#include <functional>

#ifdef USE_PHYSX

namespace physx {
    class PxCloth;
    class PxClothFabric;
    class PxClothParticle;
}

class PhysXBackend;
class Mesh;
class AsyncClothFactory;
class ClothTearPattern;
class ClothTearPatternLibrary;
class ClothMeshSynchronizer;
struct ClothSyncConfig;

/**
 * @brief PhysX implementation of cloth simulation
 * 
 * Uses PhysX 5.x GPU-accelerated cloth simulation with Position-Based Dynamics.
 * Supports stretching, bending, shearing constraints, wind forces, and collision.
 */
class PhysXCloth : public IPhysicsCloth {
public:
    PhysXCloth(PhysXBackend* backend);
    ~PhysXCloth() override;

    // IPhysicsCloth implementation
    void Initialize(const ClothDesc& desc) override;
    void Update(float deltaTime) override;
    void SetEnabled(bool enabled) override;
    bool IsEnabled() const override;
    int GetParticleCount() const override;
    void GetParticlePositions(Vec3* positions) const override;
    void SetParticlePositions(const Vec3* positions) override;
    void GetParticleNormals(Vec3* normals) const override;
    void SetStretchStiffness(float stiffness) override;
    void SetBendStiffness(float stiffness) override;
    void SetShearStiffness(float stiffness) override;
    void AddForce(const Vec3& force) override;
    void SetWindVelocity(const Vec3& velocity) override;
    void SetDamping(float damping) override;
    void AttachParticleToActor(int particleIndex, void* actor, const Vec3& localPos) override;
    void FreeParticle(int particleIndex) override;
    void AddCollisionSphere(const Vec3& center, float radius) override;
    void AddCollisionCapsule(const Vec3& p0, const Vec3& p1, float radius) override;
    void SetTearable(bool tearable) override;
    void SetMaxStretchRatio(float ratio) override;
    bool TearAtParticle(int particleIndex) override;
    int TearAlongLine(const Vec3& start, const Vec3& end) override;
    int GetTearCount() const override;
    void ResetTears() override;
    void* GetNativeCloth() override;
    void SetSceneCollision(bool enabled) override;
    void SetSelfCollision(bool enabled) override;
    void SetSelfCollisionDistance(float distance) override;
    void SetSelfCollisionStiffness(float stiffness) override;
    void SetTwoWayCoupling(bool enabled) override;
    void SetCollisionMassScale(float scale) override;

    // PhysX specific
    physx::PxCloth* GetPxCloth() const { return m_Cloth; }
    /**
     * @brief Update mesh data from cloth simulation
     * @param mesh Mesh to update with current particle positions/normals
     */
    void UpdateMeshData(Mesh* mesh);
    
    /**
     * @brief Get mesh synchronizer for advanced configuration
     */
    ClothMeshSynchronizer* GetMeshSynchronizer() { return m_MeshSynchronizer.get(); }
    
    /**
     * @brief Set mesh synchronization configuration
     */
    void SetSyncConfig(const ClothSyncConfig& config);
    
    /**
     * @brief Get current sync configuration
     */
    const ClothSyncConfig& GetSyncConfig() const;

    /**
     * @brief Get triangle count
     */
    int GetTriangleCount() const { return m_TriangleCount; }

    /**
     * @brief Get triangle indices
     */
    const std::vector<int>& GetTriangleIndices() const { return m_TriangleIndices; }

    // Mesh splitting (full tearing)
    /**
     * @brief Split cloth into two pieces at particle
     * @param tearParticle Particle to split at
     * @param outPiece1 First cloth piece (output)
     * @param outPiece2 Second cloth piece (output)
     * @return True if split was successful
     */
    bool SplitAtParticle(
        int tearParticle,
        std::shared_ptr<PhysXCloth>& outPiece1,
        std::shared_ptr<PhysXCloth>& outPiece2
    );

    /**
     * @brief Tear callback type
     * Called when cloth splits into two pieces
     * @param piece1 First cloth piece
     * @param piece2 Second cloth piece
     */
    using TearCallback = std::function<void(std::shared_ptr<PhysXCloth>, std::shared_ptr<PhysXCloth>)>;

    /**
     * @brief Set callback for tear events
     */
    void SetTearCallback(TearCallback callback) { m_TearCallback = callback; }

    /**
     * @brief Async initialization callback types
     */
    using AsyncCallback = std::function<void(std::shared_ptr<PhysXCloth>)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    /**
     * @brief Initialize cloth asynchronously
     * @param backend PhysX backend
     * @param desc Cloth descriptor
     * @param onComplete Callback when cloth is ready (main thread)
     * @param onError Callback on error (main thread)
     * @return Job ID for tracking
     */
    static int InitializeAsync(
        PhysXBackend* backend,
        const ClothDesc& desc,
        AsyncCallback onComplete,
        ErrorCallback onError = nullptr
    );

    /**
     * @brief Check if cloth is ready (has valid PhysX cloth actor)
     */
    bool IsReady() const { return m_Cloth != nullptr; }

    // LOD System
    /**
     * @brief Set LOD configuration
     */
    void SetLODConfig(const ClothLODConfig& config) { m_LODConfig = config; }

    /**
     * @brief Get LOD configuration
     */
    const ClothLODConfig& GetLODConfig() const { return m_LODConfig; }

    /**
     * @brief Set current LOD level
     */
    void SetLOD(int lodLevel);

    /**
     * @brief Get current LOD level
     */
    int GetCurrentLOD() const { return m_CurrentLOD; }

    /**
     * @brief Freeze cloth simulation (for distant cloth)
     */
    void Freeze();

    /**
     * @brief Unfreeze cloth simulation
     */
    void Unfreeze();

    /**
     * @brief Check if cloth is frozen
     */
    bool IsFrozen() const { return m_IsFrozen; }

    /**
     * @brief Split cloth along a line
     * @param start Line start position
     * @param end Line end position
     * @param outPiece1 First cloth piece (output)
     * @param outPiece2 Second cloth piece (output)
     * @return True if split was successful
     */
    bool SplitAlongLine(
        const Vec3& start,
        const Vec3& end,
        std::shared_ptr<PhysXCloth>& outPiece1,
        std::shared_ptr<PhysXCloth>& outPiece2
    );

    // ========================================================================
    // Pattern-Based Tearing System
    // ========================================================================

    /**
     * @brief Apply a tear pattern by name from the pattern library
     * @param patternName Name of pattern in library
     * @param position Pattern application position (world space)
     * @param direction Pattern orientation direction (normalized)
     * @param scale Pattern scale multiplier
     * @return True if pattern was applied successfully
     */
    bool ApplyTearPattern(
        const std::string& patternName,
        const Vec3& position,
        const Vec3& direction = Vec3(0, 0, 1),
        float scale = 1.0f
    );

    /**
     * @brief Apply a tear pattern directly
     * @param pattern Pattern to apply
     * @param position Pattern application position (world space)
     * @param direction Pattern orientation direction (normalized)
     * @param scale Pattern scale multiplier
     * @return True if pattern was applied successfully
     */
    bool ApplyTearPattern(
        std::shared_ptr<ClothTearPattern> pattern,
        const Vec3& position,
        const Vec3& direction = Vec3(0, 0, 1),
        float scale = 1.0f
    );

    /**
     * @brief Start a progressive tear that propagates over time
     * @param pattern Pattern to apply progressively
     * @param position Pattern application position
     * @param direction Pattern orientation
     * @param duration Time in seconds for tear to complete
     * @param scale Pattern scale multiplier
     */
    void StartProgressiveTear(
        std::shared_ptr<ClothTearPattern> pattern,
        const Vec3& position,
        const Vec3& direction,
        float duration,
        float scale = 1.0f
    );

    /**
     * @brief Start a progressive tear by pattern name
     * @param patternName Name of pattern in library
     * @param position Pattern application position
     * @param direction Pattern orientation
     * @param duration Time in seconds for tear to complete
     * @param scale Pattern scale multiplier
     */
    void StartProgressiveTear(
        const std::string& patternName,
        const Vec3& position,
        const Vec3& direction,
        float duration,
        float scale = 1.0f
    );

    /**
     * @brief Update progressive tears (called automatically in Update)
     * @param deltaTime Time step
     */
    void UpdateProgressiveTears(float deltaTime);

    // Rendering Configuration
    void SetTwoSidedRendering(bool enabled) { m_TwoSidedRendering = enabled; }
    bool GetTwoSidedRendering() const { return m_TwoSidedRendering; }

    void SetEnableSubsurface(bool enabled) { m_EnableSubsurface = enabled; }
    bool GetEnableSubsurface() const { return m_EnableSubsurface; }

    void SetTranslucency(float translucency) { m_Translucency = translucency; }
    float GetTranslucency() const { return m_Translucency; }

    void SetWrinkleScale(float scale) { m_WrinkleScale = scale; }
    float GetWrinkleScale() const { return m_WrinkleScale; }

    void SetEnableWrinkleDetail(bool enabled) { m_EnableWrinkleDetail = enabled; }
    bool GetEnableWrinkleDetail() const { return m_EnableWrinkleDetail; }

    /**
     * @brief Get pattern library instance
     * @return Reference to global pattern library
     */
    static ClothTearPatternLibrary& GetPatternLibrary();
    
    // Soft Body Collision
    /**
     * @brief Register soft body for collision
     * @param softBody Soft body to collide with
     */
    void RegisterSoftBodyCollision(class PhysXSoftBody* softBody);
    
    /**
     * @brief Unregister soft body collision
     * @param softBody Soft body to stop colliding with
     */
    void UnregisterSoftBodyCollision(class PhysXSoftBody* softBody);
    
    /**
     * @brief Clear all registered soft bodies
     */
    void ClearSoftBodyCollisions();

private:
    // Allow AsyncClothFactory to access internals for finalization
    friend class AsyncClothFactory;
    PhysXBackend* m_Backend;
    physx::PxCloth* m_Cloth;
    physx::PxClothFabric* m_Fabric;
    
    std::vector<Vec3> m_ParticlePositions;
    std::vector<Vec3> m_ParticleNormals;
    std::vector<int> m_TriangleIndices;
    
    int m_ParticleCount;
    int m_TriangleCount;
    bool m_Enabled;
    
    // Rendering Properties
    bool m_TwoSidedRendering = true;
    bool m_EnableSubsurface = true;
    float m_Translucency = 0.5f;
    float m_WrinkleScale = 1.0f;
    bool m_EnableWrinkleDetail = false;
    bool m_Tearable;
    float m_MaxStretchRatio;
    
    Vec3 m_WindVelocity;
    float m_StretchStiffness;
    float m_BendStiffness;
    float m_ShearStiffness;
    float m_Damping;

    // Collision settings
    bool m_EnableSceneCollision;
    bool m_EnableSelfCollision;
    float m_SelfCollisionDistance;
    float m_SelfCollisionStiffness;
    bool m_EnableTwoWayCoupling;
    float m_CollisionMassScale;

    // Tearing state
    struct TearInfo {
        int particleIndex;
        float stretchRatio;
        Vec3 position;
    };
    
    std::vector<TearInfo> m_TearCandidates;
    std::vector<int> m_TornParticles;    // Tearing state
    bool m_Tearable;
    float m_MaxStretchRatio;
    int m_MaxParticles; // Capacity for tearing
    int m_TearCount;
    float m_LastTearTime;
    
    // Spatial Grid
    std::unique_ptr<SpatialGrid<int>> m_SpatialGrid;

    // Tear callback
    TearCallback m_TearCallback;
    
    // Mesh synchronization
    std::unique_ptr<ClothMeshSynchronizer> m_MeshSynchronizer;

    std::shared_ptr<ClothLOD> m_LODLayer;
    ClothLODConfig m_LODConfig;
    int m_CurrentLODLevel;
    int m_UpdateCounter;
    int m_UpdateFrequency;

    // Frozen state
    bool m_IsFrozen;
    std::vector<Vec3> m_FrozenPositions;  // Saved state when frozen
    std::vector<Vec3> m_FrozenNormals;

    // Progressive tear state
    struct ProgressiveTear {
        std::shared_ptr<ClothTearPattern> pattern;
        Vec3 position;
        Vec3 direction;
        float scale;
        float progress;  // 0.0 to 1.0
        float duration;
        float elapsed;
    };
    std::vector<ProgressiveTear> m_ProgressiveTears;

    // Helper methods
    void CreateClothFabric(const ClothDesc& desc);
    void CreateClothActor(const ClothDesc& desc);
    void SetupConstraints();
    void UpdateParticleData();
    void RecalculateNormals();
    
    /**
     * @brief Update collision shapes sent to PhysX based on cloth bounds
     */
    void UpdateCollisionShapes();
    
    /**
     * @brief Get world bounds of the cloth
     */
    void GetWorldBounds(Vec3& outMin, Vec3& outMax) const;

    /**
     * @brief Update mesh from simplified physics data
     * @param mesh Render mesh to update
     * @param mapping Mapping from mesh vertices to physics particles
     */
    void UpdateProxyMesh(Mesh* mesh, const std::vector<int>& mapping);

    // Spatial Partitioning for efficient collision
    struct InternalCollisionShape {
        int id;
        int type; // 0 = sphere, 1 = capsule
        Vec3 pos0;
        Vec3 pos1; // Only for capsule
        float radius;
    };
    
    // We store indices to the shapes vector in the grid
    std::vector<InternalCollisionShape> m_CollisionShapesList;
    
    // Soft body collision
    std::vector<class PhysXSoftBody*> m_RegisteredSoftBodies;
    
    // Tearing methods
    void DetectTears(float deltaTime);
    void ProcessTear(const TearInfo& tear);
    bool CanTear() const;
    int FindNearestParticle(const Vec3& position) const;
    
    // LOD mesh recreation helpers
    /**
     * @brief Recreate cloth with simplified mesh from LOD level
     * @param level LOD level with mesh data
     * @return True if successful
     */
    bool RecreateClothWithLOD(const ClothLODLevel* level);
    
    /**
     * @brief Transfer particle state from current mesh to LOD mesh
     * @param level Target LOD level
     * @param outParticles Output particle data for new cloth
     */
    void TransferParticleState(
        const ClothLODLevel* level,
        std::vector<physx::PxClothParticle>& outParticles
    );
    
    // Mesh splitting helper
    std::shared_ptr<PhysXCloth> CreateFromSplit(
        const std::vector<Vec3>& positions,
        const std::vector<int>& indices
    );
};

#endif // USE_PHYSX
