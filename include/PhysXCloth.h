#pragma once

#include "IPhysicsCloth.h"
#include "Math/Vec3.h"
#include "ClothLOD.h"
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

    // Rendering support
    /**
     * @brief Update mesh data from cloth simulation
     * @param mesh Mesh to update with current particle positions/normals
     */
    void UpdateMeshData(Mesh* mesh);

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
    bool m_Tearable;
    float m_MaxStretchRatio;
    
    Vec3 m_WindVelocity;
    float m_StretchStiffness;
    float m_BendStiffness;
    float m_ShearStiffness;
    float m_Damping;

    // Tearing state
    struct TearInfo {
        int particleIndex;
        float stretchRatio;
        Vec3 position;
    };
    
    std::vector<TearInfo> m_TearCandidates;
    std::vector<int> m_TornParticles;  // Indices of particles that have been torn
    int m_MaxParticles;
    int m_TearCount;
    float m_LastTearTime;

    // Tear callback
    TearCallback m_TearCallback;

    // LOD system
    ClothLODConfig m_LODConfig;
    int m_CurrentLOD;
    bool m_IsFrozen;
    std::vector<Vec3> m_FrozenPositions;  // Saved state when frozen
    std::vector<Vec3> m_FrozenNormals;

    // Helper methods
    void CreateClothFabric(const ClothDesc& desc);
    void SetupConstraints();
    void UpdateParticleData();
    void RecalculateNormals();
    
    // Tearing methods
    void DetectTears(float deltaTime);
    void ProcessTear(const TearInfo& tear);
    bool CanTear() const;
    int FindNearestParticle(const Vec3& position) const;
    
    // Mesh splitting helper
    std::shared_ptr<PhysXCloth> CreateFromSplit(
        const std::vector<Vec3>& positions,
        const std::vector<int>& indices
    );
};

#endif // USE_PHYSX
