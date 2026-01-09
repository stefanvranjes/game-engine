#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

class Mesh;
class SpatialHashGrid;

/**
 * @brief Synchronization mode for cloth-to-mesh updates
 */
enum class ClothSyncMode {
    Full,        // Update all vertices every frame
    Partial,     // Update only dirty regions
    Progressive  // Spread updates across multiple frames
};

/**
 * @brief Vertex attribute flags for synchronization
 */
enum class VertexAttribute {
    Position = 1 << 0,
    Normal   = 1 << 1,
    Tangent  = 1 << 2,
    UV       = 1 << 3,
    All      = Position | Normal | Tangent | UV
};

inline VertexAttribute operator|(VertexAttribute a, VertexAttribute b) {
    return static_cast<VertexAttribute>(static_cast<int>(a) | static_cast<int>(b));
}

inline bool operator&(VertexAttribute a, VertexAttribute b) {
    return (static_cast<int>(a) & static_cast<int>(b)) != 0;
}

/**
 * @brief Configuration for cloth-mesh synchronization
 */
struct ClothSyncConfig {
    ClothSyncMode mode = ClothSyncMode::Full;
    VertexAttribute attributes = VertexAttribute::All;
    
    // Progressive mode settings
    int maxVerticesPerFrame = 1000;  // Max vertices to update per frame
    
    // Partial mode settings
    float dirtyRegionRadius = 0.5f;  // Radius around changed particles
    float changeThreshold = 0.01f;   // Min movement to mark as dirty (meters)
    
    // LOD settings
    bool enableInterpolation = true;  // Smooth LOD transitions
    float interpolationDuration = 0.2f; // Seconds
    
    // Threading
    bool enableAsync = false;  // Use async synchronization
    int asyncThreadCount = 2;
};

/**
 * @brief Mapping between cloth particles and mesh vertices
 */
struct VertexMapping {
    // For LOD 0 (1:1 mapping): particleIndices[i] = i
    // For LOD > 0: particleIndices[i] = closest particle for vertex i
    std::vector<int> particleIndices;
    
    // Weights for interpolation (for multi-particle influence)
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<int>> influenceParticles;
    
    bool isOneToOne = true;  // True if 1:1 mapping (LOD 0)
};

/**
 * @brief Tracks dirty regions for partial updates
 */
class DirtyRegionTracker {
public:
    DirtyRegionTracker();
    ~DirtyRegionTracker();
    
    /**
     * @brief Initialize with particle positions
     */
    void Initialize(const std::vector<Vec3>& positions, float cellSize);
    
    /**
     * @brief Mark particles as dirty based on position changes
     */
    void UpdateDirtyParticles(
        const std::vector<Vec3>& oldPositions,
        const std::vector<Vec3>& newPositions,
        float threshold
    );
    
    /**
     * @brief Get all dirty particles
     */
    const std::unordered_set<int>& GetDirtyParticles() const { return m_DirtyParticles; }
    
    /**
     * @brief Get vertices affected by dirty particles
     */
    std::vector<int> GetAffectedVertices(
        const VertexMapping& mapping,
        float influenceRadius
    ) const;
    
    /**
     * @brief Clear dirty flags
     */
    void Clear();
    
    /**
     * @brief Mark specific particle as dirty
     */
    void MarkDirty(int particleIndex);
    
private:
    std::unordered_set<int> m_DirtyParticles;
    std::unique_ptr<SpatialHashGrid> m_SpatialGrid;
};

/**
 * @brief Handles synchronization between cloth simulation and render mesh
 */
class ClothMeshSynchronizer {
public:
    ClothMeshSynchronizer();
    ~ClothMeshSynchronizer();
    
    /**
     * @brief Initialize synchronizer with mesh and particle data
     */
    void Initialize(
        Mesh* mesh,
        int particleCount,
        const std::vector<int>& triangleIndices
    );
    
    /**
     * @brief Set synchronization configuration
     */
    void SetConfig(const ClothSyncConfig& config) { m_Config = config; }
    
    /**
     * @brief Get current configuration
     */
    const ClothSyncConfig& GetConfig() const { return m_Config; }
    
    /**
     * @brief Set vertex mapping for LOD support
     */
    void SetVertexMapping(const VertexMapping& mapping);
    
    /**
     * @brief Synchronize mesh with cloth particle data
     * @param positions Current particle positions
     * @param normals Current particle normals (optional, can be nullptr)
     * @param deltaTime Time since last update (for progressive mode)
     * @return True if synchronization completed this frame
     */
    bool Synchronize(
        const std::vector<Vec3>& positions,
        const std::vector<Vec3>* normals = nullptr,
        float deltaTime = 0.0f
    );
    
    /**
     * @brief Force full synchronization (ignores mode)
     */
    void ForceFullSync(
        const std::vector<Vec3>& positions,
        const std::vector<Vec3>* normals = nullptr
    );
    
    /**
     * @brief Mark specific particles as dirty for next update
     */
    void MarkParticlesDirty(const std::vector<int>& particleIndices);
    
    /**
     * @brief Start LOD transition with interpolation
     */
    void BeginLODTransition(
        const VertexMapping& newMapping,
        const std::vector<Vec3>& oldPositions,
        const std::vector<Vec3>& newPositions
    );
    
    /**
     * @brief Update LOD transition interpolation
     */
    void UpdateLODTransition(float deltaTime);
    
    /**
     * @brief Check if LOD transition is active
     */
    bool IsLODTransitionActive() const { return m_LODTransitionActive; }
    
    /**
     * @brief Get progress of current operation (0.0 to 1.0)
     */
    float GetProgress() const;
    
    /**
     * @brief Reset synchronizer state
     */
    void Reset();
    
private:
    // Core synchronization methods
    void SynchronizeFull(
        const std::vector<Vec3>& positions,
        const std::vector<Vec3>* normals
    );
    
    void SynchronizePartial(
        const std::vector<Vec3>& positions,
        const std::vector<Vec3>* normals
    );
    
    void SynchronizeProgressive(
        const std::vector<Vec3>& positions,
        const std::vector<Vec3>* normals,
        float deltaTime
    );
    
    // Helper methods
    void UpdateVertexPosition(int vertexIndex, const Vec3& position);
    void UpdateVertexNormal(int vertexIndex, const Vec3& normal);
    void RecalculateNormals(const std::vector<Vec3>& positions);
    
    Vec3 InterpolatePosition(
        int vertexIndex,
        const std::vector<Vec3>& positions
    ) const;
    
    void UpdateMeshBuffers(const std::vector<int>& dirtyVertices = {});
    
    // State
    Mesh* m_Mesh;
    int m_ParticleCount;
    std::vector<int> m_TriangleIndices;
    
    ClothSyncConfig m_Config;
    VertexMapping m_Mapping;
    
    // Dirty tracking
    DirtyRegionTracker m_DirtyTracker;
    std::vector<Vec3> m_PreviousPositions;
    
    // Progressive update state
    int m_ProgressiveVertexIndex;
    std::vector<int> m_ProgressiveUpdateQueue;
    
    // LOD transition state
    bool m_LODTransitionActive;
    float m_LODTransitionProgress;
    VertexMapping m_OldMapping;
    VertexMapping m_NewMapping;
    std::vector<Vec3> m_LODOldPositions;
    std::vector<Vec3> m_LODNewPositions;
    
    // Cached mesh data
    std::vector<Vec3> m_VertexPositions;
    std::vector<Vec3> m_VertexNormals;
    
    // Async support
    struct AsyncTask {
        std::function<void()> task;
        bool completed;
    };
    std::vector<AsyncTask> m_AsyncTasks;
};
