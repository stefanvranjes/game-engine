#pragma once

#include <vector>
#include <memory>
#include <cstdint>

#ifdef USE_PHYSX
#include <PxPhysicsAPI.h>
#ifdef HAS_CUDA_TOOLKIT
#include <cuda_runtime.h>
#endif
#endif

class PhysXSoftBody;

/**
 * @brief Manages batch GPU operations for multiple soft bodies
 * 
 * Reduces CPU-GPU synchronization overhead by batching data transfers
 * using PhysX's copySoftBodyData/applySoftBodyData API and CUDA streams
 * for concurrent operations.
 */
class GpuBatchManager {
public:
    /**
     * @brief Priority levels for soft body processing
     */
    enum class Priority {
        CRITICAL = 4,  // Player character, immediate vicinity
        HIGH = 3,      // Visible, interactive objects
        MEDIUM = 2,    // Nearby, background objects
        LOW = 1,       // Far, decorative objects
        DEFERRED = 0   // Very far, can skip frames
    };
    
    /**
     * @brief GPU distribution strategy for multi-GPU systems
     */
    enum class GpuDistribution {
        ROUND_ROBIN,      // Distribute evenly across GPUs
        LOAD_BALANCED,    // Distribute based on GPU load
        PRIORITY_BASED,   // High priority on fastest GPU
        MANUAL,           // User-specified GPU per soft body
        AUTO              // Automatic selection based on workload and GPU capabilities
    };
    
    GpuBatchManager();
    ~GpuBatchManager();
    
    /**
     * @brief Initialize batch manager with CUDA streams
     * @param streamCount Number of CUDA streams for concurrent operations
     */
    void Initialize(size_t streamCount = 4);
    
    /**
     * @brief Shutdown and cleanup resources
     */
    void Shutdown();
    
    /**
     * @brief Add soft body to batch processing queue
     * @param softBody Soft body to add
     */
    void AddSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Remove soft body from batch processing
     * @param softBody Soft body to remove
     */
    void RemoveSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Set soft body priority
     * @param softBody Soft body to prioritize
     * @param priority Priority level
     */
    void SetSoftBodyPriority(PhysXSoftBody* softBody, Priority priority);
    
    /**
     * @brief Update priorities based on camera position
     * @param cameraPosition Camera position for distance calculation
     */
    void UpdatePriorities(const Vec3& cameraPosition);
    
    /**
     * @brief Enable/disable automatic priority updates
     * @param enable True to automatically update priorities each frame
     */
    void EnableAutoPriority(bool enable);
    
    /**
     * @brief Enable/disable adaptive priority adjustments
     * @param enable True to adapt priorities based on performance
     * @param targetFrameTimeMs Target frame time in milliseconds (default: 16.67ms for 60 FPS)
     */
    void EnableAdaptivePriority(bool enable, float targetFrameTimeMs = 16.67f);
    
    /**
     * @brief Update adaptive priority based on last frame performance
     * @param lastFrameTimeMs Last frame time in milliseconds
     */
    void UpdateAdaptivePriority(float lastFrameTimeMs);
    
    /**
     * @brief Enable/disable predictive priority
     * @param enable True to predict future visibility
     * @param predictionTimeSeconds How far ahead to predict (default: 0.5s)
     */
    void EnablePredictivePriority(bool enable, float predictionTimeSeconds = 0.5f);
    
    /**
     * @brief Update predictive priority based on camera movement
     * @param currentCameraPosition Current camera position
     * @param cameraVelocity Camera velocity vector
     */
    void UpdatePredictivePriority(const Vec3& currentCameraPosition, const Vec3& cameraVelocity);
    
    /**
     * @brief Update priority decay (reduces temporary boosts over time)
     * @param deltaTime Time since last update in seconds
     */
    void UpdatePriorityDecay(float deltaTime);
    
    /**
     * @brief Apply temporary priority boost to a soft body
     * @param softBody Soft body to boost
     * @param boostAmount Priority boost amount (0-100)
     * @param decayRate How fast boost decays (points per second, default: 10)
     */
    void ApplyTemporaryBoost(PhysXSoftBody* softBody, float boostAmount, float decayRate = 10.0f);
    
    /**
     * @brief Enable multi-GPU support
     * @param enable True to use multiple GPUs
     * @param gpuIds List of GPU IDs to use (empty = use all available)
     * @param distribution Distribution strategy
     */
    void EnableMultiGpu(bool enable, 
                        const std::vector<int>& gpuIds = {}, 
                        GpuDistribution distribution = GpuDistribution::ROUND_ROBIN);
    
    /**
     * @brief Enable dynamic migration between GPUs
     * @param enable True to enable automatic migration
     * @param loadThreshold Load imbalance threshold (default: 0.3 = 30%)
     * @param migrationInterval Frames between migration checks (default: 60)
     */
    void EnableDynamicMigration(bool enable, float loadThreshold = 0.3f, size_t migrationInterval = 60);
    
    /**
     * @brief Manually migrate soft body to specific GPU
     * @param softBody Soft body to migrate
     * @param targetGpu Target GPU ID
     */
    void MigrateSoftBody(PhysXSoftBody* softBody, int targetGpu);
    
    /**
     * @brief Manually assign soft body to specific GPU
     * @param softBody Soft body to assign
     * @param gpuId GPU device ID
     */
    void AssignToGpu(PhysXSoftBody* softBody, int gpuId);
    
    /**
     * @brief Get multi-GPU statistics
     */
    struct MultiGpuStats {
        size_t gpuCount;
        std::vector<size_t> batchesPerGpu;
        std::vector<float> loadPerGpu;
        std::vector<size_t> memoryUsedPerGpu;
        bool peerAccessEnabled;
        
        // Dynamic migration stats
        bool migrationEnabled;
        size_t totalMigrations;
        size_t migrationsThisFrame;
        float maxLoadImbalance;
    };
    MultiGpuStats GetMultiGpuStatistics() const;
    
    /**
     * @brief Execute batched GPU data copy operations
     * @param scene PhysX scene containing soft bodies
     * 
     * Uses PhysX copySoftBodyData to read data from GPU buffers
     * in batches, reducing synchronization overhead.
     */
    void BatchCopyData(physx::PxScene* scene);
    
    /**
     * @brief Execute batched GPU data apply operations
     * @param scene PhysX scene containing soft bodies
     * 
     * Uses PhysX applySoftBodyData to write data to GPU buffers
     * in batches, reducing synchronization overhead.
     */
    void BatchApplyData(physx::PxScene* scene);
    
    /**
     * @brief Synchronize all pending GPU operations
     * 
     * Blocks until all CUDA streams complete their work.
     */
    void Synchronize();
    
    /**
     * @brief Get batch processing statistics
     */
    struct BatchStats {
        size_t softBodyCount;           // Total soft bodies managed
        size_t batchesProcessed;        // Total batches executed
        float avgBatchSizeKB;           // Average batch size in KB
        float lastBatchTimeMs;          // Last batch operation time
        size_t streamCount;             // Number of CUDA streams
        float totalDataTransferredMB;   // Total data transferred
        
        // Adaptive priority stats
        float currentFrameBudgetMs;     // Current frame budget for batching
        size_t lowPrioritySkipRate;     // How often LOW priority is skipped (1 = every frame, 2 = every other)
        size_t deferredSkipRate;        // How often DEFERRED priority is skipped
        bool adaptiveEnabled;           // Whether adaptive priority is active
        
        // Predictive priority stats
        bool predictiveEnabled;         // Whether predictive priority is active
        float predictionTimeSeconds;    // How far ahead we predict
        size_t predictiveBoosted;       // Number of soft bodies boosted by prediction
    };
    BatchStats GetStatistics() const;
    
    /**
     * @brief Clear all statistics
     */
    void ResetStatistics();
    
    /**
     * @brief Check if batch manager is initialized
     */
    bool IsInitialized() const;
    
private:
    struct SoftBodyEntry {
        PhysXSoftBody* softBody;
        Priority priority;
        float priorityScore;        // Calculated priority (0-100)
        uint64_t lastProcessedFrame;
        
        // Priority decay
        float temporaryBoost;       // Temporary priority boost (decays over time)
        float boostDecayRate;       // How fast boost decays (points per second)
        float timeSinceBoost;       // Time since last boost applied
        
        // Multi-GPU
        int assignedGpu;            // GPU ID (-1 = auto-assign)
    };
    
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
    
    /**
     * @brief Sort soft bodies by priority for batching
     */
    void SortByPriority();
    
    /**
     * @brief Calculate dynamic priority score
     */
    float CalculatePriorityScore(const SoftBodyEntry& entry, const Vec3& cameraPosition);
    
    /**
     * @brief Select GPU for next batch
     */
    int SelectGpuForBatch(const SoftBodyEntry& entry);
    
    /**
     * @brief Calculate workload complexity for a soft body
     * @return Complexity score (0-100)
     */
    float CalculateWorkloadComplexity(const SoftBodyEntry& entry);
    
    /**
     * @brief Select optimal GPU based on workload and GPU capabilities
     * @param complexity Workload complexity score
     * @param priority Soft body priority
     * @return GPU index
     */
    int SelectOptimalGpu(float complexity, Priority priority);
    
    /**
     * @brief Enable peer-to-peer access between GPUs
     */
    void EnablePeerAccess();
    
    /**
     * @brief Process batches on multiple GPUs
     */
    void ProcessMultiGpuBatches(physx::PxScene* scene);
    
    /**
     * @brief Process batches on single GPU
     */
    void ProcessSingleGpuBatches(physx::PxScene* scene);
    
    /**
     * @brief Check for load imbalance and perform migration if needed
     */
    void CheckAndPerformMigration();
    
    /**
     * @brief Calculate current load imbalance across GPUs
     * @return Maximum load difference (0-1)
     */
    float CalculateLoadImbalance();
    
    // Batch configuration
    static constexpr size_t MAX_BATCH_SIZE = 64;  // Soft bodies per batch
    static constexpr size_t MIN_BATCH_SIZE = 4;   // Minimum for batching
    static constexpr size_t DEFAULT_STREAM_COUNT = 4;
};
