#pragma once

#include "IPhysicsCloth.h"
#include "Math/Vec3.h"
#include <memory>
#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>

#ifdef USE_PHYSX

namespace physx {
    class PxClothFabric;
    class PxClothParticle;
}

class PhysXBackend;
class PhysXCloth;

/**
 * @brief Async cloth creation factory using thread pool
 * 
 * Performs expensive fabric cooking on worker threads to avoid
 * blocking the main thread during cloth creation.
 */
class AsyncClothFactory {
public:
    /**
     * @brief Job state
     */
    enum class JobState {
        Pending,      // Waiting in queue
        Processing,   // Being cooked on worker thread
        Ready,        // Cooked, ready for main thread finalization
        Completed,    // Finalized and callback invoked
        Failed        // Error occurred
    };

    /**
     * @brief Cloth creation job data
     */
    struct ClothCreationJob {
        int jobID;
        JobState state;
        
        // Input data (copied to avoid race conditions)
        PhysXBackend* backend;
        ClothDesc desc;
        std::vector<Vec3> particlePositions;
        std::vector<int> triangleIndices;
        
        // Cooked data (filled by worker thread)
        physx::PxClothFabric* cookedFabric;
        std::vector<physx::PxClothParticle> particles;
        
        // Result (filled by main thread)
        std::shared_ptr<PhysXCloth> cloth;
        
        // Callbacks
        std::function<void(std::shared_ptr<PhysXCloth>)> onComplete;
        std::function<void(const std::string&)> onError;
        
        // Error info
        std::string errorMessage;
        
        ClothCreationJob()
            : jobID(-1)
            , state(JobState::Pending)
            , backend(nullptr)
            , cookedFabric(nullptr)
        {}
    };

    /**
     * @brief Get singleton instance
     */
    static AsyncClothFactory& GetInstance();

    /**
     * @brief Create cloth asynchronously
     * @param backend PhysX backend
     * @param desc Cloth descriptor
     * @param onComplete Callback when cloth is ready (main thread)
     * @param onError Callback on error (main thread)
     * @return Job ID for tracking
     */
    int CreateClothAsync(
        PhysXBackend* backend,
        const ClothDesc& desc,
        std::function<void(std::shared_ptr<PhysXCloth>)> onComplete,
        std::function<void(const std::string&)> onError = nullptr
    );

    /**
     * @brief Cancel pending job
     * @param jobID Job ID to cancel
     * @return True if cancelled, false if already processing/completed
     */
    bool CancelJob(int jobID);

    /**
     * @brief Process completed jobs (call from main thread each frame)
     */
    void ProcessCompletedJobs();

    /**
     * @brief Get job state
     */
    JobState GetJobState(int jobID) const;

    /**
     * @brief Set number of worker threads
     */
    void SetWorkerThreadCount(int count);

    /**
     * @brief Get number of worker threads
     */
    int GetWorkerThreadCount() const { return static_cast<int>(m_Workers.size()); }

    /**
     * @brief Get number of pending jobs
     */
    int GetPendingJobCount() const;

    /**
     * @brief Get number of processing jobs
     */
    int GetProcessingJobCount() const;

    /**
     * @brief Shutdown factory and wait for all jobs to complete
     */
    void Shutdown();

private:
    AsyncClothFactory();
    ~AsyncClothFactory();

    // Prevent copying
    AsyncClothFactory(const AsyncClothFactory&) = delete;
    AsyncClothFactory& operator=(const AsyncClothFactory&) = delete;

    /**
     * @brief Worker thread function
     */
    void WorkerThreadFunc();

    /**
     * @brief Cook fabric on worker thread
     */
    void CookFabric(ClothCreationJob& job);

    /**
     * @brief Finalize cloth on main thread
     */
    void FinalizeCloth(ClothCreationJob& job);

    // Thread pool
    std::vector<std::thread> m_Workers;
    std::atomic<bool> m_Shutdown;

    // Job queues
    std::queue<std::shared_ptr<ClothCreationJob>> m_PendingJobs;
    std::queue<std::shared_ptr<ClothCreationJob>> m_ReadyJobs;

    // Job tracking
    std::unordered_map<int, std::shared_ptr<ClothCreationJob>> m_AllJobs;
    std::atomic<int> m_NextJobID;

    // Synchronization
    mutable std::mutex m_PendingMutex;
    mutable std::mutex m_ReadyMutex;
    mutable std::mutex m_JobsMutex;
    std::condition_variable m_WorkAvailable;
};

#endif // USE_PHYSX
