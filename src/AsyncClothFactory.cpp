#include "AsyncClothFactory.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include "PhysXCloth.h"
#include <PxPhysicsAPI.h>
#include <extensions/PxClothFabricCooker.h>
#include <iostream>
#include <algorithm>

using namespace physx;

AsyncClothFactory::AsyncClothFactory()
    : m_Shutdown(false)
    , m_NextJobID(1)
{
    // Start with 2 worker threads by default
    SetWorkerThreadCount(2);
    
    std::cout << "AsyncClothFactory initialized with " 
              << m_Workers.size() << " worker threads" << std::endl;
}

AsyncClothFactory::~AsyncClothFactory() {
    Shutdown();
}

AsyncClothFactory& AsyncClothFactory::GetInstance() {
    static AsyncClothFactory instance;
    return instance;
}

int AsyncClothFactory::CreateClothAsync(
    PhysXBackend* backend,
    const ClothDesc& desc,
    std::function<void(std::shared_ptr<PhysXCloth>)> onComplete,
    std::function<void(const std::string&)> onError)
{
    if (!backend) {
        if (onError) {
            onError("Invalid backend");
        }
        return -1;
    }

    // Create job
    auto job = std::make_shared<ClothCreationJob>();
    job->jobID = m_NextJobID++;
    job->state = JobState::Pending;
    job->backend = backend;
    job->desc = desc;
    job->onComplete = onComplete;
    job->onError = onError;

    // Copy mesh data to avoid race conditions
    job->particlePositions.resize(desc.particleCount);
    for (int i = 0; i < desc.particleCount; ++i) {
        job->particlePositions[i] = desc.particlePositions[i];
    }

    job->triangleIndices.resize(desc.triangleCount * 3);
    for (int i = 0; i < desc.triangleCount * 3; ++i) {
        job->triangleIndices[i] = desc.triangleIndices[i];
    }

    // Add to tracking
    {
        std::lock_guard<std::mutex> lock(m_JobsMutex);
        m_AllJobs[job->jobID] = job;
    }

    // Add to pending queue
    {
        std::lock_guard<std::mutex> lock(m_PendingMutex);
        m_PendingJobs.push(job);
    }

    // Wake up a worker thread
    m_WorkAvailable.notify_one();

    std::cout << "Queued async cloth creation job " << job->jobID << std::endl;

    return job->jobID;
}

bool AsyncClothFactory::CancelJob(int jobID) {
    std::lock_guard<std::mutex> lock(m_JobsMutex);
    
    auto it = m_AllJobs.find(jobID);
    if (it == m_AllJobs.end()) {
        return false;
    }

    auto job = it->second;
    
    // Can only cancel if still pending
    if (job->state == JobState::Pending) {
        job->state = JobState::Failed;
        job->errorMessage = "Job cancelled";
        m_AllJobs.erase(it);
        return true;
    }

    return false;
}

void AsyncClothFactory::ProcessCompletedJobs() {
    std::vector<std::shared_ptr<ClothCreationJob>> readyJobs;

    // Get all ready jobs
    {
        std::lock_guard<std::mutex> lock(m_ReadyMutex);
        while (!m_ReadyJobs.empty()) {
            readyJobs.push_back(m_ReadyJobs.front());
            m_ReadyJobs.pop();
        }
    }

    // Process on main thread
    for (auto& job : readyJobs) {
        if (job->state == JobState::Ready) {
            FinalizeCloth(*job);
        }

        // Invoke callbacks
        if (job->state == JobState::Completed && job->onComplete) {
            job->onComplete(job->cloth);
        }
        else if (job->state == JobState::Failed && job->onError) {
            job->onError(job->errorMessage);
        }

        // Remove from tracking
        {
            std::lock_guard<std::mutex> lock(m_JobsMutex);
            m_AllJobs.erase(job->jobID);
        }
    }
}

AsyncClothFactory::JobState AsyncClothFactory::GetJobState(int jobID) const {
    std::lock_guard<std::mutex> lock(m_JobsMutex);
    
    auto it = m_AllJobs.find(jobID);
    if (it == m_AllJobs.end()) {
        return JobState::Failed;
    }

    return it->second->state;
}

void AsyncClothFactory::SetWorkerThreadCount(int count) {
    if (count < 1) count = 1;
    if (count > 8) count = 8;

    // Shutdown existing workers
    if (!m_Workers.empty()) {
        m_Shutdown = true;
        m_WorkAvailable.notify_all();
        
        for (auto& worker : m_Workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        m_Workers.clear();
        m_Shutdown = false;
    }

    // Start new workers
    for (int i = 0; i < count; ++i) {
        m_Workers.emplace_back(&AsyncClothFactory::WorkerThreadFunc, this);
    }

    std::cout << "AsyncClothFactory: Set worker thread count to " << count << std::endl;
}

int AsyncClothFactory::GetPendingJobCount() const {
    std::lock_guard<std::mutex> lock(m_PendingMutex);
    return static_cast<int>(m_PendingJobs.size());
}

int AsyncClothFactory::GetProcessingJobCount() const {
    std::lock_guard<std::mutex> lock(m_JobsMutex);
    
    int count = 0;
    for (const auto& pair : m_AllJobs) {
        if (pair.second->state == JobState::Processing) {
            count++;
        }
    }
    return count;
}

void AsyncClothFactory::Shutdown() {
    if (m_Shutdown) return;

    std::cout << "AsyncClothFactory: Shutting down..." << std::endl;

    m_Shutdown = true;
    m_WorkAvailable.notify_all();

    for (auto& worker : m_Workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    m_Workers.clear();

    std::cout << "AsyncClothFactory: Shutdown complete" << std::endl;
}

void AsyncClothFactory::WorkerThreadFunc() {
    while (!m_Shutdown) {
        std::shared_ptr<ClothCreationJob> job;

        // Get next job
        {
            std::unique_lock<std::mutex> lock(m_PendingMutex);
            m_WorkAvailable.wait(lock, [this] {
                return m_Shutdown || !m_PendingJobs.empty();
            });

            if (m_Shutdown) break;

            if (!m_PendingJobs.empty()) {
                job = m_PendingJobs.front();
                m_PendingJobs.pop();
                job->state = JobState::Processing;
            }
        }

        if (!job) continue;

        std::cout << "Worker thread processing job " << job->jobID << std::endl;

        // Cook fabric (expensive operation)
        CookFabric(*job);

        // Move to ready queue for main thread finalization
        {
            std::lock_guard<std::mutex> lock(m_ReadyMutex);
            m_ReadyJobs.push(job);
        }

        std::cout << "Worker thread completed job " << job->jobID 
                  << " (state: " << (job->state == JobState::Ready ? "Ready" : "Failed") << ")" 
                  << std::endl;
    }
}

void AsyncClothFactory::CookFabric(ClothCreationJob& job) {
    try {
        if (!job.backend || !job.backend->GetPhysics()) {
            job.state = JobState::Failed;
            job.errorMessage = "Invalid backend or physics";
            return;
        }

        PxPhysics* physics = job.backend->GetPhysics();

        // Prepare mesh descriptor
        PxClothMeshDesc meshDesc;
        meshDesc.setToDefault();

        // Set vertices
        std::vector<PxVec3> vertices(job.desc.particleCount);
        for (int i = 0; i < job.desc.particleCount; ++i) {
            vertices[i] = PxVec3(
                job.particlePositions[i].x,
                job.particlePositions[i].y,
                job.particlePositions[i].z
            );
        }
        meshDesc.points.data = vertices.data();
        meshDesc.points.count = job.desc.particleCount;
        meshDesc.points.stride = sizeof(PxVec3);

        // Set triangles
        meshDesc.triangles.data = job.triangleIndices.data();
        meshDesc.triangles.count = job.desc.triangleCount;
        meshDesc.triangles.stride = sizeof(int) * 3;

        // Cook fabric (CPU-intensive operation)
        PxClothFabricCooker cooker(meshDesc, PxVec3(0, -1, 0));
        job.cookedFabric = cooker.getClothFabric(*physics);

        if (!job.cookedFabric) {
            job.state = JobState::Failed;
            job.errorMessage = "Fabric cooking failed";
            return;
        }

        // Prepare particles
        job.particles.resize(job.desc.particleCount);
        for (int i = 0; i < job.desc.particleCount; ++i) {
            job.particles[i].pos = PxVec3(
                job.particlePositions[i].x,
                job.particlePositions[i].y,
                job.particlePositions[i].z
            );
            job.particles[i].invWeight = 1.0f / job.desc.particleMass;
        }

        job.state = JobState::Ready;
    }
    catch (const std::exception& e) {
        job.state = JobState::Failed;
        job.errorMessage = std::string("Exception during fabric cooking: ") + e.what();
    }
    catch (...) {
        job.state = JobState::Failed;
        job.errorMessage = "Unknown exception during fabric cooking";
    }
}

void AsyncClothFactory::FinalizeCloth(ClothCreationJob& job) {
    try {
        if (!job.backend || !job.backend->GetPhysics() || !job.backend->GetScene()) {
            job.state = JobState::Failed;
            job.errorMessage = "Invalid backend, physics, or scene";
            return;
        }

        if (!job.cookedFabric) {
            job.state = JobState::Failed;
            job.errorMessage = "No cooked fabric available";
            return;
        }

        // Create cloth object
        auto cloth = std::make_shared<PhysXCloth>(job.backend);

        // Create PhysX cloth actor (must be on main thread)
        PxTransform clothTransform(PxVec3(0, 0, 0));
        PxCloth* pxCloth = job.backend->GetPhysics()->createCloth(
            clothTransform,
            *job.cookedFabric,
            job.particles.data(),
            PxClothFlags()
        );

        if (!pxCloth) {
            job.state = JobState::Failed;
            job.errorMessage = "Failed to create PhysX cloth actor";
            return;
        }

        // Add to scene
        job.backend->GetScene()->addActor(*pxCloth);

        // Set internal state (using friend class access)
        cloth->m_Cloth = pxCloth;
        cloth->m_Fabric = job.cookedFabric;
        cloth->m_ParticleCount = job.desc.particleCount;
        cloth->m_TriangleCount = job.desc.triangleCount;

        // Store mesh data
        cloth->m_ParticlePositions.resize(job.desc.particleCount);
        cloth->m_ParticleNormals.resize(job.desc.particleCount);
        cloth->m_TriangleIndices.resize(job.desc.triangleCount * 3);

        for (int i = 0; i < job.desc.particleCount; ++i) {
            cloth->m_ParticlePositions[i] = job.particlePositions[i];
        }

        for (int i = 0; i < job.desc.triangleCount * 3; ++i) {
            cloth->m_TriangleIndices[i] = job.triangleIndices[i];
        }

        // Setup constraints and parameters
        cloth->SetupConstraints();

        // Set gravity
        pxCloth->setExternalAcceleration(PxVec3(
            job.desc.gravity.x,
            job.desc.gravity.y,
            job.desc.gravity.z
        ));

        // Calculate initial normals
        cloth->RecalculateNormals();

        // Set default properties
        cloth->m_Enabled = true;
        cloth->m_Tearable = false;
        cloth->m_MaxStretchRatio = 1.5f;
        cloth->m_WindVelocity = Vec3(0, 0, 0);
        cloth->m_StretchStiffness = 0.8f;
        cloth->m_BendStiffness = 0.5f;
        cloth->m_ShearStiffness = 0.6f;
        cloth->m_Damping = 0.2f;

        job.cloth = cloth;
        job.state = JobState::Completed;

        std::cout << "Finalized cloth creation for job " << job.jobID 
                  << " with " << job.desc.particleCount << " particles" << std::endl;
    }
    catch (const std::exception& e) {
        job.state = JobState::Failed;
        job.errorMessage = std::string("Exception during finalization: ") + e.what();
    }
    catch (...) {
        job.state = JobState::Failed;
        job.errorMessage = "Unknown exception during finalization";
    }
}

#endif // USE_PHYSX
