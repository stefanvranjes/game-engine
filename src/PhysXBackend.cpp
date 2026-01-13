#include "PhysXBackend.h"
#include "IPhysicsRigidBody.h"
#include "EngineConfig.h"
#include "IPhysicsCharacterController.h"
#include "IPhysicsCharacterController.h"
#include "IPhysicsSoftBody.h"
#include "IPhysicsCloth.h"
#include "PhysXVehicle.h"
#include "PhysXAggregate.h"

#include "PhysXRigidBody.h" // Needed for casting
#include "PhysXCharacterController.h"
#include "PhysXSoftBody.h"
#include "PhysXVehicle.h"
#include "PhysXFluidVolume.h" // Added

#ifdef USE_PHYSX

#include <PxPhysicsAPI.h>
#ifdef HAS_CUDA_TOOLKIT
#include <cuda_runtime.h>
#endif
#include <iostream>

using namespace physx;

// Define nested classes
class PhysXBackend::PhysXCollisionCallback : public PxSimulationEventCallback {
public:
    PhysXCollisionCallback(PhysXBackend* backend) : m_Backend(backend) {}
    void onConstraintBreak(PxConstraintInfo* constraints, PxU32 count) override {}
    void onWake(PxActor** actors, PxU32 count) override {}
    void onSleep(PxActor** actors, PxU32 count) override {}
    void onContact(const PxContactPairHeader& pairHeader, const PxContactPair* pairs, PxU32 nbPairs) override;
    void onTrigger(PxTriggerPair* pairs, PxU32 count) override;
    void onAdvance(const PxRigidBody* const* bodyBuffer, const PxTransform* poseBuffer, const PxU32 count) override {}
private:
    PhysXBackend* m_Backend;
};

class PhysXBackend::PhysXContactModifyCallback : public PxContactModifyCallback {
public:
    PhysXContactModifyCallback(PhysXBackend* backend) : m_Backend(backend) {}
    void onContactModify(PxContactModifyPair* const pairs, PxU32 count) override;
private:
    PhysXBackend* m_Backend;
};

PhysXBackend::PhysXBackend()
    : m_Foundation(nullptr)
    , m_Physics(nullptr)
    : m_Foundation(nullptr)
    , m_Physics(nullptr)
    , m_ActiveScene(nullptr)
    , m_Dispatcher(nullptr)
    , m_Pvd(nullptr)
    , m_Dispatcher(nullptr)
    , m_Pvd(nullptr)
    , m_CudaContextManager(nullptr)
    , m_DefaultMaterial(nullptr)
    , m_Allocator(nullptr)
    , m_ErrorCallback(nullptr)
    , m_CollisionCallback(nullptr)
    , m_ContactModifyCallback(nullptr)
    , m_Initialized(false)
    , m_DebugDrawEnabled(false)
    , m_Gravity(0, -9.81f, 0)
    , m_GpuMemoryUsageMB(0)
{
}

PhysXBackend::~PhysXBackend() {
    if (m_Initialized) {
        Shutdown();
    }
}

void PhysXBackend::Initialize(const Vec3& gravity) {
    if (m_Initialized) {
        std::cerr << "PhysXBackend already initialized!" << std::endl;
        return;
    }

    std::cout << "Initializing PhysX SDK 5.x..." << std::endl;

    // Create allocator and error callback
    m_Allocator = new PxDefaultAllocator();
    m_ErrorCallback = new PxDefaultErrorCallback();

    // Create foundation
    m_Foundation = PxCreateFoundation(PX_PHYSICS_VERSION, *m_Allocator, *m_ErrorCallback);
    if (!m_Foundation) {
        std::cerr << "Failed to create PhysX foundation!" << std::endl;
        return;
    }

    // Initialize PhysX Visual Debugger (PVD)
    InitializePhysXVisualDebugger();

    // Create physics SDK
    bool recordMemoryAllocations = true;
    m_Physics = PxCreatePhysics(PX_PHYSICS_VERSION, *m_Foundation, PxTolerancesScale(), recordMemoryAllocations, m_Pvd);
    if (!m_Physics) {
        std::cerr << "Failed to create PhysX SDK!" << std::endl;
        return;
    }

    // Create default material
    m_DefaultMaterial = m_Physics->createMaterial(0.5f, 0.5f, 0.6f); // friction, friction, restitution

    // Initialize Vehicle SDK
    if (!PxInitVehicleSDK(*m_Physics)) {
        std::cerr << "Failed to initialize PhysX Vehicle SDK!" << std::endl;
        return;
    }
    PxVehicleSetBasisVectors(PxVec3(0, 1, 0), PxVec3(0, 0, 1));
    PxVehicleSetUpdateMode(PxVehicleUpdateMode::eVELOCITY_CHANGE);

    // Try to create CUDA context manager
    PxCudaContextManagerDesc cudaCtxtDesc;
    m_CudaContextManager = PxCreateCudaContextManager(*m_Foundation, cudaCtxtDesc, PxGetProfilerCallback());
    
    if (m_CudaContextManager) {
        if (!m_CudaContextManager->contextIsValid()) {
            m_CudaContextManager->release();
            m_CudaContextManager = nullptr;
            std::cout << "PhysX: CUDA context invalid. Falling back to CPU." << std::endl;
            std::cout << "WARNING: Soft bodies require GPU and will not be available!" << std::endl;
        } else {
            std::cout << "PhysX: CUDA context created. GPU acceleration enabled." << std::endl;
            
            // Log GPU device properties
            int computeCapability = 0;
            size_t totalMemoryMB = 0;
            size_t freeMemoryMB = 0;
            GetGpuDeviceProperties(computeCapability, totalMemoryMB, freeMemoryMB);
            
            std::cout << "  GPU Compute Capability: " << (computeCapability / 10) << "." << (computeCapability % 10) << std::endl;
            std::cout << "  GPU Total Memory: " << totalMemoryMB << " MB" << std::endl;
            std::cout << "  GPU Free Memory: " << freeMemoryMB << " MB" << std::endl;
            std::cout << "  Soft Body Support: " << (IsGpuSoftBodySupported() ? "YES" : "NO") << std::endl;
        }
    } else {
        std::cout << "PhysX: CUDA context creation failed. Falling back to CPU." << std::endl;
        std::cout << "WARNING: Soft bodies require GPU and will not be available!" << std::endl;
    }

    // Create default scene
    CreateScene("Main", gravity);
    
/*  
    // Legacy single scene creation (moved to CreateSceneInternal)
    
    // Create scene
    PxSceneDesc sceneDesc(m_Physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(gravity.x, gravity.y, gravity.z);
    m_Gravity = gravity;
    ...
*/
    sceneDesc.gravity = PxVec3(gravity.x, gravity.y, gravity.z);
    m_Gravity = gravity;

    // Enable GPU dynamics if CUDA context is available
    if (m_CudaContextManager) {
        sceneDesc.cudaContextManager = m_CudaContextManager;
        sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        sceneDesc.flags |= PxSceneFlag::eENABLE_PCM; // Persistent Contact Manifold (usually good for GPU)
        sceneDesc.flags |= PxSceneFlag::eENABLE_STABILIZATION; // Improve stability for GPU rigid bodies
        sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
        sceneDesc.gpuMaxNumPartitions = 8;
    } else {
        sceneDesc.cpuDispatcher = m_Dispatcher; // Only need CPU dispatcher if running on CPU? Actually need it anyway for callbacks/triggers.
    }

    // Create CPU dispatcher
    // If determinism is enabled, use 1 thread. Otherwise use 2 (or more).
    int threadCount = g_EngineConfig.enableDeterminism ? 1 : 2;
    if (g_EngineConfig.enableDeterminism) {
        std::cout << "PhysX: Determinism enabled. Enforcing single-threaded CPU dispatch." << std::endl;
    }
    m_Dispatcher = PxDefaultCpuDispatcherCreate(threadCount);
    sceneDesc.cpuDispatcher = m_Dispatcher;
    
    // Set custom filter shader for callbacks
    sceneDesc.filterShader = PhysXSimulationFilterShader;

    // Enable CCD in scene
    sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;

    enum class CollisionEventType {
        Enter,
        Stay,
        Exit
    };
    // Global Collision Callback
    using GlobalCollisionCallback = std::function<void(IPhysicsRigidBody*, IPhysicsRigidBody*, const Vec3&, const Vec3&, float, CollisionEventType)>;
    void SetGlobalCollisionCallback(GlobalCollisionCallback callback);

    // Set simulation event callback
    m_CollisionCallback = new PhysXCollisionCallback(this);
    sceneDesc.simulationEventCallback = m_CollisionCallback;

    // Set contact modify callback
    m_ContactModifyCallback = new PhysXContactModifyCallback(this);
    sceneDesc.contactModifyCallback = m_ContactModifyCallback;

/*
    // Replaced by CreateScene call above
    m_Scene = m_Physics->createScene(sceneDesc);
    if (!m_Scene) {
        std::cerr << "Failed to create PhysX scene!" << std::endl;
        return;
    }
*/

/*
    // Configure scene for PVD (Moved to CreateSceneInternal)
    if (m_Pvd) {
        PxPvdSceneClient* pvdClient = m_Scene->getScenePvdClient();
        if (pvdClient) {
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
        }
    }
*/

    m_Initialized = true;
    std::cout << "PhysX initialized successfully!" << std::endl;
}

void PhysXBackend::InitializePhysXVisualDebugger() {
    if (!g_EngineConfig.enablePhysXVisualDebugger) {
        return;
    }

    // Create PVD connection
    m_Pvd = PxCreatePvd(*m_Foundation);
    if (!m_Pvd) {
        std::cerr << "Warning: Failed to create PhysX Visual Debugger" << std::endl;
        return;
    }

    // Try to connect to PVD (localhost:5425)
    // Increased timeout to 100ms to allow for better connection reliability
    PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 100);
    if (transport) {
        bool connected = m_Pvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
        if (connected) {
            std::cout << "PhysX Visual Debugger connected!" << std::endl;
        } else {
            std::cout << "PhysX Visual Debugger not running (connection attempt failed)" << std::endl;
        }
    }
}

void PhysXBackend::Shutdown() {
    if (!m_Initialized) {
        return;
    }

    std::cout << "Shutting down PhysX..." << std::endl;

    // Clear registered components
    m_RigidBodies.clear();
    m_CharacterControllers.clear();
    m_SoftBodies.clear();
    m_SoftBodies.clear();
    m_Vehicles.clear();
    m_Articulations.clear();
    m_Aggregates.clear();
    
    // Close Vehicle SDK
    PxCloseVehicleSDK();

    // Release PhysX objects in reverse order of creation
    if (m_DefaultMaterial) {
        m_DefaultMaterial->release();
        m_DefaultMaterial = nullptr;
    }

    // Release all scenes
    for (auto& pair : m_Scenes) {
        if (pair.second) {
            pair.second->release();
        }
    }
    m_Scenes.clear();
    m_ActiveScene = nullptr;

    if (m_Dispatcher) {
        m_Dispatcher->release();
        m_Dispatcher = nullptr;
    }

    if (m_Physics) {
        m_Physics->release();
        m_Physics = nullptr;
    }

    if (m_Pvd) {
        PxPvdTransport* transport = m_Pvd->getTransport();
        if (transport) {
            transport->release();
        }
        m_Pvd->release();
        m_Pvd = nullptr;
    }

    if (m_CudaContextManager) {
        m_CudaContextManager->release();
        m_CudaContextManager = nullptr;
    }

    if (m_Foundation) {
        m_Foundation->release();
        m_Foundation = nullptr;
    }

    delete m_CollisionCallback;
    delete m_ContactModifyCallback;
    delete m_ErrorCallback;
    delete m_Allocator;

    m_Initialized = false;
    std::cout << "PhysX shutdown complete" << std::endl;
}

void PhysXBackend::Update(float deltaTime, int subSteps) {
    if (!m_Initialized || m_Scenes.empty()) {
        return;
    }

    m_SimulationTime += deltaTime;

    // Update character controllers first
    UpdateCharacterControllers(deltaTime);

    // Update soft bodies
    for (auto* softBody : m_SoftBodies) {
        if (softBody && softBody->IsEnabled()) {
            softBody->Update(deltaTime);
        }
    }

    for (auto* cloth : m_Cloths) {
        if (cloth && cloth->IsEnabled()) {
            cloth->Update(deltaTime);
        }
    }

    // Update vehicles
    if (!m_Vehicles.empty()) {
        // Prepare vehicle update data
        std::vector<PxVehicleWheels*> vehicles;
        for (auto* vehicle : m_Vehicles) {
            if (vehicle && vehicle->GetPxVehicle()) {
                vehicles.push_back(vehicle->GetPxVehicle());
            }
        }

        if(!vehicles.empty()) {
            // Raycasts for suspension
            PxRaycastQueryResult* raycastResults = new PxRaycastQueryResult[vehicles.size()];
            PxVehicleSuspensionRaycasts(
                m_Scene->getBatchQuery(PxBatchQueryDesc(vehicles.size(), 0, 0)), 
                vehicles.size(), 
                vehicles.data(), 
                vehicles.size(), 
                raycastResults
            );

            // Vehicle updates
            std::vector<PxVehicleWheelQueryResult> vehicleQueryResults(vehicles.size());
            std::vector<PxVehicleConcurrentUpdateData> concurrentUpdateData(vehicles.size());
            PxVehicleUpdates(
                deltaTime, 
                m_ActiveScene->getGravity(), 
                m_ActiveScene->getFrictionType(), 
                vehicles.size(), 
                vehicles.data(), 
                NULL, 
                vehicleQueryResults.data()
            );

            delete[] raycastResults;
        }
        
        // Post-update sync
        for (auto* vehicle : m_Vehicles) {
            vehicle->SyncTransform();
        }
    }

    // Update vehicles
    if (!m_Vehicles.empty() && m_ActiveScene) {
        // Prepare vehicle update data
        std::vector<PxVehicleWheels*> vehicles;
        for (auto* vehicle : m_Vehicles) {
            if (vehicle && vehicle->GetPxVehicle()) {
                vehicles.push_back(vehicle->GetPxVehicle());
            }
        }

        if(!vehicles.empty()) {
            // Raycasts for suspension
            PxRaycastQueryResult* raycastResults = new PxRaycastQueryResult[vehicles.size()];
            PxVehicleSuspensionRaycasts(
                m_ActiveScene->getBatchQuery(PxBatchQueryDesc(vehicles.size(), 0, 0)), 
                vehicles.size(), 
                vehicles.data(), 
                vehicles.size(), 
                raycastResults
            );

            // Vehicle updates
            std::vector<PxVehicleWheelQueryResult> vehicleQueryResults(vehicles.size());
            std::vector<PxVehicleConcurrentUpdateData> concurrentUpdateData(vehicles.size());
            PxVehicleUpdates(
                deltaTime, 
                m_ActiveScene->getGravity(), 
                m_ActiveScene->getFrictionType(), 
                vehicles.size(), 
                vehicles.data(), 
                NULL, 
                vehicleQueryResults.data()
            );

            delete[] raycastResults;
        }
        
        // Post-update sync
        for (auto* vehicle : m_Vehicles) {
            vehicle->SyncTransform();
        }
    }

    // Apply buoyancy and other per-frame forces to rigid bodies
    // Optimization: Only update active bodies? 
    // For now iterate all registered bodies to ensure we catch everything, 
    // but PhysXRigidBody::UpdateBuoyancy checks if it has active fluids.
    // If we want to be faster, we should only check active bodies.
    for (auto* body : m_RigidBodies) {
        if (body && body->IsActive()) {
             static_cast<PhysXRigidBody*>(body)->UpdateBuoyancy(deltaTime);
        }
    }

    // Simulate all scenes
    PhysicsStats frameStats = {};
    frameStats.activeScenes = 0;
    
    for (auto& pair : m_Scenes) {
         PxScene* scene = pair.second;
         if (scene) {
             scene->simulate(deltaTime);
             scene->fetchResults(true);
             
             // Gather stats
             PxSimulationStatistics pxStats;
             scene->getSimulationStatistics(pxStats);
             
             frameStats.activeScenes++;
             frameStats.activeRigidBodies += pxStats.nbActiveDynamicBodies; // + nbActiveKinematic?
             frameStats.staticRigidBodies += pxStats.nbStaticBodies;
             frameStats.kinematicRigidBodies += pxStats.nbActiveKinematicBodies;
             frameStats.dynamicRigidBodies += pxStats.nbDynamicBodies;
             frameStats.aggregates += pxStats.nbAggregates;
             frameStats.articulations += pxStats.nbArticulations;
             frameStats.constraints += pxStats.nbActiveConstraints;
             frameStats.broadPhaseAdds += pxStats.nbBroadPhaseAdds;
             frameStats.broadPhaseRemoves += pxStats.nbBroadPhaseRemoves;
             frameStats.narrowPhaseTouches += pxStats.nbDiscreteContactPairs[PxGeometryType::eBOX][PxGeometryType::eBOX]; // This is 2D array, we want total
             
             // Sum all contact pairs
             for(int i=0; i < PxGeometryType::eGEOMETRY_COUNT; i++) {
                 for(int j=0; j < PxGeometryType::eGEOMETRY_COUNT; j++) {
                    frameStats.narrowPhaseTouches += pxStats.nbDiscreteContactPairs[i][j];
                 }
             }
             
             frameStats.lostTouches += pxStats.nbLostTouchPairs;
             frameStats.lostPairs += pxStats.nbLostPairs;
         }
    }
    
    // Update GPU memory usage
    frameStats.gpuMemoryUsageMB = GetGpuMemoryUsageMB();
    
    m_LastFrameStats = frameStats;
}

PhysicsStats PhysXBackend::GetStatistics() const {
    return m_LastFrameStats;
}

// Scene Management

physx::PxScene* PhysXBackend::CreateScene(const std::string& name, const Vec3& gravity) {
    PxVec3 pxGravity(gravity.x, gravity.y, gravity.z);
    PxScene* newScene = CreateSceneInternal(name, pxGravity);
    if (newScene) {
        m_Scenes[name] = newScene;
        if (!m_ActiveScene) {
            m_ActiveScene = newScene;
        }
        return newScene;
    }
    return nullptr;
}

void PhysXBackend::ReleaseScene(const std::string& name) {
    auto it = m_Scenes.find(name);
    if (it != m_Scenes.end()) {
        if (m_ActiveScene == it->second) {
            m_ActiveScene = nullptr; 
            // Pick another if available?
            if (m_Scenes.size() > 1) {
                // Find one that isn't this one
                 for(auto& pair : m_Scenes) {
                     if (pair.first != name) {
                         m_ActiveScene = pair.second;
                         break;
                     }
                 }
            }
        }
        
        it->second->release();
        m_Scenes.erase(it);
    }
}

physx::PxScene* PhysXBackend::GetScene(const std::string& name) {
    auto it = m_Scenes.find(name);
    if (it != m_Scenes.end()) {
        return it->second;
    }
    return nullptr;
}

void PhysXBackend::SetActiveScene(const std::string& name) {
    auto it = m_Scenes.find(name);
    if (it != m_Scenes.end()) {
        m_ActiveScene = it->second;
    }
}


physx::PxScene* PhysXBackend::CreateSceneInternal(const std::string& name, const Vec3& gravity) {
    if (!m_Physics || !m_Dispatcher) return nullptr;

    PxSceneDesc sceneDesc(m_Physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(gravity.x, gravity.y, gravity.z);

    if (m_CudaContextManager) {
        sceneDesc.cudaContextManager = m_CudaContextManager;
        sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        sceneDesc.flags |= PxSceneFlag::eENABLE_PCM;
        sceneDesc.flags |= PxSceneFlag::eENABLE_STABILIZATION;
        sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
        sceneDesc.gpuMaxNumPartitions = 8;
    } else {
        sceneDesc.cpuDispatcher = m_Dispatcher;
    }

    // Always use CPU dispatcher for callbacks
    sceneDesc.cpuDispatcher = m_Dispatcher;
    sceneDesc.filterShader = PhysXSimulationFilterShader;
    sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;

    if (g_EngineConfig.enableDeterminism) {
        sceneDesc.flags |= PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;
        // Disable GPU dynamics for strict determinism? 
        // PhysX docs say GPU can be deterministic, but let's stick to CPU single-thread for now unless specified otherwise.
        // If we want STRICT determinism, usually avoiding GPU is safer or requires specific config.
        // For now, if determinism is enabled, we keep GPU check above logic but ensure Enhanced flag is set.
        // However, if we forced 1 CPU thread, we should probably respect that for the scene logic too. 
        // But the scene logic above checks m_CudaContextManager. 
        // Let's modify logic: if Determinism is enabled, ignore CUDA?
        // Or trust that user won't have CUDA if they want CPU determinism?
        // Let's force CPU if determinism is ON for safety.
        
        // Remove GPU flags if they were added
        if (g_EngineConfig.enableDeterminism) {
             sceneDesc.flags &= ~PxSceneFlag::eENABLE_GPU_DYNAMICS;
             sceneDesc.cudaContextManager = nullptr;
             sceneDesc.broadPhaseType = PxBroadPhaseType::eSAP; // SAP is often more deterministic than MBP? MBP is default. 
        }
    }

    // Use same callbacks - Need to make sure callbacks handle multiple scenes or stick to single instance?
    // Same callback instance is fine, but it will receive events from ALL scenes.
    sceneDesc.simulationEventCallback = m_CollisionCallback;
    sceneDesc.contactModifyCallback = m_ContactModifyCallback;

    PxScene* scene = m_Physics->createScene(sceneDesc);
    if (!scene) {
        std::cerr << "Failed to create PhysX scene: " << name << std::endl;
        return nullptr;
    }

    if (m_Pvd) {
        PxPvdSceneClient* pvdClient = scene->getScenePvdClient();
        if (pvdClient) {
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
        }
    }
    
    return scene;
}

void PhysXBackend::UpdateCharacterControllers(float deltaTime) {
    for (auto* controller : m_CharacterControllers) {
        if (controller) {
            controller->Update(deltaTime, m_Gravity);
        }
    }
}

void PhysXBackend::SetGravity(const Vec3& gravity) {
    m_Gravity = gravity;
    // Update all scenes gravity? Or just active?
    // Let's update all for now since global gravity setter implies global setting
    for (auto& pair : m_Scenes) {
        if (pair.second) {
            pair.second->setGravity(PxVec3(gravity.x, gravity.y, gravity.z));
        }
    }
}

Vec3 PhysXBackend::GetGravity() const {
    return m_Gravity;
}

bool PhysXBackend::Raycast(const Vec3& from, const Vec3& to, PhysicsRaycastHit& hit, uint32_t filter) {
    if (!m_ActiveScene) {
        return false;
    }
    
    // Use Active Scene
    PxScene* scene = m_ActiveScene;

    PxVec3 origin(from.x, from.y, from.z);
    PxVec3 direction = PxVec3(to.x - from.x, to.y - from.y, to.z - from.z);
    float distance = direction.magnitude();
    direction.normalize();

    PxQueryFilterData filterData;
    filterData.data.word0 = filter;
    filterData.flags = PxQueryFlag::eSTATIC | PxQueryFlag::eDYNAMIC | PxQueryFlag::ePREFILTER;

    PxRaycastBuffer hitBuffer;
    bool hasHit = scene->raycast(origin, direction, distance, hitBuffer, PxHitFlag::eDEFAULT, filterData);

    if (hasHit && hitBuffer.hasBlock) {
        const PxRaycastHit& pxHit = hitBuffer.block;
        hit.point = Vec3(pxHit.position.x, pxHit.position.y, pxHit.position.z);
        hit.normal = Vec3(pxHit.normal.x, pxHit.normal.y, pxHit.normal.z);
        hit.distance = pxHit.distance;
        hit.userData = pxHit.actor ? pxHit.actor->userData : nullptr;
    } else {
        // If no physx hit, initialize distance to max
        hit.distance = distance;
    }

    // Raycast against cloths
    // Since cloths are not fully in scene raycast (or mesh not updated), check manually
    // Optimization: Check AABB first
    for (auto* cloth : m_Cloths) {
        if (!cloth || !cloth->IsEnabled()) continue;

        // Get bounds
        Vec3 minBounds, maxBounds;
        cloth->GetWorldBounds(minBounds, maxBounds);
        
        // Simple Ray-AABB intersection check
        // slabs method
        float tmin = 0.0f; 
        float tmax = hit.distance; // Don't check further than current closest hit
        
        Vec3 dirInv(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
        
        float tx1 = (minBounds.x - origin.x) * dirInv.x;
        float tx2 = (maxBounds.x - origin.x) * dirInv.x;
        
        float tmin_x = std::min(tx1, tx2);
        float tmax_x = std::max(tx1, tx2);
        
        float ty1 = (minBounds.y - origin.y) * dirInv.y;
        float ty2 = (maxBounds.y - origin.y) * dirInv.y;
        
        float tmin_y = std::min(ty1, ty2);
        float tmax_y = std::max(ty1, ty2);
        
        float tz1 = (minBounds.z - origin.z) * dirInv.z;
        float tz2 = (maxBounds.z - origin.z) * dirInv.z;
        
        float tmin_z = std::min(tz1, tz2);
        float tmax_z = std::max(tz1, tz2);
        
        float t_enter = std::max(std::max(tmin_x, tmin_y), tmin_z);
        float t_exit = std::min(std::min(tmax_x, tmax_y), tmax_z);
        
        // If no intersection or intersection is larger than current hit distance
        if (t_enter > t_exit || t_exit < 0 || t_enter > hit.distance) {
            continue;
        }

        PhysicsRaycastHit clothHit;
        if (cloth->Raycast(from, to, clothHit)) {
            if (clothHit.distance < hit.distance) {
                hit = clothHit;
                hasHit = true;
                // hit.userData = cloth->GetNativeCloth(); // Optionally set userData
            }
        }
    }

    return hasHit;
}

int PhysXBackend::OverlapSphere(const Vec3& center, float radius, std::vector<void*>& results, uint32_t filter) {
    if (!m_ActiveScene) return 0;
    PxScene* scene = m_ActiveScene;

    PxSphereGeometry sphereGeom(radius);
    PxTransform pose(PxVec3(center.x, center.y, center.z));
    
    // Max overlaps to retrieve
    const PxU32 bufferSize = 256;
    PxOverlapHit hitBuffer[bufferSize];
    PxOverlapBuffer buf(hitBuffer, bufferSize);
    
    PxQueryFilterData filterData;
    filterData.data.word0 = filter;
    filterData.flags = PxQueryFlag::eSTATIC | PxQueryFlag::eDYNAMIC | PxQueryFlag::ePREFILTER | PxQueryFlag::eNO_BLOCK;

    if (scene->overlap(sphereGeom, pose, buf, filterData)) {
        for (PxU32 i = 0; i < buf.nbTouches; i++) {
            if (buf.touches[i].actor && buf.touches[i].actor->userData) {
                results.push_back(buf.touches[i].actor->userData);
            }
        }
        return static_cast<int>(buf.nbTouches);
    }
    
    return 0;
}

int PhysXBackend::OverlapBox(const Vec3& center, const Vec3& halfExtents, const Quat& rotation, std::vector<void*>& results, uint32_t filter) {
    if (!m_ActiveScene) return 0;
    PxScene* scene = m_ActiveScene;

    PxBoxGeometry boxGeom(halfExtents.x, halfExtents.y, halfExtents.z);
    PxTransform pose(
        PxVec3(center.x, center.y, center.z),
        PxQuat(rotation.x, rotation.y, rotation.z, rotation.w)
    );
    
    const PxU32 bufferSize = 256;
    PxOverlapHit hitBuffer[bufferSize];
    PxOverlapBuffer buf(hitBuffer, bufferSize);
    
    PxQueryFilterData filterData;
    filterData.data.word0 = filter;
    filterData.flags = PxQueryFlag::eSTATIC | PxQueryFlag::eDYNAMIC | PxQueryFlag::ePREFILTER | PxQueryFlag::eNO_BLOCK;

    if (scene->overlap(boxGeom, pose, buf, filterData)) {
        for (PxU32 i = 0; i < buf.nbTouches; i++) {
            if (buf.touches[i].actor && buf.touches[i].actor->userData) {
                results.push_back(buf.touches[i].actor->userData);
            }
        }
        return static_cast<int>(buf.nbTouches);
    }
    
    return 0;
}

int PhysXBackend::OverlapCapsule(const Vec3& center, float radius, float halfHeight, const Quat& rotation, std::vector<void*>& results, uint32_t filter) {
    if (!m_ActiveScene) return 0;
    PxScene* scene = m_ActiveScene;

    // PhysX capsule is defined along X axis by default geometry constructor in previous versions?
    // Actually PxCapsuleGeometry: "The capsule is centered at the origin, extending along the x-axis" is FALSE.
    // Documentation says: "The capsule data is centered at the origin, extending along the x-axis." - Wait, let me check.
    // Standard PhysX capsule aligns with X axis. PxCapsuleGeometry(radius, halfHeight).
    // Our engine likely expects Y or Z alignment if not specified, but rotation handles it.
    // Generally capsules are local X-axis in PhysX. If the user passes a rotation that assumes Y-axis capsule (like Unity/Unreal defaults often do), we might need to pre-rotate.
    // Let's assume the user passes a rotation that correctly orients the capsule.
    
    PxCapsuleGeometry capsuleGeom(radius, halfHeight);
    PxTransform pose(
        PxVec3(center.x, center.y, center.z),
        PxQuat(rotation.x, rotation.y, rotation.z, rotation.w)
    );
    
    const PxU32 bufferSize = 256;
    PxOverlapHit hitBuffer[bufferSize];
    PxOverlapBuffer buf(hitBuffer, bufferSize);
    
    PxQueryFilterData filterData;
    filterData.data.word0 = filter;
    filterData.flags = PxQueryFlag::eSTATIC | PxQueryFlag::eDYNAMIC | PxQueryFlag::ePREFILTER | PxQueryFlag::eNO_BLOCK;

    if (scene->overlap(capsuleGeom, pose, buf, filterData)) {
        for (PxU32 i = 0; i < buf.nbTouches; i++) {
            if (buf.touches[i].actor && buf.touches[i].actor->userData) {
                results.push_back(buf.touches[i].actor->userData);
            }
        }
        return static_cast<int>(buf.nbTouches);
    }
    
    return 0;
}

int PhysXBackend::GetNumRigidBodies() const {
    return static_cast<int>(m_RigidBodies.size());
}

void PhysXBackend::SetDebugDrawEnabled(bool enabled) {
    m_DebugDrawEnabled = enabled;
    // PhysX debug visualization is handled through PVD
}

bool PhysXBackend::IsDebugDrawEnabled() const {
    return m_DebugDrawEnabled;
}

void* PhysXBackend::GetNativeWorld() {
    return m_ActiveScene;
}

void PhysXBackend::ApplyImpulse(void* userData, const Vec3& impulse, const Vec3& point) {
    if (!userData) return;
    
    // Cast generic pointer to PxRigidActor
    physx::PxRigidActor* actor = static_cast<physx::PxRigidActor*>(userData);
    
    // Check if it's dynamic
    if (actor->getType() == physx::PxActorType::eRIGID_DYNAMIC) {
        physx::PxRigidDynamic* dynamicActor = static_cast<physx::PxRigidDynamic*>(actor);
        
        // Only apply if not kinematic
        if (!(dynamicActor->getRigidBodyFlags() & physx::PxRigidBodyFlag::eKINEMATIC)) {
            physx::PxVec3 pxImpulse(impulse.x, impulse.y, impulse.z);
            physx::PxVec3 pxPoint(point.x, point.y, point.z);
            
            physx::PxRigidBodyExt::addForceAtPos(*dynamicActor, pxImpulse, pxPoint, physx::PxForceMode::eIMPULSE);
        }
    }
    // TODO: Verify if Articulations need special handling, but PxRigidActor covers standard rigid bodies
}

void PhysXBackend::RegisterRigidBody(IPhysicsRigidBody* body) {
    if (body) {
        m_RigidBodies.push_back(body);
    }
}

void PhysXBackend::UnregisterRigidBody(IPhysicsRigidBody* body) {
    auto it = std::find(m_RigidBodies.begin(), m_RigidBodies.end(), body);
    if (it != m_RigidBodies.end()) {
        m_RigidBodies.erase(it);
    }
}

void PhysXBackend::RegisterCharacterController(IPhysicsCharacterController* controller) {
    if (controller) {
        m_CharacterControllers.push_back(controller);
    }
}

void PhysXBackend::UnregisterCharacterController(IPhysicsCharacterController* controller) {
    auto it = std::find(m_CharacterControllers.begin(), m_CharacterControllers.end(), controller);
    if (it != m_CharacterControllers.end()) {
        m_CharacterControllers.erase(it);
    }
}

void PhysXBackend::RegisterSoftBody(IPhysicsSoftBody* softBody) {
    if (softBody) {
        m_SoftBodies.push_back(softBody);
    }
}

void PhysXBackend::UnregisterSoftBody(IPhysicsSoftBody* softBody) {
    auto it = std::find(m_SoftBodies.begin(), m_SoftBodies.end(), softBody);
    if (it != m_SoftBodies.end()) {
        m_SoftBodies.erase(it);
    }
}

void PhysXBackend::RegisterCloth(IPhysicsCloth* cloth) {
    if (cloth) {
        m_Cloths.push_back(cloth);
    }
}

void PhysXBackend::UnregisterCloth(IPhysicsCloth* cloth) {
    auto it = std::find(m_Cloths.begin(), m_Cloths.end(), cloth);
    if (it != m_Cloths.end()) {
        m_Cloths.erase(it);
    }
}

void PhysXBackend::RegisterVehicle(PhysXVehicle* vehicle) {
    if (vehicle) {
        m_Vehicles.push_back(vehicle);
    }
}

void PhysXBackend::UnregisterVehicle(PhysXVehicle* vehicle) {
    auto it = std::find(m_Vehicles.begin(), m_Vehicles.end(), vehicle);
    if (it != m_Vehicles.end()) {
        m_Vehicles.erase(it);
    }
}

void PhysXBackend::RegisterArticulation(PhysXArticulation* articulation) {
    if (articulation) {
        m_Articulations.push_back(articulation);
    }
}

void PhysXBackend::UnregisterArticulation(PhysXArticulation* articulation) {
    auto it = std::find(m_Articulations.begin(), m_Articulations.end(), articulation);
    if (it != m_Articulations.end()) {
        m_Articulations.erase(it);
    }
}

void PhysXBackend::RegisterAggregate(PhysXAggregate* aggregate) {
    if (aggregate) {
        m_Aggregates.push_back(aggregate);
    }
}

    void PhysXBackend::UnregisterAggregate(PhysXAggregate* aggregate) {
    auto it = std::find(m_Aggregates.begin(), m_Aggregates.end(), aggregate);
    if (it != m_Aggregates.end()) {
        m_Aggregates.erase(it);
    }
}

void PhysXBackend::RegisterContactModifyListener(IPhysXContactModifyListener* listener) {
    if (listener) {
        m_ContactModifyListeners.push_back(listener);
    }
}

void PhysXBackend::UnregisterContactModifyListener(IPhysXContactModifyListener* listener) {
    auto it = std::find(m_ContactModifyListeners.begin(), m_ContactModifyListeners.end(), listener);
    if (it != m_ContactModifyListeners.end()) {
        m_ContactModifyListeners.erase(it);
    }
}

void PhysXBackend::RegisterFluidVolume(PhysXFluidVolume* volume) {
    if (volume && volume->GetTriggerBody()) {
        m_FluidVolumeMap[volume->GetTriggerBody()] = volume;
    }
}

void PhysXBackend::UnregisterFluidVolume(PhysXFluidVolume* volume) {
    if (volume && volume->GetTriggerBody()) {
        m_FluidVolumeMap.erase(volume->GetTriggerBody());
    }
}

PhysXFluidVolume* PhysXBackend::GetFluidVolume(IPhysicsRigidBody* body) {
    if (!body) return nullptr;
    auto it = m_FluidVolumeMap.find(body);
    if (it != m_FluidVolumeMap.end()) {
        return it->second;
    }
    return nullptr;
}

bool PhysXBackend::IsGpuSoftBodySupported() const {
    // PhysX 5.x soft bodies require GPU
    // Check if CUDA context is available and valid
    if (!m_CudaContextManager || !m_CudaContextManager->contextIsValid()) {
        return false;
    }
    
    // Check compute capability (soft bodies typically require at least 3.0)
    int computeCapability = 0;
    size_t totalMem = 0, freeMem = 0;
    GetGpuDeviceProperties(computeCapability, totalMem, freeMem);
    
    // Require at least compute capability 3.0 and 512MB free memory
    return (computeCapability >= 30) && (freeMem >= 512);
}

void PhysXBackend::GetGpuDeviceProperties(int& computeCapability, size_t& totalMemoryMB, size_t& freeMemoryMB) const {
    computeCapability = 0;
    totalMemoryMB = 0;
    freeMemoryMB = 0;
    
    if (!m_CudaContextManager) {
        return;
    }
    
    // Get device ordinal from PhysX CUDA context manager
    int deviceOrdinal = m_CudaContextManager->getDeviceOrdinal();
    
    #ifdef HAS_CUDA_TOOLKIT
    // Query compute capability using CUDA Runtime API
    int major = 0, minor = 0;
    cudaError_t err1 = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceOrdinal);
    cudaError_t err2 = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceOrdinal);
    
    if (err1 == cudaSuccess && err2 == cudaSuccess) {
        computeCapability = major * 10 + minor;
    } else {
        // Fallback to estimate if CUDA query fails
        std::cerr << "Warning: CUDA device attribute query failed, using estimate" << std::endl;
        computeCapability = 75; // Assume modern GPU
    }
    
    // Query memory info using CUDA Runtime API
    size_t free = 0, total = 0;
    cudaError_t err3 = cudaMemGetInfo(&free, &total);
    
    if (err3 == cudaSuccess) {
        totalMemoryMB = total / (1024 * 1024);
        freeMemoryMB = free / (1024 * 1024);
    } else {
        // Fallback to estimates if CUDA query fails
        std::cerr << "Warning: CUDA memory query failed, using estimates" << std::endl;
        totalMemoryMB = 8192;  // 8GB typical
        freeMemoryMB = 6144;   // 6GB free typical
    }
    #else
    // CUDA Toolkit not available - use estimates
    if (m_CudaContextManager->contextIsValid()) {
        computeCapability = 75;  // Assume modern GPU
        totalMemoryMB = 8192;    // 8GB typical
        freeMemoryMB = 6144;     // 6GB free typical
    }
    #endif
}

size_t PhysXBackend::GetGpuMemoryUsageMB() const {
    return m_GpuMemoryUsageMB;
}

bool PhysXBackend::IsGpuRigidBodySupported() const {
    return m_CudaContextManager && m_CudaContextManager->contextIsValid();
}

void PhysXBackend::GetActiveRigidBodies(std::vector<IPhysicsRigidBody*>& outBodies) {
    if (!m_Scene) return;
    
    outBodies.clear();
    
    physx::PxU32 nbActiveActors = 0;
    physx::PxActor** activeActors = m_Scene->getActiveActors(nbActiveActors);
    
    if (nbActiveActors > 0) {
        outBodies.reserve(nbActiveActors);
        for (physx::PxU32 i = 0; i < nbActiveActors; i++) {
            if (activeActors[i]->userData) {
                // Ensure it's a rigid body (could verify type if needed, but userData convention holds)
                outBodies.push_back(static_cast<IPhysicsRigidBody*>(activeActors[i]->userData));
            }
        }
    }
}


// Custom filter shader
physx::PxFilterFlags PhysXSimulationFilterShader(
    physx::PxFilterObjectAttributes attributes0, physx::PxFilterData filterData0,
    physx::PxFilterObjectAttributes attributes1, physx::PxFilterData filterData1,
    physx::PxPairFlags& pairFlags, const void* constantBlock, physx::PxU32 constantBlockSize)
{
    // Let the default filter set the basic flags
    PxFilterFlags flags = PxDefaultSimulationFilterShader(attributes0, filterData0, attributes1, filterData1, pairFlags, constantBlock, constantBlockSize);
    
    // Check if either object is a trigger
    if (PxFilterObjectIsTrigger(attributes0) || PxFilterObjectIsTrigger(attributes1)) {
        pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
        return PxFilterFlag::eDEFAULT;
    }

    // Enable contact reports for all colliding pairs
    if (!(flags & PxFilterFlag::eSUPPRESS)) {
        pairFlags |= PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_TOUCH_LOST | PxPairFlag::eNOTIFY_TOUCH_UHD_FORCE | PxPairFlag::eMODIFY_CONTACTS;
    }
    
    return flags;
}

void PhysXBackend::PhysXContactModifyCallback::onContactModify(physx::PxContactModifyPair* const pairs, physx::PxU32 count) {
    for (auto* listener : m_Backend->m_ContactModifyListeners) {
        listener->OnContactModify(pairs, count);
    }
}

void PhysXBackend::PhysXCollisionCallback::onTrigger(physx::PxTriggerPair* pairs, physx::PxU32 count) {
    for (PxU32 i = 0; i < count; i++) {
        PxTriggerPair& pair = pairs[i];
        
        // Ignore pairs with deleted shapes
        if (pair.flags & (PxTriggerPairFlag::eREMOVED_SHAPE_TRIGGER | PxTriggerPairFlag::eREMOVED_SHAPE_OTHER))
            continue;
            
        PxActor* triggerActor = pair.triggerActor;
        PxActor* otherActor = pair.otherActor;
        
        if (!triggerActor || !otherActor) continue;
        
        IPhysicsRigidBody* triggerBody = static_cast<IPhysicsRigidBody*>(triggerActor->userData);
        IPhysicsRigidBody* otherBody = static_cast<IPhysicsRigidBody*>(otherActor->userData);
        
        if (!triggerBody) continue;
        
        // Calculate relative velocity
        Vec3 relVel(0,0,0);
        PxRigidActor* actor0 = triggerActor; // trigger
        PxRigidActor* actor1 = otherActor;   // other
        
        if (actor0->is<PxRigidDynamic>() && actor1->is<PxRigidDynamic>()) {
             PxVec3 v0 = static_cast<PxRigidDynamic*>(actor0)->getLinearVelocity();
             PxVec3 v1 = static_cast<PxRigidDynamic*>(actor1)->getLinearVelocity();
             PxVec3 v = v1 - v0; // Relative to trigger body (velocity of other IN trigger frame? usually other - trigger)
             relVel = Vec3(v.x, v.y, v.z);
        }
        else if (actor1->is<PxRigidDynamic>()) {
             PxVec3 v = static_cast<PxRigidDynamic*>(actor1)->getLinearVelocity();
             relVel = Vec3(v.x, v.y, v.z);
        }
         else if (actor0->is<PxRigidDynamic>()) {
             PxVec3 v = static_cast<PxRigidDynamic*>(actor0)->getLinearVelocity();
             relVel = Vec3(-v.x, -v.y, -v.z); // Relative to trigger
        }

        IPhysicsRigidBody::TriggerInfo info;
        info.otherBody = otherBody;
        info.relativeVelocity = relVel;
        
        if (pair.status & PxPairFlag::eNOTIFY_TOUCH_FOUND) {
            static_cast<PhysXRigidBody*>(triggerBody)->HandleTriggerEnter(info);
        }
        else if (pair.status & PxPairFlag::eNOTIFY_TOUCH_LOST) {
            static_cast<PhysXRigidBody*>(triggerBody)->HandleTriggerExit(info);
        }
    }
}

void PhysXBackend::PhysXCollisionCallback::onContact(const physx::PxContactPairHeader& pairHeader, const physx::PxContactPair* pairs, physx::PxU32 nbPairs) {
    for (PxU32 i = 0; i < nbPairs; i++) {
        const PxContactPair& cp = pairs[i];
        
        PhysXBackend::CollisionEventType eventType = PhysXBackend::CollisionEventType::Stay;
        if (cp.events & PxPairFlag::eNOTIFY_TOUCH_FOUND) eventType = PhysXBackend::CollisionEventType::Enter;
        else if (cp.events & PxPairFlag::eNOTIFY_TOUCH_LOST) eventType = PhysXBackend::CollisionEventType::Exit;
        // else PERSISTS is Stay

        // For Exit, contact points might not be available or valid in the same way, 
        // but we still want to notify systems to stop sounds.
        
        PxActor* actor0 = pairHeader.actors[0];
        PxActor* actor1 = pairHeader.actors[1];

        if (!actor0 || !actor1) continue;

        IPhysicsRigidBody* body0 = static_cast<IPhysicsRigidBody*>(actor0->userData);
        IPhysicsRigidBody* body1 = static_cast<IPhysicsRigidBody*>(actor1->userData);

        if (!body0 && !body1) continue;
        
        // Extract info
        Vec3 point(0,0,0);
        Vec3 normal(0,1,0);
        float impulseMag = 0.0f;
        
        // If Exit, we might not have patches
        if (eventType != PhysXBackend::CollisionEventType::Exit) {
            PxContactStreamIterator iter(cp.contactPatches, cp.contactPoints, cp.getInternalFaceIndices(), cp.patchCount, cp.contactCount);
            if (iter.hasNextPatch()) {
                iter.nextPatch();
                if (iter.hasNextPoint()) {
                    iter.nextPoint();
                    PxVec3 p = iter.getContactPoint();
                    PxVec3 n = iter.getContactNormal();
                    point = Vec3(p.x, p.y, p.z);
                    normal = Vec3(n.x, n.y, n.z);
                }
            }
            
            // Calculate relative velocity/impulse
            Vec3 relVel(0,0,0);
            if (actor0->is<PxRigidDynamic>() && actor1->is<PxRigidDynamic>()) {
                 PxVec3 v0 = static_cast<PxRigidDynamic*>(actor0)->getLinearVelocity();
                 PxVec3 v1 = static_cast<PxRigidDynamic*>(actor1)->getLinearVelocity();
                 PxVec3 v = v0 - v1;
                 relVel = Vec3(v.x, v.y, v.z);
            }
            else if (actor0->is<PxRigidDynamic>()) {
                 PxVec3 v = static_cast<PxRigidDynamic*>(actor0)->getLinearVelocity();
                 relVel = Vec3(v.x, v.y, v.z);
            }
             else if (actor1->is<PxRigidDynamic>()) {
                 PxVec3 v = static_cast<PxRigidDynamic*>(actor1)->getLinearVelocity();
                 relVel = Vec3(-v.x, -v.y, -v.z); 
            }
            impulseMag = relVel.Length();
        }

        // Notify RigidBodies ONLY on Enter (legacy behavior provided by HandleCollision)
        if (eventType == PhysXBackend::CollisionEventType::Enter) {
            if (body0) {
                IPhysicsRigidBody::CollisionInfo info;
                info.point = point;
                info.normal = normal;
                info.impulse = impulseMag;
                info.otherBody = body1;
                // info.relativeVelocity // We computed it but didn't store it in valid scope earlier? 
                // Let's recalculate or clean up. 
                // For this edit, we keep it simple.
                static_cast<PhysXRigidBody*>(body0)->HandleCollision(info);
            }
            if (body1) {
                IPhysicsRigidBody::CollisionInfo info;
                info.point = point;
                info.normal = -normal;
                info.impulse = impulseMag;
                info.otherBody = body0;
                static_cast<PhysXRigidBody*>(body1)->HandleCollision(info);
            }
        }

        // Notify Global Listener (Impact System needs Enter, Rolling System needs Stay/Exit)
        if (m_Backend->m_GlobalCollisionCallback) {
            m_Backend->m_GlobalCollisionCallback(body0, body1, point, normal, impulseMag, eventType);
        }
    }
}
#endif // USE_PHYSX

