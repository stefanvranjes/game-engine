#include "PhysXBackend.h"
#include "IPhysicsRigidBody.h"
#include "IPhysicsCharacterController.h"

#ifdef USE_PHYSX

#include <PxPhysicsAPI.h>
#include <iostream>

using namespace physx;

PhysXBackend::PhysXBackend()
    : m_Foundation(nullptr)
    , m_Physics(nullptr)
    , m_Scene(nullptr)
    , m_Dispatcher(nullptr)
    , m_Pvd(nullptr)
    , m_CudaContextManager(nullptr)
    , m_DefaultMaterial(nullptr)
    , m_Allocator(nullptr)
    , m_ErrorCallback(nullptr)
    , m_Initialized(false)
    , m_DebugDrawEnabled(false)
    , m_Gravity(0, -9.81f, 0)
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

    // Try to create CUDA context manager
    PxCudaContextManagerDesc cudaCtxtDesc;
    m_CudaContextManager = PxCreateCudaContextManager(*m_Foundation, cudaCtxtDesc, PxGetProfilerCallback());
    
    if (m_CudaContextManager) {
        if (!m_CudaContextManager->contextIsValid()) {
            m_CudaContextManager->release();
            m_CudaContextManager = nullptr;
            std::cout << "PhysX: CUDA context invalid. Falling back to CPU." << std::endl;
        } else {
            std::cout << "PhysX: CUDA context created. GPU acceleration enabled." << std::endl;
        }
    } else {
        std::cout << "PhysX: CUDA context creation failed. Falling back to CPU." << std::endl;
    }

    // Create scene
    PxSceneDesc sceneDesc(m_Physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(gravity.x, gravity.y, gravity.z);
    m_Gravity = gravity;

    // Enable GPU dynamics if CUDA context is available
    if (m_CudaContextManager) {
        sceneDesc.cudaContextManager = m_CudaContextManager;
        sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        sceneDesc.flags |= PxSceneFlag::eENABLE_PCM; // Persistent Contact Manifold (usually good for GPU)
        sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
        sceneDesc.gpuMaxNumPartitions = 8;
    } else {
        sceneDesc.cpuDispatcher = m_Dispatcher; // Only need CPU dispatcher if running on CPU? Actually need it anyway for callbacks/triggers.
    }

    // Create CPU dispatcher (Always needed for API callbacks and non-GPU tasks)
    m_Dispatcher = PxDefaultCpuDispatcherCreate(2); // 2 worker threads
    sceneDesc.cpuDispatcher = m_Dispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;

    m_Scene = m_Physics->createScene(sceneDesc);
    if (!m_Scene) {
        std::cerr << "Failed to create PhysX scene!" << std::endl;
        return;
    }

    // Configure scene for PVD
    if (m_Pvd) {
        PxPvdSceneClient* pvdClient = m_Scene->getScenePvdClient();
        if (pvdClient) {
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
        }
    }

    m_Initialized = true;
    std::cout << "PhysX initialized successfully!" << std::endl;
}

void PhysXBackend::InitializePhysXVisualDebugger() {
    // Create PVD connection
    m_Pvd = PxCreatePvd(*m_Foundation);
    if (!m_Pvd) {
        std::cerr << "Warning: Failed to create PhysX Visual Debugger" << std::endl;
        return;
    }

    // Try to connect to PVD (localhost:5425)
    PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 10);
    if (transport) {
        bool connected = m_Pvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
        if (connected) {
            std::cout << "PhysX Visual Debugger connected!" << std::endl;
        } else {
            std::cout << "PhysX Visual Debugger not running (optional)" << std::endl;
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

    // Release PhysX objects in reverse order of creation
    if (m_DefaultMaterial) {
        m_DefaultMaterial->release();
        m_DefaultMaterial = nullptr;
    }

    if (m_Scene) {
        m_Scene->release();
        m_Scene = nullptr;
    }

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

    delete m_ErrorCallback;
    delete m_Allocator;

    m_Initialized = false;
    std::cout << "PhysX shutdown complete" << std::endl;
}

void PhysXBackend::Update(float deltaTime, int subSteps) {
    if (!m_Initialized || !m_Scene) {
        return;
    }

    // Update character controllers first
    UpdateCharacterControllers(deltaTime);

    // Simulate physics
    m_Scene->simulate(deltaTime);
    m_Scene->fetchResults(true); // Block until simulation completes
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
    if (m_Scene) {
        m_Scene->setGravity(PxVec3(gravity.x, gravity.y, gravity.z));
    }
}

Vec3 PhysXBackend::GetGravity() const {
    return m_Gravity;
}

bool PhysXBackend::Raycast(const Vec3& from, const Vec3& to, RaycastHit& hit, uint32_t filter) {
    if (!m_Scene) {
        return false;
    }

    PxVec3 origin(from.x, from.y, from.z);
    PxVec3 direction = PxVec3(to.x - from.x, to.y - from.y, to.z - from.z);
    float distance = direction.magnitude();
    direction.normalize();

    PxRaycastBuffer hitBuffer;
    bool hasHit = m_Scene->raycast(origin, direction, distance, hitBuffer);

    if (hasHit && hitBuffer.hasBlock) {
        const PxRaycastHit& pxHit = hitBuffer.block;
        hit.point = Vec3(pxHit.position.x, pxHit.position.y, pxHit.position.z);
        hit.normal = Vec3(pxHit.normal.x, pxHit.normal.y, pxHit.normal.z);
        hit.distance = pxHit.distance;
        hit.userData = pxHit.actor;
        return true;
    }

    return false;
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
    return m_Scene;
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

#endif // USE_PHYSX
