#pragma once

#include "IPhysicsBackend.h"
#include "Math/Vec3.h"
#include "Math/Quat.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <map>
#include <string>

#include <cstdint>

// PhysX forward declarations
namespace physx {
    class PxFoundation;
    class PxPhysics;
    class PxScene;
    class PxDefaultCpuDispatcher;
    class PxPvd;
    class PxMaterial;
    class PxDefaultAllocator;
    class PxDefaultErrorCallback;
    class PxCudaContextManager;
    class PxSimulationEventCallback;
    class PxContactModifyCallback;
    class PxContactModifyPair;
    class PxActor;
    class PxRigidBody;
    class PxTransform;
    
    // Structs
    struct PxConstraintInfo;
    struct PxContactPairHeader;
    struct PxContactPair;
    struct PxTriggerPair;
}

class IPhysicsRigidBody;
class IPhysicsCharacterController;
class IPhysicsSoftBody;
class IPhysicsCloth;
class PhysXVehicle;

/**
 * @brief Interface for listening to contact modification events
 */
class IPhysXContactModifyListener {
public:
    virtual ~IPhysXContactModifyListener() = default;
    
    /**
     * @brief Called when contacts are being modified
     * @param pairs Array of contact pairs to modify
     * @param count Number of pairs
     */
    virtual void OnContactModify(physx::PxContactModifyPair* const pairs, uint32_t count) = 0;
};

/**
 * @brief Physics simulation statistics
 */
struct PhysicsStats {
    uint32_t activeRigidBodies = 0;
    uint32_t staticRigidBodies = 0;
    uint32_t kinematicRigidBodies = 0;
    uint32_t dynamicRigidBodies = 0; // Total dynamic (active + sleeping)
    uint32_t aggregates = 0;
    uint32_t articulations = 0;
    uint32_t constraints = 0;
    uint32_t broadPhaseAdds = 0;
    uint32_t broadPhaseRemoves = 0;
    uint32_t narrowPhaseTouches = 0; // Pairs in contact
    uint32_t lostTouches = 0;
    uint32_t lostPairs = 0;
    size_t gpuMemoryUsageMB = 0;
    uint32_t activeScenes = 0;
};

/**
 * @brief PhysX implementation of physics backend
 * 
 * Provides physics simulation using NVIDIA PhysX SDK 5.x
 * Supports CPU and GPU acceleration (CUDA)
 */
class PhysXBackend : public IPhysicsBackend {
public:
    PhysXBackend();
    ~PhysXBackend() override;

    // IPhysicsBackend implementation
    void Initialize(const Vec3& gravity) override;
    void Shutdown() override;
    void Update(float deltaTime, int subSteps = 1) override;
    
    // Scene Management
    physx::PxScene* CreateScene(const std::string& name, const Vec3& gravity);
    void ReleaseScene(const std::string& name);
    physx::PxScene* GetScene(const std::string& name);
    void SetActiveScene(const std::string& name);
    physx::PxScene* GetActiveScene() const { return m_ActiveScene; }

    // Profiling
    PhysicsStats GetStatistics() const;
    
    // Time tracking
    double GetSimulationTime() const { return m_SimulationTime; }
    void SetGravity(const Vec3& gravity) override;
    Vec3 GetGravity() const override;
    bool Raycast(const Vec3& from, const Vec3& to, PhysicsRaycastHit& hit, uint32_t filter = ~0u) override;
    int GetNumRigidBodies() const override;
    void SetDebugDrawEnabled(bool enabled) override;
    bool IsDebugDrawEnabled() const override;
    const char* GetBackendName() const override { return "PhysX 5.x"; }
    void* GetNativeWorld() override;
    void ApplyImpulse(void* userData, const Vec3& impulse, const Vec3& point) override;
    
    int OverlapSphere(const Vec3& center, float radius, std::vector<void*>& results, uint32_t filter = ~0u) override;
    int OverlapBox(const Vec3& center, const Vec3& halfExtents, const Quat& rotation, std::vector<void*>& results, uint32_t filter = ~0u) override;
    int OverlapCapsule(const Vec3& center, float radius, float halfHeight, const Quat& rotation, std::vector<void*>& results, uint32_t filter = ~0u) override;

    // PhysX-specific methods
    physx::PxPhysics* GetPhysics() const { return m_Physics; }
    physx::PxScene* GetScene() const { return m_ActiveScene; } // Deprecated: returns active scene
    physx::PxMaterial* GetDefaultMaterial() const { return m_DefaultMaterial; }
    bool IsGpuEnabled() const { return m_CudaContextManager != nullptr; }
    
    // GPU-specific methods
    physx::PxCudaContextManager* GetCudaContextManager() const { return m_CudaContextManager; }
    bool IsGpuSoftBodySupported() const;
    void GetGpuDeviceProperties(int& computeCapability, size_t& totalMemoryMB, size_t& freeMemoryMB) const;
    size_t GetGpuMemoryUsageMB() const;
    
    // Rigid body optimization
    bool IsGpuRigidBodySupported() const;
    void GetActiveRigidBodies(std::vector<IPhysicsRigidBody*>& outBodies);

    // Registration methods (called by physics components)
    void RegisterRigidBody(IPhysicsRigidBody* body);
    void UnregisterRigidBody(IPhysicsRigidBody* body);
    void RegisterCharacterController(IPhysicsCharacterController* controller);
    void UnregisterCharacterController(IPhysicsCharacterController* controller);
    void RegisterSoftBody(IPhysicsSoftBody* softBody);
    void UnregisterSoftBody(IPhysicsSoftBody* softBody);
    void RegisterCloth(class IPhysicsCloth* cloth);
    void UnregisterCloth(class IPhysicsCloth* cloth);

    void RegisterVehicle(PhysXVehicle* vehicle);
    void UnregisterVehicle(PhysXVehicle* vehicle);

    void RegisterArticulation(class PhysXArticulation* articulation);
    void UnregisterArticulation(class PhysXArticulation* articulation);

    void RegisterAggregate(class PhysXAggregate* aggregate);
    void UnregisterAggregate(class PhysXAggregate* aggregate);

    void RegisterFluidVolume(class PhysXFluidVolume* volume);
    void UnregisterFluidVolume(class PhysXFluidVolume* volume);
    class PhysXFluidVolume* GetFluidVolume(IPhysicsRigidBody* body);

    // Global Collision Callback
    using GlobalCollisionCallback = std::function<void(IPhysicsRigidBody*, IPhysicsRigidBody*, const Vec3&, const Vec3&, float)>;
    void SetGlobalCollisionCallback(GlobalCollisionCallback callback);

    // Contact Modification
    void RegisterContactModifyListener(IPhysXContactModifyListener* listener);
    void UnregisterContactModifyListener(IPhysXContactModifyListener* listener);

private:
    friend class PhysXCollisionCallback; // Allow callback to access member
    
    GlobalCollisionCallback m_GlobalCollisionCallback;

    // PhysX core objects
    physx::PxFoundation* m_Foundation;
    physx::PxPhysics* m_Physics;
    std::map<std::string, physx::PxScene*> m_Scenes;
    physx::PxScene* m_ActiveScene;
    
    physx::PxDefaultCpuDispatcher* m_Dispatcher;
    physx::PxPvd* m_Pvd; // PhysX Visual Debugger
    physx::PxCudaContextManager* m_CudaContextManager; // CUDA Context Manager
    physx::PxMaterial* m_DefaultMaterial;

    // Helper to create a scene with current settings
    physx::PxScene* CreateSceneInternal(const std::string& name, const Vec3& gravity);

    // Custom allocator and error callback
    physx::PxDefaultAllocator* m_Allocator;
    physx::PxDefaultErrorCallback* m_ErrorCallback;

    // Registered components
    std::vector<IPhysicsRigidBody*> m_RigidBodies;
    std::vector<IPhysicsCharacterController*> m_CharacterControllers;
    std::vector<IPhysicsSoftBody*> m_SoftBodies;
    std::vector<class IPhysicsCloth*> m_Cloths;
    std::vector<PhysXVehicle*> m_Vehicles;
    std::vector<class PhysXArticulation*> m_Articulations;
    std::vector<class PhysXAggregate*> m_Aggregates;
    std::unordered_map<IPhysicsRigidBody*, class PhysXFluidVolume*> m_FluidVolumeMap;
    std::vector<IPhysXContactModifyListener*> m_ContactModifyListeners;

    // State
    bool m_Initialized;
    bool m_DebugDrawEnabled;
    Vec3 m_Gravity;
    double m_SimulationTime = 0.0;
    

    PhysicsStats m_LastFrameStats;
    
    // GPU tracking
    mutable size_t m_GpuMemoryUsageMB;

    // Helper methods
    void InitializePhysXVisualDebugger();
    void UpdateCharacterControllers(float deltaTime);

    // Collision Callback
    class PhysXCollisionCallback;
    PhysXCollisionCallback* m_CollisionCallback;

    // Contact Modification Callback
    class PhysXContactModifyCallback;
    PhysXContactModifyCallback* m_ContactModifyCallback;
};
