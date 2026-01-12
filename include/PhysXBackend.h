#pragma once

#include "IPhysicsBackend.h"
#include "Math/Vec3.h"
#include "Math/Quat.h"
#include <memory>
#include <vector>

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
    class PxContactModifyCallback;
    class PxContactModifyPair;
}

class IPhysicsRigidBody;
class IPhysicsCharacterController;
class IPhysicsSoftBody;
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
    virtual void OnContactModify(physx::PxContactModifyPair* const pairs, physx::PxU32 count) = 0;
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
    void SetGravity(const Vec3& gravity) override;
    Vec3 GetGravity() const override;
    bool Raycast(const Vec3& from, const Vec3& to, RaycastHit& hit, uint32_t filter = ~0u) override;
    int GetNumRigidBodies() const override;
    void SetDebugDrawEnabled(bool enabled) override;
    bool IsDebugDrawEnabled() const override;
    const char* GetBackendName() const override { return "PhysX 5.x"; }
    const char* GetBackendName() const override { return "PhysX 5.x"; }
    void* GetNativeWorld() override;
    void ApplyImpulse(void* userData, const Vec3& impulse, const Vec3& point) override;
    
    int OverlapSphere(const Vec3& center, float radius, std::vector<void*>& results, uint32_t filter = ~0u) override;
    int OverlapBox(const Vec3& center, const Vec3& halfExtents, const Quat& rotation, std::vector<void*>& results, uint32_t filter = ~0u) override;
    int OverlapCapsule(const Vec3& center, float radius, float halfHeight, const Quat& rotation, std::vector<void*>& results, uint32_t filter = ~0u) override;

    // PhysX-specific methods
    physx::PxPhysics* GetPhysics() const { return m_Physics; }
    physx::PxScene* GetScene() const { return m_Scene; }
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

    // Contact Modification
    void RegisterContactModifyListener(IPhysXContactModifyListener* listener);
    void UnregisterContactModifyListener(IPhysXContactModifyListener* listener);

private:
    // PhysX core objects
    physx::PxFoundation* m_Foundation;
    physx::PxPhysics* m_Physics;
    physx::PxScene* m_Scene;
    physx::PxDefaultCpuDispatcher* m_Dispatcher;
    physx::PxPvd* m_Pvd; // PhysX Visual Debugger
    physx::PxCudaContextManager* m_CudaContextManager; // CUDA Context Manager
    physx::PxMaterial* m_DefaultMaterial;

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
    std::vector<IPhysXContactModifyListener*> m_ContactModifyListeners;

    // State
    bool m_Initialized;
    bool m_DebugDrawEnabled;
    Vec3 m_Gravity;
    
    // GPU tracking
    mutable size_t m_GpuMemoryUsageMB;

    // Helper methods
    void InitializePhysXVisualDebugger();
    void UpdateCharacterControllers(float deltaTime);

    // Collision Callback
    class PhysXCollisionCallback : public physx::PxSimulationEventCallback {
    public:
        PhysXCollisionCallback(PhysXBackend* backend) : m_Backend(backend) {}
        void onConstraintBreak(physx::PxConstraintInfo* constraints, physx::PxU32 count) override {}
        void onWake(physx::PxActor** actors, physx::PxU32 count) override {}
        void onSleep(physx::PxActor** actors, physx::PxU32 count) override {}
        void onContact(const physx::PxContactPairHeader& pairHeader, const physx::PxContactPair* pairs, physx::PxU32 nbPairs) override;
        void onTrigger(physx::PxTriggerPair* pairs, physx::PxU32 count) override {}
        void onAdvance(const physx::PxRigidBody* const* bodyBuffer, const physx::PxTransform* poseBuffer, const physx::PxU32 count) override {}
    private:
        PhysXBackend* m_Backend;
    };

    PhysXCollisionCallback* m_CollisionCallback;

    // Contact Modification Callback
    class PhysXContactModifyCallback : public physx::PxContactModifyCallback {
    public:
        PhysXContactModifyCallback(PhysXBackend* backend) : m_Backend(backend) {}
        void onContactModify(physx::PxContactModifyPair* const pairs, physx::PxU32 count) override;
    private:
        PhysXBackend* m_Backend;
    };

    PhysXContactModifyCallback* m_ContactModifyCallback;
};

// Custom filter shader to enable simulation callbacks
physx::PxFilterFlags PhysXSimulationFilterShader(
    physx::PxFilterObjectAttributes attributes0, physx::PxFilterData filterData0,
    physx::PxFilterObjectAttributes attributes1, physx::PxFilterData filterData1,
    physx::PxPairFlags& pairFlags, const void* constantBlock, physx::PxU32 constantBlockSize);
