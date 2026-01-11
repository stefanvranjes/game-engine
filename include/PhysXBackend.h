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
}

class IPhysicsRigidBody;
class IPhysicsCharacterController;
class IPhysicsSoftBody;
class IPhysicsCloth;

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
    void* GetNativeWorld() override;

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

    // Registration methods (called by physics components)
    void RegisterRigidBody(IPhysicsRigidBody* body);
    void UnregisterRigidBody(IPhysicsRigidBody* body);
    void RegisterCharacterController(IPhysicsCharacterController* controller);
    void UnregisterCharacterController(IPhysicsCharacterController* controller);
    void RegisterSoftBody(IPhysicsSoftBody* softBody);
    void UnregisterSoftBody(IPhysicsSoftBody* softBody);
    void RegisterCloth(class IPhysicsCloth* cloth);
    void UnregisterCloth(class IPhysicsCloth* cloth);

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

    // State
    bool m_Initialized;
    bool m_DebugDrawEnabled;
    Vec3 m_Gravity;
    
    // GPU tracking
    mutable size_t m_GpuMemoryUsageMB;

    // Helper methods
    void InitializePhysXVisualDebugger();
    void UpdateCharacterControllers(float deltaTime);
};
