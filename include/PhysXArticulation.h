#pragma once

#include "PhysXArticulationLink.h"
#include <vector>

#ifdef USE_PHYSX
namespace physx {
    class PxArticulationReducedCoordinate;
    class PxPhysics;
    class PxScene;
}

class PhysXBackend;

class PhysXArticulation {
public:
    PhysXArticulation(PhysXBackend* backend);
    ~PhysXArticulation();

    void Initialize(bool fixBase = false);
    
    // Structure creation
    PhysXArticulationLink* AddLink(PhysXArticulationLink* parent, const Vec3& position, const Quat& rotation);
    
    // Runtime
    void AddToScene(physx::PxScene* scene);
    void RemoveFromScene(physx::PxScene* scene);
    
    // Properties
    void SetSolverIterationCounts(uint32_t minPositionIters, uint32_t minVelocityIters);
    void SetSleepThreshold(float threshold);
    void WakeUp();
    void PutToSleep();
    bool IsSleeping() const;

    // Accessors
    std::vector<PhysXArticulationLink*> GetLinks() const { return m_Links; }
    physx::PxArticulationReducedCoordinate* GetNativeArticulation() const { return m_Articulation; }

private:
    PhysXBackend* m_Backend;
    physx::PxArticulationReducedCoordinate* m_Articulation;
    std::vector<PhysXArticulationLink*> m_Links;
};
#endif // USE_PHYSX
