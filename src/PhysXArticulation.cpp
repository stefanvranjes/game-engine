#include "PhysXArticulation.h"
#include "PhysXBackend.h"
#include <iostream>

#ifdef USE_PHYSX
#include <PxPhysicsAPI.h>

using namespace physx;

PhysXArticulation::PhysXArticulation(PhysXBackend* backend)
    : m_Backend(backend)
    , m_Articulation(nullptr)
{
}

PhysXArticulation::~PhysXArticulation() {
    if (m_Articulation) {
        m_Articulation->release();
    }
    // Links are destroyed by articulation
    for (auto* link : m_Links) {
        delete link;
    }
    m_Links.clear();
}

void PhysXArticulation::Initialize(bool fixBase) {
    if (!m_Backend || !m_Backend->GetPhysics()) return;

    m_Articulation = m_Backend->GetPhysics()->createArticulationReducedCoordinate();
    if (!m_Articulation) {
        std::cerr << "Failed to create PhysX Articulation Reduced Coordinate!" << std::endl;
        return;
    }

    m_Articulation->setArticulationFlag(PxArticulationFlag::eFIX_BASE, fixBase);
    m_Articulation->setSolverIterationCounts(8, 1); // Default higher iterations for stability
}

PhysXArticulationLink* PhysXArticulation::AddLink(PhysXArticulationLink* parent, const Vec3& position, const Quat& rotation) {
    if (!m_Articulation) return nullptr;

    PxArticulationLink* pxParent = parent ? static_cast<PxArticulationLink*>(parent->GetNativeBody()) : nullptr;
    
    PxTransform pose(PxVec3(position.x, position.y, position.z), PxQuat(rotation.x, rotation.y, rotation.z, rotation.w));
    
    PxArticulationLink* pxLink = m_Articulation->createLink(pxParent, pose);
    if (!pxLink) {
        std::cerr << "Failed to create Articulation Link!" << std::endl;
        return nullptr;
    }

    PhysXArticulationLink* linkWrapper = new PhysXArticulationLink(pxLink, this);
    m_Links.push_back(linkWrapper);
    
    return linkWrapper;
}

void PhysXArticulation::AddToScene(PxScene* scene) {
    if (m_Articulation && scene) {
        scene->addArticulation(*m_Articulation);
    }
}

void PhysXArticulation::RemoveFromScene(PxScene* scene) {
    if (m_Articulation && scene) {
        scene->removeArticulation(*m_Articulation);
    }
}

void PhysXArticulation::SetSolverIterationCounts(uint32_t minPositionIters, uint32_t minVelocityIters) {
    if (m_Articulation) {
        m_Articulation->setSolverIterationCounts(minPositionIters, minVelocityIters);
    }
}

void PhysXArticulation::SetSleepThreshold(float threshold) {
    if (m_Articulation) {
        m_Articulation->setSleepThreshold(threshold);
    }
}

void PhysXArticulation::WakeUp() {
    if (m_Articulation) m_Articulation->wakeUp();
}

void PhysXArticulation::PutToSleep() {
    if (m_Articulation) m_Articulation->putToSleep();
}

bool PhysXArticulation::IsSleeping() const {
    return m_Articulation && m_Articulation->isSleeping();
}

#endif // USE_PHYSX
