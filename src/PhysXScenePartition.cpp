#include "PhysXScenePartition.h"
#include "PhysXSoftBody.h"
#ifdef USE_PHYSX
#include <PxPhysicsAPI.h>
#include <iostream>
#include <algorithm>

PhysXScenePartition::PhysXScenePartition(physx::PxPhysics* physics, int partitionIndex)
    : m_Scene(nullptr)
    , m_PartitionIndex(partitionIndex)
{
    if (!physics) {
        std::cerr << "PhysXScenePartition: Invalid physics instance" << std::endl;
        return;
    }
    
    // Create scene descriptor
    physx::PxSceneDesc sceneDesc(physics->getTolerancesScale());
    
    // Set gravity
    sceneDesc.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
    
    // Create CPU dispatcher (single thread per scene)
    sceneDesc.cpuDispatcher = physx::PxDefaultCpuDispatcherCreate(1);
    
    // Set filter shader
    sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;
    
    // Enable CCD (Continuous Collision Detection)
    sceneDesc.flags |= physx::PxSceneFlag::eENABLE_CCD;
    
    // Create scene
    m_Scene = physics->createScene(sceneDesc);
    
    if (m_Scene) {
        std::cout << "Created PhysX scene partition " << partitionIndex << std::endl;
    } else {
        std::cerr << "Failed to create PhysX scene partition " << partitionIndex << std::endl;
    }
}

PhysXScenePartition::~PhysXScenePartition() {
    if (m_Scene) {
        // Remove all soft bodies
        m_SoftBodies.clear();
        
        // Release scene
        m_Scene->release();
        m_Scene = nullptr;
        
        std::cout << "Destroyed PhysX scene partition " << m_PartitionIndex << std::endl;
    }
}

void PhysXScenePartition::AddSoftBody(PhysXSoftBody* softBody) {
    if (!softBody) {
        return;
    }
    
    // Check if already added
    auto it = std::find(m_SoftBodies.begin(), m_SoftBodies.end(), softBody);
    if (it != m_SoftBodies.end()) {
        return;
    }
    
    m_SoftBodies.push_back(softBody);
    softBody->SetScenePartition(m_PartitionIndex);
    
    std::cout << "Added soft body to partition " << m_PartitionIndex 
              << " (total: " << m_SoftBodies.size() << ")" << std::endl;
}

void PhysXScenePartition::RemoveSoftBody(PhysXSoftBody* softBody) {
    auto it = std::find(m_SoftBodies.begin(), m_SoftBodies.end(), softBody);
    if (it != m_SoftBodies.end()) {
        m_SoftBodies.erase(it);
        std::cout << "Removed soft body from partition " << m_PartitionIndex 
                  << " (remaining: " << m_SoftBodies.size() << ")" << std::endl;
    }
}

void PhysXScenePartition::Simulate(float deltaTime) {
    if (!m_Scene) {
        return;
    }
    
    // Start simulation
    m_Scene->simulate(deltaTime);
}

void PhysXScenePartition::FetchResults(bool block) {
    if (!m_Scene) {
        return;
    }
    
    // Fetch results
    m_Scene->fetchResults(block);
}
#endif
