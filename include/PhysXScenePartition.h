#pragma once

#include <vector>
#include <memory>

// Forward declarations
namespace physx {
    class PxPhysics;
    class PxScene;
}

class PhysXSoftBody;

/**
 * @brief Wrapper for a single PhysX scene partition
 * 
 * Manages a PhysX scene and its associated soft bodies for parallel simulation.
 */
#ifdef USE_PHYSX
class PhysXScenePartition {
public:
    /**
     * @brief Constructor
     * @param physics PhysX physics instance
     * @param partitionIndex Index of this partition
     */
    PhysXScenePartition(physx::PxPhysics* physics, int partitionIndex);
    
    /**
     * @brief Destructor
     */
    ~PhysXScenePartition();
    
    /**
     * @brief Add soft body to this scene
     */
    void AddSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Remove soft body from this scene
     */
    void RemoveSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Simulate this scene
     * @param deltaTime Time step
     */
    void Simulate(float deltaTime);
    
    /**
     * @brief Fetch simulation results
     * @param block Whether to block until results are ready
     */
    void FetchResults(bool block = true);
    
    /**
     * @brief Get number of soft bodies in this partition
     */
    size_t GetSoftBodyCount() const { return m_SoftBodies.size(); }
    
    /**
     * @brief Get partition index
     */
    int GetPartitionIndex() const { return m_PartitionIndex; }
    
    /**
     * @brief Get underlying PhysX scene
     */
    physx::PxScene* GetPxScene() { return m_Scene; }

private:
    physx::PxScene* m_Scene;
    std::vector<PhysXSoftBody*> m_SoftBodies;
    int m_PartitionIndex;
};
#endif
