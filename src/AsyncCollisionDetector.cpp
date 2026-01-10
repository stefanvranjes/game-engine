#include "AsyncCollisionDetector.h"
#include "PhysXSoftBody.h"
#include <iostream>
#include <algorithm>

AsyncCollisionDetector::AsyncCollisionDetector(WorkStealingThreadPool& threadPool, float cellSize)
    : m_ThreadPool(threadPool)
    , m_BroadPhase(cellSize)
{
}

void AsyncCollisionDetector::RegisterObject(PhysXSoftBody* object) {
    if (!object) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_ObjectsMutex);
    
    // Check if already registered
    auto it = std::find(m_Objects.begin(), m_Objects.end(), object);
    if (it == m_Objects.end()) {
        m_Objects.push_back(object);
    }
}

void AsyncCollisionDetector::UnregisterObject(PhysXSoftBody* object) {
    std::lock_guard<std::mutex> lock(m_ObjectsMutex);
    
    auto it = std::find(m_Objects.begin(), m_Objects.end(), object);
    if (it != m_Objects.end()) {
        m_Objects.erase(it);
    }
}

void AsyncCollisionDetector::DetectCollisionsAsync() {
    // Update broad-phase spatial grid
    UpdateBroadPhase();
    
    // Get potential collision pairs from spatial grid
    std::vector<std::pair<PhysXSoftBody*, PhysXSoftBody*>> potentialPairs;
    
    {
        std::lock_guard<std::mutex> lock(m_ObjectsMutex);
        
        // Query each object against the grid
        for (auto* object : m_Objects) {
            // Get AABB for this object
            // Note: This requires PhysXSoftBody to have GetAABB method
            // For now, we'll use a simple approach
            
            std::vector<PhysXSoftBody*> nearby;
            SpatialGrid<PhysXSoftBody*>::AABB queryArea;
            // Simplified: query large area around object
            // In production, use actual AABB
            queryArea.min = Vec3(-1000, -1000, -1000);
            queryArea.max = Vec3(1000, 1000, 1000);
            
            m_BroadPhase.Query(queryArea, nearby);
            
            // Create pairs
            for (auto* other : nearby) {
                if (object < other) {  // Avoid duplicate pairs
                    potentialPairs.push_back({object, other});
                }
            }
        }
    }
    
    // Submit narrow-phase collision jobs
    std::vector<std::future<void>> futures;
    
    for (const auto& pair : potentialPairs) {
        futures.push_back(m_ThreadPool.Submit([this, pair]() {
            NarrowPhaseCollision(pair.first, pair.second);
        }));
    }
    
    // Wait for all collision jobs to complete
    for (auto& future : futures) {
        future.get();
    }
}

std::vector<CollisionResult> AsyncCollisionDetector::FetchResults() {
    std::lock_guard<std::mutex> lock(m_ResultsMutex);
    return m_Results;
}

void AsyncCollisionDetector::ClearResults() {
    std::lock_guard<std::mutex> lock(m_ResultsMutex);
    m_Results.clear();
}

size_t AsyncCollisionDetector::GetObjectCount() const {
    std::lock_guard<std::mutex> lock(m_ObjectsMutex);
    return m_Objects.size();
}

void AsyncCollisionDetector::UpdateBroadPhase() {
    std::lock_guard<std::mutex> lock(m_ObjectsMutex);
    
    m_BroadPhase.Clear();
    
    for (auto* object : m_Objects) {
        // Insert object into spatial grid
        // Note: This requires PhysXSoftBody to have position/radius
        // For now, use placeholder values
        Vec3 position(0, 0, 0);  // Get from object
        float radius = 5.0f;      // Get from object
        
        m_BroadPhase.Insert(object, position, radius);
    }
}

void AsyncCollisionDetector::NarrowPhaseCollision(PhysXSoftBody* a, PhysXSoftBody* b) {
    // Perform detailed collision detection
    // This is a simplified placeholder
    // In production, use actual collision detection algorithms
    
    // For now, just create a dummy result
    CollisionResult result;
    result.objectA = a;
    result.objectB = b;
    result.contactPoint = Vec3(0, 0, 0);
    result.contactNormal = Vec3(0, 1, 0);
    result.penetrationDepth = 0.0f;
    
    // Add to results (thread-safe)
    {
        std::lock_guard<std::mutex> lock(m_ResultsMutex);
        m_Results.push_back(result);
    }
}
