#include "TearPropagationSystem.h"
#include <algorithm>
#include <iostream>

TearPropagationSystem::TearPropagationSystem()
    : m_PropagationSpeed(5.0f)  // Default: 5 tets per second
{
}

TearPropagationSystem::~TearPropagationSystem() {
}

void TearPropagationSystem::Update(float deltaTime) {
    if (m_ActiveTears.empty()) {
        return;
    }

    // Calculate how much to propagate this frame
    float propagationAmount = m_PropagationSpeed * deltaTime;

    // Update each active tear
    for (auto it = m_ActiveTears.begin(); it != m_ActiveTears.end();) {
        PropagatingTear& tear = *it;

        // Advance propagation
        tear.propagationProgress += propagationAmount;

        // Check if we've completed the current tetrahedron
        while (tear.propagationProgress >= 1.0f && tear.pathIndex < static_cast<int>(tear.fullPath.tetrahedronSequence.size())) {
            tear.propagationProgress -= 1.0f;

            // Move to next tetrahedron in path
            tear.pathIndex++;

            if (tear.pathIndex < static_cast<int>(tear.fullPath.tetrahedronSequence.size())) {
                int nextTet = tear.fullPath.tetrahedronSequence[tear.pathIndex];
                tear.currentTetrahedron = nextTet;
                tear.currentPosition = tear.fullPath.pathPoints[tear.pathIndex];
                tear.path.push_back(nextTet);

                // Notify callback
                if (m_TearCallback) {
                    m_TearCallback(nextTet, tear.currentPosition, tear.direction);
                }

                std::cout << "Tear propagated to tet " << nextTet 
                         << " (step " << tear.pathIndex << "/" << tear.fullPath.tetrahedronSequence.size() << ")" 
                         << std::endl;
            } else {
                // Reached end of path
                std::cout << "Tear propagation complete after " << tear.path.size() << " tets" << std::endl;
                break;
            }
        }

        // Check if tear is complete
        if (tear.pathIndex >= static_cast<int>(tear.fullPath.tetrahedronSequence.size()) || 
            tear.fullPath.isComplete) {
            it = m_ActiveTears.erase(it);
        } else {
            ++it;
        }
    }
}

void TearPropagationSystem::StartPropagation(
    const SoftBodyTearSystem::TearInfo& initialTear,
    const ProceduralTearGenerator::TearPath& generatedPath)
{
    PropagatingTear tear;
    tear.currentTetrahedron = initialTear.tetrahedronIndex;
    tear.currentPosition = initialTear.tearPosition;
    tear.direction = initialTear.tearNormal;
    tear.energy = generatedPath.totalEnergy;
    tear.propagationProgress = 0.0f;
    tear.fullPath = generatedPath;
    tear.pathIndex = 0;
    tear.path.push_back(initialTear.tetrahedronIndex);

    m_ActiveTears.push_back(tear);

    std::cout << "Started tear propagation with " << generatedPath.tetrahedronSequence.size() 
             << " tets in path, total energy: " << generatedPath.totalEnergy << std::endl;
}

void TearPropagationSystem::SetPropagationSpeed(float tetsPerSecond) {
    m_PropagationSpeed = std::max(0.1f, tetsPerSecond);
}

void TearPropagationSystem::ClearActiveTears() {
    m_ActiveTears.clear();
}

void TearPropagationSystem::SetTearCallback(std::function<void(int, const Vec3&, const Vec3&)> callback) {
    m_TearCallback = callback;
}
