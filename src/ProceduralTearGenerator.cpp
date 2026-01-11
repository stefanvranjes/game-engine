#include "ProceduralTearGenerator.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <iostream>

ProceduralTearGenerator::TearPath ProceduralTearGenerator::GenerateTearPath(
    const SoftBodyTearSystem::TearInfo& initialTear,
    const Vec3* currentPositions,
    const Vec3* restPositions,
    const int* tetrahedronIndices,
    int tetrahedronCount,
    const SoftBodyTearSystem::StressData* stressData,
    float energyThreshold,
    int maxPropagationSteps)
{
    TearPath path;
    path.isComplete = false;
    path.totalEnergy = 0.0f;

    // Start from the initial tear
    int currentTet = initialTear.tetrahedronIndex;
    path.tetrahedronSequence.push_back(currentTet);
    path.pathPoints.push_back(initialTear.tearPosition);

    // Calculate initial propagation direction from principal stress
    path.propagationDirection = CalculatePrincipalStressDirection(
        currentTet, currentPositions, restPositions, tetrahedronIndices
    );

    // Track visited tetrahedra
    std::vector<bool> visitedTets(tetrahedronCount, false);
    visitedTets[currentTet] = true;

    // Propagate tear along stress lines
    for (int step = 0; step < maxPropagationSteps; ++step) {
        // Calculate energy at current position
        float stepEnergy = CalculateTearEnergy(
            currentTet, stressData[currentTet],
            currentPositions, restPositions, tetrahedronIndices
        );

        path.totalEnergy += stepEnergy;

        // Check if we have enough energy to continue
        if (stepEnergy < energyThreshold) {
            path.isComplete = true;
            break;
        }

        // Find next tetrahedron to propagate into
        int nextTet = FindNextTetrahedron(
            currentTet, tetrahedronIndices, tetrahedronCount,
            stressData, visitedTets, path.propagationDirection
        );

        if (nextTet == -1) {
            // No more neighbors to propagate into
            path.isComplete = true;
            break;
        }

        // Move to next tetrahedron
        currentTet = nextTet;
        visitedTets[currentTet] = true;
        path.tetrahedronSequence.push_back(currentTet);

        // Calculate center of current tet for path point
        const int* tet = &tetrahedronIndices[currentTet * 4];
        Vec3 center = CalculateTetrahedronCenter(
            currentPositions[tet[0]], currentPositions[tet[1]],
            currentPositions[tet[2]], currentPositions[tet[3]]
        );
        path.pathPoints.push_back(center);

        // Update propagation direction
        path.propagationDirection = CalculatePrincipalStressDirection(
            currentTet, currentPositions, restPositions, tetrahedronIndices
        );
    }

    if (!path.isComplete) {
        std::cout << "Tear propagation reached max steps (" << maxPropagationSteps << ")" << std::endl;
    }

    return path;
}

Vec3 ProceduralTearGenerator::CalculatePrincipalStressDirection(
    int tetrahedronIndex,
    const Vec3* currentPositions,
    const Vec3* restPositions,
    const int* tetrahedronIndices)
{
    const int* tet = &tetrahedronIndices[tetrahedronIndex * 4];

    // Calculate deformation gradient (simplified)
    // We'll use the average edge direction weighted by stress
    Vec3 avgDirection(0, 0, 0);
    float totalWeight = 0.0f;

    // Check all 6 edges
    const int edges[6][2] = {
        {0, 1}, {0, 2}, {0, 3},
        {1, 2}, {1, 3}, {2, 3}
    };

    for (int i = 0; i < 6; ++i) {
        int v0 = tet[edges[i][0]];
        int v1 = tet[edges[i][1]];

        // Calculate edge stress
        Vec3 currentEdge = currentPositions[v1] - currentPositions[v0];
        Vec3 restEdge = restPositions[v1] - restPositions[v0];

        float currentLength = currentEdge.Length();
        float restLength = restEdge.Length();

        if (restLength > 0.0001f) {
            float stress = currentLength / restLength;
            
            // Weight by stress (higher stress = more influence)
            if (stress > 1.0f) {  // Only consider tensile stress
                float weight = (stress - 1.0f);
                avgDirection = avgDirection + currentEdge.Normalized() * weight;
                totalWeight += weight;
            }
        }
    }

    // Normalize the result
    if (totalWeight > 0.0001f) {
        avgDirection = avgDirection * (1.0f / totalWeight);
        return avgDirection.Normalized();
    }

    // Fallback: use first edge direction
    Vec3 edge = currentPositions[tet[1]] - currentPositions[tet[0]];
    return edge.Normalized();
}

int ProceduralTearGenerator::FindNextTetrahedron(
    int currentTetIndex,
    const int* tetrahedronIndices,
    int tetrahedronCount,
    const SoftBodyTearSystem::StressData* stressData,
    const std::vector<bool>& visitedTets,
    const Vec3& preferredDirection)
{
    // Build adjacency on the fly (could be cached for performance)
    std::vector<int> neighbors;

    const int* currentTet = &tetrahedronIndices[currentTetIndex * 4];

    // Find all neighboring tetrahedra (those sharing a face)
    for (int i = 0; i < tetrahedronCount; ++i) {
        if (i == currentTetIndex || visitedTets[i]) {
            continue;
        }

        const int* otherTet = &tetrahedronIndices[i * 4];

        if (TetrahedraShareFace(currentTet, otherTet)) {
            neighbors.push_back(i);
        }
    }

    if (neighbors.empty()) {
        return -1;
    }

    // Find neighbor with highest stress that aligns with preferred direction
    int bestNeighbor = -1;
    float bestScore = -1.0f;

    for (int neighborIdx : neighbors) {
        // Calculate stress score
        const auto& stress = stressData[neighborIdx];
        float maxEdgeStress = 0.0f;
        for (int i = 0; i < 6; ++i) {
            maxEdgeStress = std::max(maxEdgeStress, stress.edgeStress[i]);
        }

        // Calculate alignment with preferred direction
        // (This is simplified - in production, you'd calculate the actual direction to neighbor)
        float alignmentScore = 1.0f;  // Placeholder

        // Combined score: stress * alignment
        float score = maxEdgeStress * alignmentScore;

        if (score > bestScore) {
            bestScore = score;
            bestNeighbor = neighborIdx;
        }
    }

    return bestNeighbor;
}

float ProceduralTearGenerator::CalculateTearEnergy(
    int tetIndex,
    const SoftBodyTearSystem::StressData& stressData,
    const Vec3* currentPositions,
    const Vec3* restPositions,
    const int* tetrahedronIndices)
{
    // Energy is proportional to stress and volume
    // Calculate average edge stress
    float avgStress = 0.0f;
    for (int i = 0; i < 6; ++i) {
        avgStress += stressData.edgeStress[i];
    }
    avgStress /= 6.0f;

    // Calculate tetrahedron volume (simplified)
    const int* tet = &tetrahedronIndices[tetIndex * 4];
    Vec3 v0 = currentPositions[tet[0]];
    Vec3 v1 = currentPositions[tet[1]];
    Vec3 v2 = currentPositions[tet[2]];
    Vec3 v3 = currentPositions[tet[3]];

    Vec3 a = v1 - v0;
    Vec3 b = v2 - v0;
    Vec3 c = v3 - v0;
    float volume = std::abs(a.Dot(b.Cross(c))) / 6.0f;

    // Energy = stress * volume
    return avgStress * volume;
}

void ProceduralTearGenerator::BuildAdjacencyList(
    const int* tetrahedronIndices,
    int tetrahedronCount,
    std::vector<std::vector<int>>& outAdjacency)
{
    outAdjacency.resize(tetrahedronCount);

    for (int i = 0; i < tetrahedronCount; ++i) {
        const int* tet1 = &tetrahedronIndices[i * 4];

        for (int j = i + 1; j < tetrahedronCount; ++j) {
            const int* tet2 = &tetrahedronIndices[j * 4];

            if (TetrahedraShareFace(tet1, tet2)) {
                outAdjacency[i].push_back(j);
                outAdjacency[j].push_back(i);
            }
        }
    }
}

Vec3 ProceduralTearGenerator::CalculateTetrahedronCenter(
    const Vec3& v0, const Vec3& v1,
    const Vec3& v2, const Vec3& v3)
{
    return (v0 + v1 + v2 + v3) * 0.25f;
}

bool ProceduralTearGenerator::TetrahedraShareFace(
    const int* tet1,
    const int* tet2)
{
    // Count shared vertices
    int sharedCount = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (tet1[i] == tet2[j]) {
                sharedCount++;
                break;
            }
        }
    }

    // Tetrahedra share a face if they have exactly 3 vertices in common
    return sharedCount == 3;
}
