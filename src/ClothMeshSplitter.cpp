#include "ClothMeshSplitter.h"
#include "ClothTearPattern.h"
#include "SpatialHashGrid.h"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <iostream>

ClothMeshSplitter::SplitResult ClothMeshSplitter::SplitAtParticle(
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices,
    int tearParticle)
{
    SplitResult result;
    
    if (tearParticle < 0 || tearParticle >= static_cast<int>(positions.size())) {
        std::cerr << "Invalid tear particle index" << std::endl;
        return result;
    }
    
    // Find connected components after removing tear particle
    std::vector<int> removedParticles = { tearParticle };
    std::vector<std::vector<int>> components;
    FindConnectedComponents(indices, positions.size(), removedParticles, components);
    
    if (components.size() < 2) {
        std::cerr << "Mesh didn't split into multiple pieces" << std::endl;
        return result;
    }
    
    // Build piece 1 from first component
    BuildMeshPiece(
        positions,
        indices,
        components[0],
        result.piece1Positions,
        result.piece1Indices,
        result.piece1OriginalParticles
    );
    
    // Build piece 2 from second component (merge remaining if more than 2)
    std::vector<int> piece2Component = components[1];
    for (size_t i = 2; i < components.size(); ++i) {
        piece2Component.insert(
            piece2Component.end(),
            components[i].begin(),
            components[i].end()
        );
    }
    
    BuildMeshPiece(
        positions,
        indices,
        piece2Component,
        result.piece2Positions,
        result.piece2Indices,
        result.piece2OriginalParticles
    );
    
    // Mark tear edge
    result.tearEdgeParticles.push_back(tearParticle);
    
    result.success = true;
    
    std::cout << "Split mesh at particle " << tearParticle 
              << " into " << result.piece1Positions.size() << " and " 
              << result.piece2Positions.size() << " particles" << std::endl;
    
    return result;
}

ClothMeshSplitter::SplitResult ClothMeshSplitter::SplitAlongLine(
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices,
    const Vec3& start,
    const Vec3& end)
{
    SplitResult result;
    
    Vec3 lineDir = end - start;
    float lineLength = lineDir.Length();
    
    if (lineLength < 0.001f) {
        std::cerr << "Line too short for splitting" << std::endl;
        return result;
    }
    
    lineDir = lineDir * (1.0f / lineLength);
    
    // Find particles along the line
    std::vector<int> lineParticles;
    for (size_t i = 0; i < positions.size(); ++i) {
        Vec3 toParticle = positions[i] - start;
        float projection = toParticle.Dot(lineDir);
        
        if (projection >= 0.0f && projection <= lineLength) {
            Vec3 closestPoint = start + lineDir * projection;
            float distance = (positions[i] - closestPoint).Length();
            
            if (distance < 0.15f) { // 15cm threshold
                lineParticles.push_back(i);
            }
        }
    }
    
    if (lineParticles.empty()) {
        std::cerr << "No particles found along line" << std::endl;
        return result;
    }
    
    // Find connected components after removing line particles
    std::vector<std::vector<int>> components;
    FindConnectedComponents(indices, positions.size(), lineParticles, components);
    
    if (components.size() < 2) {
        std::cerr << "Line didn't split mesh into multiple pieces" << std::endl;
        return result;
    }
    
    // Build pieces
    BuildMeshPiece(
        positions,
        indices,
        components[0],
        result.piece1Positions,
        result.piece1Indices,
        result.piece1OriginalParticles
    );
    
    std::vector<int> piece2Component = components[1];
    for (size_t i = 2; i < components.size(); ++i) {
        piece2Component.insert(
            piece2Component.end(),
            components[i].begin(),
            components[i].end()
        );
    }
    
    BuildMeshPiece(
        positions,
        indices,
        piece2Component,
        result.piece2Positions,
        result.piece2Indices,
        result.piece2OriginalParticles
    );
    
    result.tearEdgeParticles = lineParticles;
    result.success = true;
    
    std::cout << "Split mesh along line, removed " << lineParticles.size() 
              << " particles" << std::endl;
    
    return result;
}

ClothMeshSplitter::SplitResult ClothMeshSplitter::SplitWithPattern(
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices,
    std::shared_ptr<ClothTearPattern> pattern,
    const Vec3& position,
    const Vec3& direction,
    float scale,
    const SpatialHashGrid* spatialGrid)
{
    SplitResult result;
    
    if (!pattern) {
        std::cerr << "Invalid pattern for splitting" << std::endl;
        return result;
    }
    
    std::cout << "Splitting cloth with pattern '" << pattern->GetName() 
              << "' at position (" << position.x << ", " << position.y << ", " << position.z << ")"
              << std::endl;
    
    // Get affected particles from pattern (with spatial grid optimization if available)
    std::vector<int> affectedParticles;
    if (spatialGrid) {
        affectedParticles = pattern->GetAffectedParticles(
            positions,
            position,
            direction,
            scale,
            spatialGrid
        );
    } else {
        affectedParticles = pattern->GetAffectedParticles(
            positions,
            position,
            direction,
            scale
        );
    }
    
    if (affectedParticles.empty()) {
        std::cerr << "Pattern did not affect any particles" << std::endl;
        return result;
    }
    
    std::cout << "Pattern affected " << affectedParticles.size() << " particles" << std::endl;
    
    // Find connected components after removing affected particles
    std::vector<std::vector<int>> components;
    FindConnectedComponents(indices, positions.size(), affectedParticles, components);
    
    if (components.size() < 2) {
        std::cerr << "Pattern didn't split mesh into multiple pieces (found " 
                  << components.size() << " component)" << std::endl;
        return result;
    }
    
    std::cout << "Found " << components.size() << " connected components" << std::endl;
    
    // Build piece 1 from largest component
    BuildMeshPiece(
        positions,
        indices,
        components[0],
        result.piece1Positions,
        result.piece1Indices,
        result.piece1OriginalParticles
    );
    
    // Build piece 2 from second largest component
    // If there are more than 2 components, merge smaller ones into piece 2
    std::vector<int> piece2Component = components[1];
    for (size_t i = 2; i < components.size(); ++i) {
        piece2Component.insert(
            piece2Component.end(),
            components[i].begin(),
            components[i].end()
        );
    }
    
    BuildMeshPiece(
        positions,
        indices,
        piece2Component,
        result.piece2Positions,
        result.piece2Indices,
        result.piece2OriginalParticles
    );
    
    // Store tear edge particles for visualization
    result.tearEdgeParticles = affectedParticles;
    result.success = true;
    
    std::cout << "Successfully split mesh with pattern into " 
              << result.piece1Positions.size() << " and " 
              << result.piece2Positions.size() << " particles" << std::endl;
    
    return result;
}

void ClothMeshSplitter::FindConnectedComponents(
    const std::vector<int>& indices,
    int particleCount,
    const std::vector<int>& removedParticles,
    std::vector<std::vector<int>>& components)
{
    components.clear();
    
    // Build adjacency graph (excluding removed particles)
    std::vector<std::unordered_set<int>> adjacency(particleCount);
    std::unordered_set<int> removedSet(removedParticles.begin(), removedParticles.end());
    
    for (size_t i = 0; i < indices.size(); i += 3) {
        int i0 = indices[i];
        int i1 = indices[i + 1];
        int i2 = indices[i + 2];
        
        // Skip triangles with removed particles
        if (removedSet.count(i0) || removedSet.count(i1) || removedSet.count(i2)) {
            continue;
        }
        
        // Add edges
        adjacency[i0].insert(i1);
        adjacency[i0].insert(i2);
        adjacency[i1].insert(i0);
        adjacency[i1].insert(i2);
        adjacency[i2].insert(i0);
        adjacency[i2].insert(i1);
    }
    
    // Find connected components using DFS
    std::vector<bool> visited(particleCount, false);
    
    for (int i = 0; i < particleCount; ++i) {
        if (visited[i] || removedSet.count(i)) {
            continue;
        }
        
        // DFS to find component
        std::vector<int> component;
        std::vector<int> stack = { i };
        
        while (!stack.empty()) {
            int current = stack.back();
            stack.pop_back();
            
            if (visited[current]) {
                continue;
            }
            
            visited[current] = true;
            component.push_back(current);
            
            for (int neighbor : adjacency[current]) {
                if (!visited[neighbor] && !removedSet.count(neighbor)) {
                    stack.push_back(neighbor);
                }
            }
        }
        
        if (!component.empty()) {
            components.push_back(component);
        }
    }
    
    std::cout << "Found " << components.size() << " connected components" << std::endl;
}

void ClothMeshSplitter::BuildMeshPiece(
    const std::vector<Vec3>& originalPositions,
    const std::vector<int>& originalIndices,
    const std::vector<int>& component,
    std::vector<Vec3>& outPositions,
    std::vector<int>& outIndices,
    std::vector<int>& outOriginalMap)
{
    outPositions.clear();
    outIndices.clear();
    outOriginalMap.clear();
    
    // Create particle map (old index -> new index)
    std::unordered_map<int, int> particleMap;
    std::unordered_set<int> componentSet(component.begin(), component.end());
    
    for (int oldIndex : component) {
        int newIndex = static_cast<int>(outPositions.size());
        particleMap[oldIndex] = newIndex;
        outPositions.push_back(originalPositions[oldIndex]);
        outOriginalMap.push_back(oldIndex);
    }
    
    // Remap triangles
    for (size_t i = 0; i < originalIndices.size(); i += 3) {
        int i0 = originalIndices[i];
        int i1 = originalIndices[i + 1];
        int i2 = originalIndices[i + 2];
        
        // Only include triangles where all vertices are in component
        if (componentSet.count(i0) && componentSet.count(i1) && componentSet.count(i2)) {
            outIndices.push_back(particleMap[i0]);
            outIndices.push_back(particleMap[i1]);
            outIndices.push_back(particleMap[i2]);
        }
    }
    
    std::cout << "Built mesh piece: " << outPositions.size() << " particles, " 
              << (outIndices.size() / 3) << " triangles" << std::endl;
}

void ClothMeshSplitter::RemapIndices(
    const std::vector<int>& originalIndices,
    const std::vector<int>& particleMap,
    std::vector<int>& newIndices)
{
    newIndices.clear();
    newIndices.reserve(originalIndices.size());
    
    for (int oldIndex : originalIndices) {
        if (oldIndex >= 0 && oldIndex < static_cast<int>(particleMap.size())) {
            newIndices.push_back(particleMap[oldIndex]);
        }
    }
}
