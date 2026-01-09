#include "ClothMeshSynchronizer.h"
#include "Mesh.h"
#include "SpatialHashGrid.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// ============================================================================
// DirtyRegionTracker Implementation
// ============================================================================

DirtyRegionTracker::DirtyRegionTracker()
    : m_SpatialGrid(nullptr)
{
}

DirtyRegionTracker::~DirtyRegionTracker()
{
}

void DirtyRegionTracker::Initialize(const std::vector<Vec3>& positions, float cellSize)
{
    m_SpatialGrid = std::make_unique<SpatialHashGrid>();
    
    // Build spatial grid from particle positions
    for (size_t i = 0; i < positions.size(); ++i) {
        m_SpatialGrid->Insert(static_cast<int>(i), positions[i], cellSize);
    }
}

void DirtyRegionTracker::UpdateDirtyParticles(
    const std::vector<Vec3>& oldPositions,
    const std::vector<Vec3>& newPositions,
    float threshold)
{
    m_DirtyParticles.clear();
    
    float thresholdSq = threshold * threshold;
    
    for (size_t i = 0; i < newPositions.size() && i < oldPositions.size(); ++i) {
        Vec3 delta = newPositions[i] - oldPositions[i];
        float distSq = delta.Dot(delta);
        
        if (distSq > thresholdSq) {
            m_DirtyParticles.insert(static_cast<int>(i));
        }
    }
}

std::vector<int> DirtyRegionTracker::GetAffectedVertices(
    const VertexMapping& mapping,
    float influenceRadius) const
{
    std::unordered_set<int> affectedVertices;
    
    // For 1:1 mapping, dirty particles directly map to vertices
    if (mapping.isOneToOne) {
        for (int particleIdx : m_DirtyParticles) {
            if (particleIdx >= 0 && particleIdx < static_cast<int>(mapping.particleIndices.size())) {
                affectedVertices.insert(particleIdx);
            }
        }
    } else {
        // For LOD mapping, find all vertices influenced by dirty particles
        for (size_t vertexIdx = 0; vertexIdx < mapping.particleIndices.size(); ++vertexIdx) {
            int particleIdx = mapping.particleIndices[vertexIdx];
            
            if (m_DirtyParticles.count(particleIdx) > 0) {
                affectedVertices.insert(static_cast<int>(vertexIdx));
                continue;
            }
            
            // Check influence particles if available
            if (vertexIdx < mapping.influenceParticles.size()) {
                for (int influenceIdx : mapping.influenceParticles[vertexIdx]) {
                    if (m_DirtyParticles.count(influenceIdx) > 0) {
                        affectedVertices.insert(static_cast<int>(vertexIdx));
                        break;
                    }
                }
            }
        }
    }
    
    return std::vector<int>(affectedVertices.begin(), affectedVertices.end());
}

void DirtyRegionTracker::Clear()
{
    m_DirtyParticles.clear();
}

void DirtyRegionTracker::MarkDirty(int particleIndex)
{
    m_DirtyParticles.insert(particleIndex);
}

// ============================================================================
// ClothMeshSynchronizer Implementation
// ============================================================================

ClothMeshSynchronizer::ClothMeshSynchronizer()
    : m_Mesh(nullptr)
    , m_ParticleCount(0)
    , m_ProgressiveVertexIndex(0)
    , m_LODTransitionActive(false)
    , m_LODTransitionProgress(0.0f)
{
}

ClothMeshSynchronizer::~ClothMeshSynchronizer()
{
}

void ClothMeshSynchronizer::Initialize(
    Mesh* mesh,
    int particleCount,
    const std::vector<int>& triangleIndices)
{
    m_Mesh = mesh;
    m_ParticleCount = particleCount;
    m_TriangleIndices = triangleIndices;
    
    // Initialize default 1:1 mapping
    m_Mapping.isOneToOne = true;
    m_Mapping.particleIndices.resize(particleCount);
    for (int i = 0; i < particleCount; ++i) {
        m_Mapping.particleIndices[i] = i;
    }
    
    // Initialize dirty tracker
    m_PreviousPositions.resize(particleCount, Vec3(0, 0, 0));
    
    // Initialize cached data
    m_VertexPositions.resize(particleCount, Vec3(0, 0, 0));
    m_VertexNormals.resize(particleCount, Vec3(0, 1, 0));
    
    std::cout << "ClothMeshSynchronizer initialized with " << particleCount 
              << " particles" << std::endl;
}

void ClothMeshSynchronizer::SetVertexMapping(const VertexMapping& mapping)
{
    m_Mapping = mapping;
    
    // Resize cached data to match vertex count
    m_VertexPositions.resize(mapping.particleIndices.size());
    m_VertexNormals.resize(mapping.particleIndices.size());
    
    std::cout << "Vertex mapping updated: " 
              << (mapping.isOneToOne ? "1:1" : "LOD") 
              << " mapping with " << mapping.particleIndices.size() 
              << " vertices" << std::endl;
}

bool ClothMeshSynchronizer::Synchronize(
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>* normals,
    float deltaTime)
{
    if (!m_Mesh || positions.empty()) {
        return false;
    }
    
    // Handle LOD transition if active
    if (m_LODTransitionActive) {
        UpdateLODTransition(deltaTime);
        return !m_LODTransitionActive; // Return true when transition completes
    }
    
    // Dispatch to appropriate synchronization method
    switch (m_Config.mode) {
        case ClothSyncMode::Full:
            SynchronizeFull(positions, normals);
            return true;
            
        case ClothSyncMode::Partial:
            SynchronizePartial(positions, normals);
            return true;
            
        case ClothSyncMode::Progressive:
            SynchronizeProgressive(positions, normals, deltaTime);
            return m_ProgressiveUpdateQueue.empty();
            
        default:
            SynchronizeFull(positions, normals);
            return true;
    }
}

void ClothMeshSynchronizer::ForceFullSync(
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>* normals)
{
    SynchronizeFull(positions, normals);
}

void ClothMeshSynchronizer::MarkParticlesDirty(const std::vector<int>& particleIndices)
{
    for (int idx : particleIndices) {
        m_DirtyTracker.MarkDirty(idx);
    }
}

void ClothMeshSynchronizer::BeginLODTransition(
    const VertexMapping& newMapping,
    const std::vector<Vec3>& oldPositions,
    const std::vector<Vec3>& newPositions)
{
    if (!m_Config.enableInterpolation) {
        // Immediate transition
        SetVertexMapping(newMapping);
        SynchronizeFull(newPositions, nullptr);
        return;
    }
    
    m_LODTransitionActive = true;
    m_LODTransitionProgress = 0.0f;
    m_OldMapping = m_Mapping;
    m_NewMapping = newMapping;
    m_LODOldPositions = oldPositions;
    m_LODNewPositions = newPositions;
    
    std::cout << "Started LOD transition from " << m_OldMapping.particleIndices.size()
              << " to " << m_NewMapping.particleIndices.size() << " vertices" << std::endl;
}

void ClothMeshSynchronizer::UpdateLODTransition(float deltaTime)
{
    if (!m_LODTransitionActive) {
        return;
    }
    
    m_LODTransitionProgress += deltaTime / m_Config.interpolationDuration;
    
    if (m_LODTransitionProgress >= 1.0f) {
        // Transition complete
        m_LODTransitionProgress = 1.0f;
        SetVertexMapping(m_NewMapping);
        SynchronizeFull(m_LODNewPositions, nullptr);
        m_LODTransitionActive = false;
        std::cout << "LOD transition completed" << std::endl;
        return;
    }
    
    // Interpolate between old and new positions
    float t = m_LODTransitionProgress;
    
    // Use the new mapping size as target
    size_t vertexCount = m_NewMapping.particleIndices.size();
    
    for (size_t i = 0; i < vertexCount; ++i) {
        Vec3 oldPos(0, 0, 0);
        Vec3 newPos(0, 0, 0);
        
        // Get old position
        if (i < m_OldMapping.particleIndices.size()) {
            int oldParticleIdx = m_OldMapping.particleIndices[i];
            if (oldParticleIdx >= 0 && oldParticleIdx < static_cast<int>(m_LODOldPositions.size())) {
                oldPos = m_LODOldPositions[oldParticleIdx];
            }
        }
        
        // Get new position
        int newParticleIdx = m_NewMapping.particleIndices[i];
        if (newParticleIdx >= 0 && newParticleIdx < static_cast<int>(m_LODNewPositions.size())) {
            newPos = m_LODNewPositions[newParticleIdx];
        }
        
        // Interpolate
        Vec3 interpolatedPos = oldPos * (1.0f - t) + newPos * t;
        UpdateVertexPosition(static_cast<int>(i), interpolatedPos);
    }
    
    // Recalculate normals
    RecalculateNormals(m_VertexPositions);
    
    // Update mesh buffers
    UpdateMeshBuffers();
}

float ClothMeshSynchronizer::GetProgress() const
{
    if (m_LODTransitionActive) {
        return m_LODTransitionProgress;
    }
    
    if (m_Config.mode == ClothSyncMode::Progressive && !m_ProgressiveUpdateQueue.empty()) {
        int totalVertices = static_cast<int>(m_Mapping.particleIndices.size());
        int remaining = static_cast<int>(m_ProgressiveUpdateQueue.size());
        return 1.0f - (static_cast<float>(remaining) / static_cast<float>(totalVertices));
    }
    
    return 1.0f;
}

void ClothMeshSynchronizer::Reset()
{
    m_ProgressiveVertexIndex = 0;
    m_ProgressiveUpdateQueue.clear();
    m_DirtyTracker.Clear();
    m_LODTransitionActive = false;
    m_LODTransitionProgress = 0.0f;
}

// ============================================================================
// Private Synchronization Methods
// ============================================================================

void ClothMeshSynchronizer::SynchronizeFull(
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>* normals)
{
    size_t vertexCount = m_Mapping.particleIndices.size();
    
    // Update all vertex positions
    for (size_t i = 0; i < vertexCount; ++i) {
        Vec3 pos = InterpolatePosition(static_cast<int>(i), positions);
        UpdateVertexPosition(static_cast<int>(i), pos);
    }
    
    // Update normals
    if (normals && m_Config.attributes & VertexAttribute::Normal) {
        for (size_t i = 0; i < vertexCount; ++i) {
            int particleIdx = m_Mapping.particleIndices[i];
            if (particleIdx >= 0 && particleIdx < static_cast<int>(normals->size())) {
                UpdateVertexNormal(static_cast<int>(i), (*normals)[particleIdx]);
            }
        }
    } else if (m_Config.attributes & VertexAttribute::Normal) {
        RecalculateNormals(positions);
    }
    
    // Update mesh buffers
    UpdateMeshBuffers();
    
    // Store current positions for next frame's dirty detection
    m_PreviousPositions = positions;
}

void ClothMeshSynchronizer::SynchronizePartial(
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>* normals)
{
    // Update dirty particles
    m_DirtyTracker.UpdateDirtyParticles(
        m_PreviousPositions,
        positions,
        m_Config.changeThreshold
    );
    
    // Get affected vertices
    std::vector<int> affectedVertices = m_DirtyTracker.GetAffectedVertices(
        m_Mapping,
        m_Config.dirtyRegionRadius
    );
    
    if (affectedVertices.empty()) {
        // Nothing to update
        return;
    }
    
    // Update affected vertices
    for (int vertexIdx : affectedVertices) {
        Vec3 pos = InterpolatePosition(vertexIdx, positions);
        UpdateVertexPosition(vertexIdx, pos);
    }
    
    // Update normals for affected region
    if (m_Config.attributes & VertexAttribute::Normal) {
        if (normals) {
            for (int vertexIdx : affectedVertices) {
                int particleIdx = m_Mapping.particleIndices[vertexIdx];
                if (particleIdx >= 0 && particleIdx < static_cast<int>(normals->size())) {
                    UpdateVertexNormal(vertexIdx, (*normals)[particleIdx]);
                }
            }
        } else {
            // Recalculate normals for affected triangles
            RecalculateNormals(positions);
        }
    }
    
    // Update only affected vertices in mesh
    UpdateMeshBuffers(affectedVertices);
    
    // Store current positions
    m_PreviousPositions = positions;
    
    // Clear dirty flags
    m_DirtyTracker.Clear();
}

void ClothMeshSynchronizer::SynchronizeProgressive(
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>* normals,
    float deltaTime)
{
    // Build update queue if empty
    if (m_ProgressiveUpdateQueue.empty()) {
        size_t vertexCount = m_Mapping.particleIndices.size();
        m_ProgressiveUpdateQueue.reserve(vertexCount);
        for (size_t i = 0; i < vertexCount; ++i) {
            m_ProgressiveUpdateQueue.push_back(static_cast<int>(i));
        }
    }
    
    // Update a batch of vertices this frame
    int verticesToUpdate = std::min(
        m_Config.maxVerticesPerFrame,
        static_cast<int>(m_ProgressiveUpdateQueue.size())
    );
    
    std::vector<int> updatedVertices;
    updatedVertices.reserve(verticesToUpdate);
    
    for (int i = 0; i < verticesToUpdate; ++i) {
        int vertexIdx = m_ProgressiveUpdateQueue.back();
        m_ProgressiveUpdateQueue.pop_back();
        
        Vec3 pos = InterpolatePosition(vertexIdx, positions);
        UpdateVertexPosition(vertexIdx, pos);
        
        if (normals && m_Config.attributes & VertexAttribute::Normal) {
            int particleIdx = m_Mapping.particleIndices[vertexIdx];
            if (particleIdx >= 0 && particleIdx < static_cast<int>(normals->size())) {
                UpdateVertexNormal(vertexIdx, (*normals)[particleIdx]);
            }
        }
        
        updatedVertices.push_back(vertexIdx);
    }
    
    // Update mesh buffers for this batch
    UpdateMeshBuffers(updatedVertices);
    
    // If queue is empty, recalculate normals if needed
    if (m_ProgressiveUpdateQueue.empty()) {
        if (!normals && m_Config.attributes & VertexAttribute::Normal) {
            RecalculateNormals(positions);
            UpdateMeshBuffers();
        }
        m_PreviousPositions = positions;
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

void ClothMeshSynchronizer::UpdateVertexPosition(int vertexIndex, const Vec3& position)
{
    if (vertexIndex >= 0 && vertexIndex < static_cast<int>(m_VertexPositions.size())) {
        m_VertexPositions[vertexIndex] = position;
    }
}

void ClothMeshSynchronizer::UpdateVertexNormal(int vertexIndex, const Vec3& normal)
{
    if (vertexIndex >= 0 && vertexIndex < static_cast<int>(m_VertexNormals.size())) {
        m_VertexNormals[vertexIndex] = normal;
    }
}

void ClothMeshSynchronizer::RecalculateNormals(const std::vector<Vec3>& positions)
{
    // Reset normals
    std::fill(m_VertexNormals.begin(), m_VertexNormals.end(), Vec3(0, 0, 0));
    
    // Accumulate face normals
    for (size_t i = 0; i < m_TriangleIndices.size(); i += 3) {
        int i0 = m_TriangleIndices[i];
        int i1 = m_TriangleIndices[i + 1];
        int i2 = m_TriangleIndices[i + 2];
        
        if (i0 < 0 || i0 >= static_cast<int>(m_VertexPositions.size()) ||
            i1 < 0 || i1 >= static_cast<int>(m_VertexPositions.size()) ||
            i2 < 0 || i2 >= static_cast<int>(m_VertexPositions.size())) {
            continue;
        }
        
        Vec3 p0 = m_VertexPositions[i0];
        Vec3 p1 = m_VertexPositions[i1];
        Vec3 p2 = m_VertexPositions[i2];
        
        Vec3 edge1 = p1 - p0;
        Vec3 edge2 = p2 - p0;
        Vec3 faceNormal = edge1.Cross(edge2);
        
        m_VertexNormals[i0] = m_VertexNormals[i0] + faceNormal;
        m_VertexNormals[i1] = m_VertexNormals[i1] + faceNormal;
        m_VertexNormals[i2] = m_VertexNormals[i2] + faceNormal;
    }
    
    // Normalize
    for (Vec3& normal : m_VertexNormals) {
        float length = std::sqrt(normal.Dot(normal));
        if (length > 0.0001f) {
            normal = normal * (1.0f / length);
        } else {
            normal = Vec3(0, 1, 0);
        }
    }
}

Vec3 ClothMeshSynchronizer::InterpolatePosition(
    int vertexIndex,
    const std::vector<Vec3>& positions) const
{
    if (vertexIndex < 0 || vertexIndex >= static_cast<int>(m_Mapping.particleIndices.size())) {
        return Vec3(0, 0, 0);
    }
    
    // Simple case: 1:1 mapping or single particle influence
    if (m_Mapping.isOneToOne || 
        vertexIndex >= static_cast<int>(m_Mapping.influenceParticles.size()) ||
        m_Mapping.influenceParticles[vertexIndex].empty()) {
        
        int particleIdx = m_Mapping.particleIndices[vertexIndex];
        if (particleIdx >= 0 && particleIdx < static_cast<int>(positions.size())) {
            return positions[particleIdx];
        }
        return Vec3(0, 0, 0);
    }
    
    // Multi-particle influence: weighted average
    Vec3 result(0, 0, 0);
    float totalWeight = 0.0f;
    
    const auto& influenceParticles = m_Mapping.influenceParticles[vertexIndex];
    const auto& weights = m_Mapping.weights[vertexIndex];
    
    for (size_t i = 0; i < influenceParticles.size() && i < weights.size(); ++i) {
        int particleIdx = influenceParticles[i];
        float weight = weights[i];
        
        if (particleIdx >= 0 && particleIdx < static_cast<int>(positions.size())) {
            result = result + positions[particleIdx] * weight;
            totalWeight += weight;
        }
    }
    
    if (totalWeight > 0.0001f) {
        result = result * (1.0f / totalWeight);
    }
    
    return result;
}

void ClothMeshSynchronizer::UpdateMeshBuffers(const std::vector<int>& dirtyVertices)
{
    if (!m_Mesh) {
        return;
    }
    
    // Get mesh vertex data
    auto& meshVertices = const_cast<std::vector<float>&>(m_Mesh->GetVertices());
    
    // Vertex format: Position(3) + Normal(3) + UV(2) = 8 floats per vertex
    const int stride = 8;
    
    if (dirtyVertices.empty()) {
        // Update all vertices
        size_t vertexCount = m_VertexPositions.size();
        meshVertices.resize(vertexCount * stride);
        
        for (size_t i = 0; i < vertexCount; ++i) {
            size_t offset = i * stride;
            
            // Position
            meshVertices[offset + 0] = m_VertexPositions[i].x;
            meshVertices[offset + 1] = m_VertexPositions[i].y;
            meshVertices[offset + 2] = m_VertexPositions[i].z;
            
            // Normal
            meshVertices[offset + 3] = m_VertexNormals[i].x;
            meshVertices[offset + 4] = m_VertexNormals[i].y;
            meshVertices[offset + 5] = m_VertexNormals[i].z;
            
            // UV (keep existing values at offset + 6, 7)
        }
    } else {
        // Update only dirty vertices
        for (int vertexIdx : dirtyVertices) {
            if (vertexIdx < 0 || vertexIdx >= static_cast<int>(m_VertexPositions.size())) {
                continue;
            }
            
            size_t offset = vertexIdx * stride;
            
            if (offset + 7 < meshVertices.size()) {
                // Position
                meshVertices[offset + 0] = m_VertexPositions[vertexIdx].x;
                meshVertices[offset + 1] = m_VertexPositions[vertexIdx].y;
                meshVertices[offset + 2] = m_VertexPositions[vertexIdx].z;
                
                // Normal
                meshVertices[offset + 3] = m_VertexNormals[vertexIdx].x;
                meshVertices[offset + 4] = m_VertexNormals[vertexIdx].y;
                meshVertices[offset + 5] = m_VertexNormals[vertexIdx].z;
            }
        }
    }
    
    // Update GPU buffers
    m_Mesh->UpdateVertices(meshVertices);
}
