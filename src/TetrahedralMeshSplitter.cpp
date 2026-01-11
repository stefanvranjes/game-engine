#include "TetrahedralMeshSplitter.h"
#include <queue>
#include <algorithm>
#include <iostream>

TetrahedralMeshSplitter::SplitResult TetrahedralMeshSplitter::SplitAlongTear(
    const Vec3* vertices,
    int vertexCount,
    const int* tetrahedra,
    int tetrahedronCount,
    const std::vector<SoftBodyTearSystem::TearInfo>& tears)
{
    SplitResult result;
    result.splitSuccessful = false;

    if (tears.empty()) {
        std::cerr << "No tears to split" << std::endl;
        return result;
    }

    // Build set of torn tetrahedra
    std::unordered_set<int> tornTets;
    for (const auto& tear : tears) {
        tornTets.insert(tear.tetrahedronIndex);
    }

    // Build connectivity graph
    std::vector<std::vector<int>> adjacency;
    BuildConnectivityGraph(tetrahedra, tetrahedronCount, adjacency);

    // Partition tetrahedra into two groups
    std::vector<int> partition1, partition2;
    PartitionTetrahedra(adjacency, tornTets, partition1, partition2);

    if (partition1.empty() || partition2.empty()) {
        std::cerr << "Failed to partition mesh" << std::endl;
        return result;
    }

    std::cout << "Partitioned mesh: " << partition1.size() << " tets in piece 1, "
              << partition2.size() << " tets in piece 2" << std::endl;

    // Collect tear vertices
    std::unordered_set<int> tearVertices;
    for (const auto& tear : tears) {
        tearVertices.insert(tear.edgeVertices[0]);
        tearVertices.insert(tear.edgeVertices[1]);
    }
    
    // Determine which partition owns each tear vertex
    std::unordered_map<int, int> vertexOwnership;
    DetermineVertexOwnership(partition1, partition2, tetrahedra, tearVertices, vertexOwnership);
    
    // Duplicate vertices along tear line
    std::unordered_map<int, int> vertexDuplication;
    DuplicateTearVertices(vertices, vertexCount, tears, partition1, partition2, 
                         tetrahedra, vertexDuplication);

    // Extract meshes for each partition (partition 0 = piece 1, partition 1 = piece 2)
    ExtractPartitionMesh(vertices, vertexCount, tetrahedra, partition1, vertexDuplication,
                        vertexOwnership, 0, result.vertices1, result.tetrahedra1, result.vertexMapping1);
    
    ExtractPartitionMesh(vertices, vertexCount, tetrahedra, partition2, vertexDuplication,
                        vertexOwnership, 1, result.vertices2, result.tetrahedra2, result.vertexMapping2);

    result.piece1VertexCount = static_cast<int>(result.vertices1.size());
    result.piece2VertexCount = static_cast<int>(result.vertices2.size());
    
    // Calculate tear boundary information
    result.tearEnergy = 0.0f;
    for (const auto& tear : tears) {
        // Store tear edge positions
        Vec3 v0 = vertices[tear.edgeVertices[0]];
        Vec3 v1 = vertices[tear.edgeVertices[1]];
        Vec3 midpoint = (v0 + v1) * 0.5f;
        result.tearBoundaryPositions.push_back(midpoint);
        result.tearBoundaryNormals.push_back(tear.tearNormal);
        
        // Accumulate tear energy (proportional to stress and edge length)
        float edgeLength = (v1 - v0).Length();
        result.tearEnergy += tear.stress * edgeLength;
    }
    
    result.splitSuccessful = true;

    return result;
}

void TetrahedralMeshSplitter::BuildConnectivityGraph(
    const int* tetrahedra,
    int tetrahedronCount,
    std::vector<std::vector<int>>& outAdjacency)
{
    outAdjacency.resize(tetrahedronCount);

    // Check each pair of tetrahedra for shared faces
    for (int i = 0; i < tetrahedronCount; ++i) {
        const int* tet1 = &tetrahedra[i * 4];
        
        for (int j = i + 1; j < tetrahedronCount; ++j) {
            const int* tet2 = &tetrahedra[j * 4];
            
            // Count shared vertices
            int sharedCount = 0;
            for (int vi = 0; vi < 4; ++vi) {
                for (int vj = 0; vj < 4; ++vj) {
                    if (tet1[vi] == tet2[vj]) {
                        sharedCount++;
                        break;
                    }
                }
            }
            
            // If they share 3 vertices, they share a face
            if (sharedCount >= 3) {
                outAdjacency[i].push_back(j);
                outAdjacency[j].push_back(i);
            }
        }
    }
}

void TetrahedralMeshSplitter::PartitionTetrahedra(
    const std::vector<std::vector<int>>& adjacency,
    const std::unordered_set<int>& tornTets,
    std::vector<int>& outPartition1,
    std::vector<int>& outPartition2)
{
    int tetCount = static_cast<int>(adjacency.size());
    std::vector<int> partition(tetCount, -1);  // -1 = unassigned, 0 = piece1, 1 = piece2

    // Mark torn tets as barriers
    for (int tornTet : tornTets) {
        partition[tornTet] = -2;  // -2 = torn (barrier)
    }

    // Find first non-torn tet as seed for partition 1
    int seed1 = -1;
    for (int i = 0; i < tetCount; ++i) {
        if (partition[i] == -1) {
            seed1 = i;
            break;
        }
    }

    if (seed1 == -1) {
        return;  // All tets are torn
    }

    // Flood fill from seed1 for partition 1
    std::queue<int> queue;
    queue.push(seed1);
    partition[seed1] = 0;

    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();

        for (int neighbor : adjacency[current]) {
            if (partition[neighbor] == -1) {  // Unassigned and not torn
                partition[neighbor] = 0;
                queue.push(neighbor);
            }
        }
    }

    // Find seed for partition 2 (first unassigned tet)
    int seed2 = -1;
    for (int i = 0; i < tetCount; ++i) {
        if (partition[i] == -1) {
            seed2 = i;
            break;
        }
    }

    if (seed2 == -1) {
        // Only one partition exists, put all in partition1
        for (int i = 0; i < tetCount; ++i) {
            if (partition[i] == 0) {
                outPartition1.push_back(i);
            }
        }
        return;
    }

    // Flood fill from seed2 for partition 2
    queue.push(seed2);
    partition[seed2] = 1;

    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();

        for (int neighbor : adjacency[current]) {
            if (partition[neighbor] == -1) {
                partition[neighbor] = 1;
                queue.push(neighbor);
            }
        }
    }

    // Collect partitions
    for (int i = 0; i < tetCount; ++i) {
        if (partition[i] == 0) {
            outPartition1.push_back(i);
        } else if (partition[i] == 1) {
            outPartition2.push_back(i);
        }
    }
}

void TetrahedralMeshSplitter::DuplicateTearVertices(
    const Vec3* vertices,
    int vertexCount,
    const std::vector<SoftBodyTearSystem::TearInfo>& tears,
    const std::vector<int>& partition1,
    const std::vector<int>& partition2,
    const int* tetrahedra,
    std::unordered_map<int, int>& outVertexDuplication)
{
    // Collect all vertices on tear edges
    std::unordered_set<int> tearVertices;
    for (const auto& tear : tears) {
        tearVertices.insert(tear.edgeVertices[0]);
        tearVertices.insert(tear.edgeVertices[1]);
    }

    // For each tear vertex, we'll duplicate it
    // The original stays with partition1, the duplicate goes to partition2
    int nextDuplicateId = vertexCount;
    
    for (int vertexId : tearVertices) {
        outVertexDuplication[vertexId] = nextDuplicateId++;
    }
}

void TetrahedralMeshSplitter::ExtractPartitionMesh(
    const Vec3* vertices,
    int vertexCount,
    const int* tetrahedra,
    const std::vector<int>& partition,
    const std::unordered_map<int, int>& vertexDuplication,
    const std::unordered_map<int, int>& vertexOwnership,
    int partitionId,
    std::vector<Vec3>& outVertices,
    std::vector<int>& outTetrahedra,
    std::vector<int>& outVertexMapping)
{
    // Build vertex remapping
    std::unordered_map<int, int> vertexRemap;
    int newVertexIndex = 0;

    // First pass: collect all vertices used by this partition
    std::unordered_set<int> usedVertices;
    for (int tetIdx : partition) {
        for (int i = 0; i < 4; ++i) {
            usedVertices.insert(tetrahedra[tetIdx * 4 + i]);
        }
    }

    // Create vertex mapping for non-duplicated vertices
    for (int oldIdx : usedVertices) {
        // Check if this is a duplicated vertex
        auto dupIt = vertexDuplication.find(oldIdx);
        if (dupIt != vertexDuplication.end()) {
            // This is a tear vertex - check ownership
            auto ownIt = vertexOwnership.find(oldIdx);
            if (ownIt != vertexOwnership.end()) {
                if (ownIt->second == partitionId) {
                    // This partition owns the original vertex
                    vertexRemap[oldIdx] = newVertexIndex++;
                    outVertices.push_back(vertices[oldIdx]);
                    outVertexMapping.push_back(oldIdx);
                } else {
                    // This partition gets the duplicate
                    vertexRemap[oldIdx] = newVertexIndex++;
                    outVertices.push_back(vertices[oldIdx]);  // Same position initially
                    outVertexMapping.push_back(oldIdx);  // Track original for velocity transfer
                }
            } else {
                // No ownership info, just use original
                vertexRemap[oldIdx] = newVertexIndex++;
                outVertices.push_back(vertices[oldIdx]);
                outVertexMapping.push_back(oldIdx);
            }
        } else {
            // Regular vertex, not duplicated
            vertexRemap[oldIdx] = newVertexIndex++;
            outVertices.push_back(vertices[oldIdx]);
            outVertexMapping.push_back(oldIdx);
        }
    }

    // Extract tetrahedra with remapped indices
    for (int tetIdx : partition) {
        for (int i = 0; i < 4; ++i) {
            int oldVertexIdx = tetrahedra[tetIdx * 4 + i];
            int newVertexIdx = vertexRemap[oldVertexIdx];
            outTetrahedra.push_back(newVertexIdx);
        }
    }
}

bool TetrahedralMeshSplitter::TetrahedraShareEdge(
    const int* tet1,
    const int* tet2,
    int& outV0,
    int& outV1)
{
    // Check all edge pairs
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            int v0 = tet1[i];
            int v1 = tet1[j];
            
            // Check if tet2 contains both vertices
            bool hasV0 = false, hasV1 = false;
            for (int k = 0; k < 4; ++k) {
                if (tet2[k] == v0) hasV0 = true;
                if (tet2[k] == v1) hasV1 = true;
            }
            
            if (hasV0 && hasV1) {
                outV0 = v0;
                outV1 = v1;
                return true;
            }
        }
    }
    
    return false;
}

TetrahedralMeshSplitter::SplitResult TetrahedralMeshSplitter::SplitWithStateTransfer(
    const Vec3* vertices,
    int vertexCount,
    const int* tetrahedra,
    int tetrahedronCount,
    const std::vector<SoftBodyTearSystem::TearInfo>& tears,
    const Vec3* velocities,
    std::vector<Vec3>& outVelocities1,
    std::vector<Vec3>& outVelocities2)
{
    // First perform the standard split
    SplitResult result = SplitAlongTear(vertices, vertexCount, tetrahedra, tetrahedronCount, tears);
    
    if (!result.splitSuccessful || !velocities) {
        return result;
    }
    
    // Transfer velocities for piece 1
    outVelocities1.resize(result.vertices1.size());
    for (size_t i = 0; i < result.vertexMapping1.size(); ++i) {
        int originalIdx = result.vertexMapping1[i];
        if (originalIdx >= 0 && originalIdx < vertexCount) {
            outVelocities1[i] = velocities[originalIdx];
        }
    }
    
    // Transfer velocities for piece 2
    outVelocities2.resize(result.vertices2.size());
    for (size_t i = 0; i < result.vertexMapping2.size(); ++i) {
        int originalIdx = result.vertexMapping2[i];
        if (originalIdx >= 0 && originalIdx < vertexCount) {
            outVelocities2[i] = velocities[originalIdx];
        }
    }
    
    return result;
}

void TetrahedralMeshSplitter::DetermineVertexOwnership(
    const std::vector<int>& partition1,
    const std::vector<int>& partition2,
    const int* tetrahedra,
    const std::unordered_set<int>& tearVertices,
    std::unordered_map<int, int>& outOwnership)
{
    outOwnership.clear();
    
    // For each tear vertex, count how many tets in each partition use it
    for (int vertexId : tearVertices) {
        int count1 = 0;
        int count2 = 0;
        
        // Count usage in partition 1
        for (int tetIdx : partition1) {
            const int* tet = &tetrahedra[tetIdx * 4];
            for (int i = 0; i < 4; ++i) {
                if (tet[i] == vertexId) {
                    count1++;
                    break;
                }
            }
        }
        
        // Count usage in partition 2
        for (int tetIdx : partition2) {
            const int* tet = &tetrahedra[tetIdx * 4];
            for (int i = 0; i < 4; ++i) {
                if (tet[i] == vertexId) {
                    count2++;
                    break;
                }
            }
        }
        
        // Assign to partition with more usage
        outOwnership[vertexId] = (count1 >= count2) ? 0 : 1;
    }
}

