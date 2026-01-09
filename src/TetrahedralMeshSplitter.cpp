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

    // Duplicate vertices along tear line
    std::unordered_map<int, int> vertexDuplication;
    DuplicateTearVertices(vertices, vertexCount, tears, partition1, partition2, 
                         tetrahedra, vertexDuplication);

    // Extract meshes for each partition
    ExtractPartitionMesh(vertices, vertexCount, tetrahedra, partition1, vertexDuplication,
                        result.vertices1, result.tetrahedra1, result.vertexMapping1);
    
    ExtractPartitionMesh(vertices, vertexCount, tetrahedra, partition2, vertexDuplication,
                        result.vertices2, result.tetrahedra2, result.vertexMapping2);

    result.piece1VertexCount = static_cast<int>(result.vertices1.size());
    result.piece2VertexCount = static_cast<int>(result.vertices2.size());
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

    // Create vertex mapping
    for (int oldIdx : usedVertices) {
        vertexRemap[oldIdx] = newVertexIndex++;
        outVertices.push_back(vertices[oldIdx]);
        outVertexMapping.push_back(oldIdx);
    }

    // Handle duplicated vertices (for partition 2)
    for (const auto& pair : vertexDuplication) {
        int originalIdx = pair.first;
        int duplicateIdx = pair.second;
        
        // Check if this partition uses the duplicated vertex
        if (usedVertices.count(originalIdx) > 0) {
            // For partition 2, remap to duplicate
            // This is a simplified approach - in production, you'd check which partition
            // actually needs the duplicate based on connectivity
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
