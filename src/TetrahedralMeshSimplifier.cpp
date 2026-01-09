#include "TetrahedralMeshSimplifier.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// ============================================================================
// Quadric Implementation
// ============================================================================

TetrahedralMeshSimplifier::Quadric::Quadric() {
    for (int i = 0; i < 10; ++i) {
        m[i] = 0.0;
    }
}

TetrahedralMeshSimplifier::Quadric::Quadric(double a, double b, double c, double d) {
    // Construct quadric from plane equation ax + by + cz + d = 0
    // Q = [a b c d]^T * [a b c d]
    m[0] = a * a;  // a^2
    m[1] = a * b;  // ab
    m[2] = a * c;  // ac
    m[3] = a * d;  // ad
    m[4] = b * b;  // b^2
    m[5] = b * c;  // bc
    m[6] = b * d;  // bd
    m[7] = c * c;  // c^2
    m[8] = c * d;  // cd
    m[9] = d * d;  // d^2
}

TetrahedralMeshSimplifier::Quadric TetrahedralMeshSimplifier::Quadric::operator+(const Quadric& q) const {
    Quadric result;
    for (int i = 0; i < 10; ++i) {
        result.m[i] = m[i] + q.m[i];
    }
    return result;
}

TetrahedralMeshSimplifier::Quadric& TetrahedralMeshSimplifier::Quadric::operator+=(const Quadric& q) {
    for (int i = 0; i < 10; ++i) {
        m[i] += q.m[i];
    }
    return *this;
}

double TetrahedralMeshSimplifier::Quadric::Evaluate(const Vec3& v) const {
    // Evaluate v^T * Q * v where v = [x, y, z, 1]
    double x = v.x, y = v.y, z = v.z;
    return m[0] * x * x + 2 * m[1] * x * y + 2 * m[2] * x * z + 2 * m[3] * x +
           m[4] * y * y + 2 * m[5] * y * z + 2 * m[6] * y +
           m[7] * z * z + 2 * m[8] * z +
           m[9];
}

bool TetrahedralMeshSimplifier::Quadric::FindOptimalPosition(Vec3& result) const {
    // Solve for optimal position by setting derivative to zero
    // This requires solving a 3x3 linear system
    
    // Build matrix A from quadric
    double A[3][3] = {
        {m[0], m[1], m[2]},
        {m[1], m[4], m[5]},
        {m[2], m[5], m[7]}
    };
    
    double b[3] = {-m[3], -m[6], -m[8]};
    
    // Compute determinant
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                 A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                 A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    
    if (std::abs(det) < 1e-10) {
        return false;  // Matrix is singular
    }
    
    // Solve using Cramer's rule
    double detX = b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                  A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2]) +
                  A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2]);
    
    double detY = A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2]) -
                  b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                  A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0]);
    
    double detZ = A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1]) -
                  A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0]) +
                  b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    
    result.x = static_cast<float>(detX / det);
    result.y = static_cast<float>(detY / det);
    result.z = static_cast<float>(detZ / det);
    
    return true;
}

// ============================================================================
// Face Hash
// ============================================================================

uint64_t TetrahedralMeshSimplifier::FaceHash(int v0, int v1, int v2) {
    // Sort vertices to ensure consistent hash
    if (v0 > v1) std::swap(v0, v1);
    if (v1 > v2) std::swap(v1, v2);
    if (v0 > v1) std::swap(v0, v1);
    
    // Pack into 64-bit hash (assuming vertex indices fit in 21 bits each)
    return (static_cast<uint64_t>(v0) << 42) | 
           (static_cast<uint64_t>(v1) << 21) | 
           static_cast<uint64_t>(v2);
}

// ============================================================================
// Main Simplification Functions
// ============================================================================

TetrahedralMeshSimplifier::SimplificationResult TetrahedralMeshSimplifier::Simplify(
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices,
    int targetVertexCount)
{
    SimplificationResult result;
    result.originalVertexCount = static_cast<int>(positions.size());
    result.originalTetrahedronCount = static_cast<int>(indices.size()) / 4;
    
    if (positions.empty() || indices.empty() || indices.size() % 4 != 0) {
        std::cerr << "TetrahedralMeshSimplifier: Invalid input mesh" << std::endl;
        return result;
    }
    
    if (targetVertexCount >= result.originalVertexCount) {
        // No simplification needed
        result.positions = positions;
        result.indices = indices;
        result.simplifiedVertexCount = result.originalVertexCount;
        result.simplifiedTetrahedronCount = result.originalTetrahedronCount;
        result.vertexMapping.resize(result.originalVertexCount);
        for (int i = 0; i < result.originalVertexCount; ++i) {
            result.vertexMapping[i] = i;
        }
        result.success = true;
        return result;
    }
    
    std::cout << "TetrahedralMeshSimplifier: Simplifying from " << result.originalVertexCount 
              << " to " << targetVertexCount << " vertices..." << std::endl;
    
    // Initialize state
    SimplificationState state;
    InitializeState(state, positions, indices);
    
    // Identify surface vertices
    IdentifySurfaceVertices(state);
    
    // Compute quadrics for all vertices
    ComputeQuadrics(state);
    
    // Build initial edge queue
    BuildEdgeQueue(state);
    
    // Perform edge collapses
    int currentVertexCount = result.originalVertexCount;
    int collapseCount = 0;
    
    while (currentVertexCount > targetVertexCount && !state.edgeQueue.empty()) {
        Edge edge = state.edgeQueue.top();
        state.edgeQueue.pop();
        
        // Check if edge is still valid
        if (state.vertices[edge.v0].deleted || state.vertices[edge.v1].deleted) {
            continue;
        }
        
        // Check if this edge was already processed
        uint64_t hash = EdgeHash(edge.v0, edge.v1);
        if (state.processedEdges.count(hash)) {
            continue;
        }
        
        // Attempt collapse
        if (CollapseEdge(state, edge)) {
            currentVertexCount--;
            collapseCount++;
            state.processedEdges.insert(hash);
            
            // Update costs for edges connected to the new vertex
            UpdateEdgeCosts(state, edge.v0);
        }
    }
    
    std::cout << "TetrahedralMeshSimplifier: Performed " << collapseCount << " edge collapses" << std::endl;
    
    // Remove degenerate tetrahedra
    RemoveDegenerateTetrahedra(state);
    
    // Extract result
    result = ExtractResult(state);
    result.success = true;
    
    std::cout << "TetrahedralMeshSimplifier: Result has " << result.simplifiedVertexCount 
              << " vertices and " << result.simplifiedTetrahedronCount << " tetrahedra" << std::endl;
    
    return result;
}

TetrahedralMeshSimplifier::SimplificationResult TetrahedralMeshSimplifier::SimplifyByRatio(
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices,
    float reductionRatio)
{
    int targetCount = static_cast<int>(positions.size() * reductionRatio);
    targetCount = std::max(8, targetCount);  // Minimum 8 vertices
    return Simplify(positions, indices, targetCount);
}

// ============================================================================
// Helper Functions
// ============================================================================

void TetrahedralMeshSimplifier::InitializeState(
    SimplificationState& state,
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices)
{
    // Initialize vertices
    state.vertices.reserve(positions.size());
    for (const auto& pos : positions) {
        state.vertices.emplace_back(pos);
    }
    
    // Initialize tetrahedra
    int tetraCount = static_cast<int>(indices.size()) / 4;
    state.tetrahedra.reserve(tetraCount);
    for (int i = 0; i < tetraCount; ++i) {
        int v0 = indices[i * 4 + 0];
        int v1 = indices[i * 4 + 1];
        int v2 = indices[i * 4 + 2];
        int v3 = indices[i * 4 + 3];
        
        state.tetrahedra.emplace_back(v0, v1, v2, v3);
        
        // Add tetrahedron to vertex adjacency
        state.vertices[v0].adjacentTetrahedra.insert(i);
        state.vertices[v1].adjacentTetrahedra.insert(i);
        state.vertices[v2].adjacentTetrahedra.insert(i);
        state.vertices[v3].adjacentTetrahedra.insert(i);
    }
    
    // Initialize vertex mapping
    state.vertexMapping.resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        state.vertexMapping[i] = static_cast<int>(i);
    }
}

void TetrahedralMeshSimplifier::IdentifySurfaceVertices(SimplificationState& state) {
    // Count how many times each face appears
    // Surface faces appear only once (boundary of the volume)
    
    for (const auto& tet : state.tetrahedra) {
        if (tet.deleted) continue;
        
        int faces[4][3];
        tet.GetFaces(faces);
        
        for (int i = 0; i < 4; ++i) {
            uint64_t hash = FaceHash(faces[i][0], faces[i][1], faces[i][2]);
            state.faceCount[hash]++;
        }
    }
    
    // Mark vertices that belong to surface faces
    for (const auto& tet : state.tetrahedra) {
        if (tet.deleted) continue;
        
        int faces[4][3];
        tet.GetFaces(faces);
        
        for (int i = 0; i < 4; ++i) {
            uint64_t hash = FaceHash(faces[i][0], faces[i][1], faces[i][2]);
            if (state.faceCount[hash] == 1) {
                // This is a surface face
                state.vertices[faces[i][0]].isSurface = true;
                state.vertices[faces[i][1]].isSurface = true;
                state.vertices[faces[i][2]].isSurface = true;
            }
        }
    }
}

void TetrahedralMeshSimplifier::ComputeQuadrics(SimplificationState& state) {
    // Compute quadric for each vertex from adjacent tetrahedral faces
    
    for (auto& tet : state.tetrahedra) {
        if (tet.deleted) continue;
        
        int faces[4][3];
        tet.GetFaces(faces);
        
        // For each face of the tetrahedron
        for (int i = 0; i < 4; ++i) {
            Vec3 p0 = state.vertices[faces[i][0]].position;
            Vec3 p1 = state.vertices[faces[i][1]].position;
            Vec3 p2 = state.vertices[faces[i][2]].position;
            
            // Compute plane equation for this face
            Vec3 v1 = p1 - p0;
            Vec3 v2 = p2 - p0;
            Vec3 normal = v1.Cross(v2);
            float length = normal.Length();
            
            if (length > 1e-6f) {
                normal = normal / length;
                
                double a = normal.x;
                double b = normal.y;
                double c = normal.z;
                double d = -(a * p0.x + b * p0.y + c * p0.z);
                
                Quadric q(a, b, c, d);
                
                // Add quadric to all three vertices of the face
                state.vertices[faces[i][0]].q += q;
                state.vertices[faces[i][1]].q += q;
                state.vertices[faces[i][2]].q += q;
            }
        }
    }
}

void TetrahedralMeshSimplifier::BuildEdgeQueue(SimplificationState& state) {
    std::unordered_set<uint64_t> processedEdges;
    
    for (const auto& tet : state.tetrahedra) {
        if (tet.deleted) continue;
        
        // Add all 6 edges of the tetrahedron
        int edges[6][2] = {
            {tet.v[0], tet.v[1]}, {tet.v[0], tet.v[2]}, {tet.v[0], tet.v[3]},
            {tet.v[1], tet.v[2]}, {tet.v[1], tet.v[3]}, {tet.v[2], tet.v[3]}
        };
        
        for (int i = 0; i < 6; ++i) {
            int v0 = edges[i][0];
            int v1 = edges[i][1];
            
            uint64_t hash = EdgeHash(v0, v1);
            if (processedEdges.count(hash)) continue;
            
            processedEdges.insert(hash);
            
            Edge edge(v0, v1);
            edge.cost = ComputeEdgeCost(state, v0, v1, edge.optimalPos);
            state.edgeQueue.push(edge);
        }
    }
}

bool TetrahedralMeshSimplifier::CollapseEdge(SimplificationState& state, const Edge& edge) {
    int v0 = edge.v0;
    int v1 = edge.v1;
    
    // Don't collapse if both vertices are on the surface (preserve surface topology)
    if (state.vertices[v0].isSurface && state.vertices[v1].isSurface) {
        if (IsBoundaryEdge(state, v0, v1)) {
            return false;  // Preserve boundary edges more aggressively
        }
    }
    
    // Update vertex position
    state.vertices[v0].position = edge.optimalPos;
    state.vertices[v0].q += state.vertices[v1].q;
    
    // Mark v1 as deleted
    state.vertices[v1].deleted = true;
    
    // Update tetrahedra
    for (int tetIdx : state.vertices[v1].adjacentTetrahedra) {
        auto& tet = state.tetrahedra[tetIdx];
        if (tet.deleted) continue;
        
        // If tetrahedron contains both v0 and v1, it will become degenerate
        if (tet.HasVertex(v0) && tet.HasVertex(v1)) {
            tet.deleted = true;
            continue;
        }
        
        // Replace v1 with v0
        tet.ReplaceVertex(v1, v0);
        state.vertices[v0].adjacentTetrahedra.insert(tetIdx);
    }
    
    // Update vertex mapping
    for (auto& mapping : state.vertexMapping) {
        if (mapping == v1) {
            mapping = v0;
        }
    }
    
    return true;
}

void TetrahedralMeshSimplifier::UpdateEdgeCosts(SimplificationState& state, int vertex) {
    // Recompute costs for all edges connected to this vertex
    std::unordered_set<int> adjacentVertices;
    
    for (int tetIdx : state.vertices[vertex].adjacentTetrahedra) {
        const auto& tet = state.tetrahedra[tetIdx];
        if (tet.deleted) continue;
        
        for (int i = 0; i < 4; ++i) {
            if (tet.v[i] != vertex && !state.vertices[tet.v[i]].deleted) {
                adjacentVertices.insert(tet.v[i]);
            }
        }
    }
    
    for (int adjVertex : adjacentVertices) {
        Edge edge(vertex, adjVertex);
        edge.cost = ComputeEdgeCost(state, vertex, adjVertex, edge.optimalPos);
        state.edgeQueue.push(edge);
    }
}

double TetrahedralMeshSimplifier::ComputeEdgeCost(
    const SimplificationState& state,
    int v0, int v1,
    Vec3& optimalPos)
{
    const Vertex& vert0 = state.vertices[v0];
    const Vertex& vert1 = state.vertices[v1];
    
    // Combine quadrics
    Quadric q = vert0.q + vert1.q;
    
    // Try to find optimal position
    if (q.FindOptimalPosition(optimalPos)) {
        return q.Evaluate(optimalPos);
    }
    
    // If optimal position can't be found, use midpoint
    optimalPos = (vert0.position + vert1.position) * 0.5f;
    return q.Evaluate(optimalPos);
}

bool TetrahedralMeshSimplifier::IsBoundaryEdge(const SimplificationState& state, int v0, int v1) {
    // An edge is a boundary edge if it belongs to a surface face
    // Check all tetrahedra containing both vertices
    
    for (int tetIdx : state.vertices[v0].adjacentTetrahedra) {
        const auto& tet = state.tetrahedra[tetIdx];
        if (tet.deleted || !tet.HasVertex(v1)) continue;
        
        int faces[4][3];
        tet.GetFaces(faces);
        
        for (int i = 0; i < 4; ++i) {
            // Check if this face contains both v0 and v1
            bool hasV0 = (faces[i][0] == v0 || faces[i][1] == v0 || faces[i][2] == v0);
            bool hasV1 = (faces[i][0] == v1 || faces[i][1] == v1 || faces[i][2] == v1);
            
            if (hasV0 && hasV1) {
                uint64_t hash = FaceHash(faces[i][0], faces[i][1], faces[i][2]);
                if (state.faceCount.count(hash) && state.faceCount.at(hash) == 1) {
                    return true;  // This face is on the boundary
                }
            }
        }
    }
    
    return false;
}

void TetrahedralMeshSimplifier::RemoveDegenerateTetrahedra(SimplificationState& state) {
    for (auto& tet : state.tetrahedra) {
        if (tet.deleted) continue;
        
        // Check if all vertices are unique
        std::unordered_set<int> uniqueVerts;
        for (int i = 0; i < 4; ++i) {
            uniqueVerts.insert(tet.v[i]);
        }
        
        if (uniqueVerts.size() < 4) {
            tet.deleted = true;
        }
    }
}

TetrahedralMeshSimplifier::SimplificationResult TetrahedralMeshSimplifier::ExtractResult(
    const SimplificationState& state)
{
    SimplificationResult result;
    
    // Build vertex remapping
    std::unordered_map<int, int> oldToNew;
    int newIndex = 0;
    
    for (size_t i = 0; i < state.vertices.size(); ++i) {
        if (!state.vertices[i].deleted) {
            oldToNew[static_cast<int>(i)] = newIndex++;
            result.positions.push_back(state.vertices[i].position);
        }
    }
    
    result.simplifiedVertexCount = static_cast<int>(result.positions.size());
    
    // Extract tetrahedra
    for (const auto& tet : state.tetrahedra) {
        if (tet.deleted) continue;
        
        result.indices.push_back(oldToNew[tet.v[0]]);
        result.indices.push_back(oldToNew[tet.v[1]]);
        result.indices.push_back(oldToNew[tet.v[2]]);
        result.indices.push_back(oldToNew[tet.v[3]]);
    }
    
    result.simplifiedTetrahedronCount = static_cast<int>(result.indices.size()) / 4;
    
    // Build vertex mapping
    result.vertexMapping.resize(state.vertexMapping.size());
    for (size_t i = 0; i < state.vertexMapping.size(); ++i) {
        int mappedVertex = state.vertexMapping[i];
        // Follow chain to final vertex
        while (state.vertices[mappedVertex].deleted && state.vertexMapping[mappedVertex] != mappedVertex) {
            mappedVertex = state.vertexMapping[mappedVertex];
        }
        result.vertexMapping[i] = oldToNew.count(mappedVertex) ? oldToNew[mappedVertex] : 0;
    }
    
    return result;
}
