#include "ClothMeshSimplifier.h"
#include "Profiler.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// Quadric implementation

ClothMeshSimplifier::Quadric::Quadric() {
    for (int i = 0; i < 10; ++i) {
        m[i] = 0.0;
    }
}

ClothMeshSimplifier::Quadric::Quadric(double a, double b, double c, double d) {
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

ClothMeshSimplifier::Quadric ClothMeshSimplifier::Quadric::operator+(const Quadric& q) const {
    Quadric result;
    for (int i = 0; i < 10; ++i) {
        result.m[i] = m[i] + q.m[i];
    }
    return result;
}

ClothMeshSimplifier::Quadric& ClothMeshSimplifier::Quadric::operator+=(const Quadric& q) {
    for (int i = 0; i < 10; ++i) {
        m[i] += q.m[i];
    }
    return *this;
}

double ClothMeshSimplifier::Quadric::Evaluate(const Vec3& v) const {
    // Evaluate v^T * Q * v where v = [x, y, z, 1]
    double x = v.x, y = v.y, z = v.z;
    
    return m[0] * x * x + 2 * m[1] * x * y + 2 * m[2] * x * z + 2 * m[3] * x
         + m[4] * y * y + 2 * m[5] * y * z + 2 * m[6] * y
         + m[7] * z * z + 2 * m[8] * z
         + m[9];
}

bool ClothMeshSimplifier::Quadric::FindOptimalPosition(Vec3& result) const {
    // Solve Q * v = [0, 0, 0, 1]^T for v = [x, y, z, 1]
    // This gives us the optimal position that minimizes the quadric error
    
    // Build 3x3 matrix from quadric
    double A[3][3] = {
        { m[0], m[1], m[2] },
        { m[1], m[4], m[5] },
        { m[2], m[5], m[7] }
    };
    
    double b[3] = { -m[3], -m[6], -m[8] };
    
    // Compute determinant
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
               - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
               + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    
    if (std::abs(det) < 1e-10) {
        return false;  // Matrix is singular
    }
    
    // Compute inverse using Cramer's rule
    double invDet = 1.0 / det;
    
    double invA[3][3];
    invA[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * invDet;
    invA[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * invDet;
    invA[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * invDet;
    invA[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * invDet;
    invA[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * invDet;
    invA[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * invDet;
    invA[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * invDet;
    invA[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * invDet;
    invA[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * invDet;
    
    // Solve for optimal position
    result.x = static_cast<float>(invA[0][0] * b[0] + invA[0][1] * b[1] + invA[0][2] * b[2]);
    result.y = static_cast<float>(invA[1][0] * b[0] + invA[1][1] * b[1] + invA[1][2] * b[2]);
    result.z = static_cast<float>(invA[2][0] * b[0] + invA[2][1] * b[1] + invA[2][2] * b[2]);
    
    return true;
}

// Public interface

ClothMeshSimplifier::SimplificationResult ClothMeshSimplifier::SimplifyByRatio(
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices,
    float reductionRatio)
{
    SCOPED_PROFILE("ClothMeshSimplifier::SimplifyByRatio");
    if (reductionRatio <= 0.0f || reductionRatio > 1.0f) {
        std::cerr << "ClothMeshSimplifier: Invalid reduction ratio " << reductionRatio << std::endl;
        return SimplificationResult();
    }
    
    int targetVertexCount = static_cast<int>(positions.size() * reductionRatio);
    targetVertexCount = std::max(4, targetVertexCount);  // Minimum 4 vertices
    
    return Simplify(positions, indices, targetVertexCount);
}

ClothMeshSimplifier::SimplificationResult ClothMeshSimplifier::Simplify(
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices,
    int targetVertexCount)
{
    SCOPED_PROFILE("ClothMeshSimplifier::Simplify");
    SimplificationResult result;
    result.originalVertexCount = static_cast<int>(positions.size());
    result.originalTriangleCount = static_cast<int>(indices.size()) / 3;
    
    if (positions.empty() || indices.empty()) {
        std::cerr << "ClothMeshSimplifier: Empty input mesh" << std::endl;
        return result;
    }
    
    if (targetVertexCount >= result.originalVertexCount) {
        // No simplification needed
        result.positions = positions;
        result.indices = indices;
        result.simplifiedVertexCount = result.originalVertexCount;
        result.simplifiedTriangleCount = result.originalTriangleCount;
        result.vertexMapping.resize(result.originalVertexCount);
        for (int i = 0; i < result.originalVertexCount; ++i) {
            result.vertexMapping[i] = i;
        }
        result.success = true;
        return result;
    }
    
    std::cout << "ClothMeshSimplifier: Simplifying from " << result.originalVertexCount 
              << " to " << targetVertexCount << " vertices..." << std::endl;
    
    // Initialize simplification state
    SimplificationState state;
    InitializeState(state, positions, indices);
    
    // Compute quadric error metrics for all vertices
    ComputeQuadrics(state);
    
    // Build initial edge queue
    BuildEdgeQueue(state);
    
    // Perform edge collapses until target is reached
    int currentVertexCount = result.originalVertexCount;
    int collapsesPerformed = 0;
    
    while (currentVertexCount > targetVertexCount && !state.edgeQueue.empty()) {
        Edge edge = state.edgeQueue.top();
        state.edgeQueue.pop();
        
        // Check if vertices still exist
        if (state.vertices[edge.v0].deleted || state.vertices[edge.v1].deleted) {
            continue;
        }
        
        // Check if edge has already been processed
        uint64_t edgeHash = EdgeHash(edge.v0, edge.v1);
        if (state.processedEdges.count(edgeHash) > 0) {
            continue;
        }
        
        // Perform collapse
        if (CollapseEdge(state, edge)) {
            currentVertexCount--;
            collapsesPerformed++;
            state.processedEdges.insert(edgeHash);
            
            // Update costs for edges connected to the merged vertex
            UpdateEdgeCosts(state, edge.v0);
        }
    }
    
    // Remove degenerate triangles
    RemoveDegenerateTriangles(state);
    
    std::cout << "ClothMeshSimplifier: Performed " << collapsesPerformed 
              << " edge collapses" << std::endl;
    
    // Extract result
    result = ExtractResult(state);
    result.success = true;
    
    std::cout << "ClothMeshSimplifier: Result has " << result.simplifiedVertexCount 
              << " vertices and " << result.simplifiedTriangleCount << " triangles" << std::endl;
    
    return result;
}

// Helper methods

void ClothMeshSimplifier::InitializeState(
    SimplificationState& state,
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices)
{
    // Initialize vertices
    state.vertices.resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        state.vertices[i].position = positions[i];
    }
    
    // Initialize triangles and build adjacency
    int triangleCount = static_cast<int>(indices.size()) / 3;
    state.triangles.reserve(triangleCount);
    
    for (int i = 0; i < triangleCount; ++i) {
        int v0 = indices[i * 3 + 0];
        int v1 = indices[i * 3 + 1];
        int v2 = indices[i * 3 + 2];
        
        state.triangles.emplace_back(v0, v1, v2);
        
        // Add triangle to vertex adjacency lists
        state.vertices[v0].adjacentTriangles.insert(i);
        state.vertices[v1].adjacentTriangles.insert(i);
        state.vertices[v2].adjacentTriangles.insert(i);
    }
    
    // Initialize vertex mapping (identity)
    state.vertexMapping.resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        state.vertexMapping[i] = static_cast<int>(i);
    }
}

void ClothMeshSimplifier::ComputeQuadrics(SimplificationState& state) {
    SCOPED_PROFILE("ClothMeshSimplifier::ComputeQuadrics");
    // Compute quadric for each vertex as sum of adjacent face quadrics
    for (size_t i = 0; i < state.vertices.size(); ++i) {
        Vertex& vertex = state.vertices[i];
        
        for (int triIdx : vertex.adjacentTriangles) {
            const Triangle& tri = state.triangles[triIdx];
            
            // Get triangle vertices
            Vec3 p0 = state.vertices[tri.v[0]].position;
            Vec3 p1 = state.vertices[tri.v[1]].position;
            Vec3 p2 = state.vertices[tri.v[2]].position;
            
            // Compute plane equation
            Vec3 edge1 = p1 - p0;
            Vec3 edge2 = p2 - p0;
            Vec3 normal = edge1.Cross(edge2);
            
            float length = normal.Length();
            if (length < 1e-6f) {
                continue;  // Degenerate triangle
            }
            
            normal = normal * (1.0f / length);  // Normalize
            
            double a = normal.x;
            double b = normal.y;
            double c = normal.z;
            double d = -(a * p0.x + b * p0.y + c * p0.z);
            
            // Add plane quadric to all vertices of this triangle
            Quadric planeQuadric(a, b, c, d);
            state.vertices[tri.v[0]].q += planeQuadric;
            state.vertices[tri.v[1]].q += planeQuadric;
            state.vertices[tri.v[2]].q += planeQuadric;
        }
    }
}

void ClothMeshSimplifier::BuildEdgeQueue(SimplificationState& state) {
    SCOPED_PROFILE("ClothMeshSimplifier::BuildEdgeQueue");
    // Build set of unique edges
    std::unordered_set<uint64_t> uniqueEdges;
    
    for (const Triangle& tri : state.triangles) {
        if (tri.deleted) continue;
        
        // Add all three edges
        uniqueEdges.insert(EdgeHash(tri.v[0], tri.v[1]));
        uniqueEdges.insert(EdgeHash(tri.v[1], tri.v[2]));
        uniqueEdges.insert(EdgeHash(tri.v[2], tri.v[0]));
    }
    
    // Compute cost for each edge and add to queue
    for (uint64_t hash : uniqueEdges) {
        int v0 = static_cast<int>(hash >> 32);
        int v1 = static_cast<int>(hash & 0xFFFFFFFF);
        
        Vec3 optimalPos;
        double cost = ComputeEdgeCost(state, v0, v1, optimalPos);
        
        Edge edge(v0, v1);
        edge.cost = cost;
        edge.optimalPos = optimalPos;
        
        state.edgeQueue.push(edge);
    }
}

double ClothMeshSimplifier::ComputeEdgeCost(
    const SimplificationState& state,
    int v0, int v1,
    Vec3& optimalPos)
{
    const Vertex& vertex0 = state.vertices[v0];
    const Vertex& vertex1 = state.vertices[v1];
    
    // Combine quadrics
    Quadric combinedQ = vertex0.q + vertex1.q;
    
    // Try to find optimal position
    if (combinedQ.FindOptimalPosition(optimalPos)) {
        return combinedQ.Evaluate(optimalPos);
    }
    
    // Fallback: try midpoint and endpoints
    Vec3 midpoint = (vertex0.position + vertex1.position) * 0.5f;
    
    double cost0 = combinedQ.Evaluate(vertex0.position);
    double cost1 = combinedQ.Evaluate(vertex1.position);
    double costMid = combinedQ.Evaluate(midpoint);
    
    double minCost = std::min({cost0, cost1, costMid});
    
    if (minCost == cost0) {
        optimalPos = vertex0.position;
    } else if (minCost == cost1) {
        optimalPos = vertex1.position;
    } else {
        optimalPos = midpoint;
    }
    
    return minCost;
}

bool ClothMeshSimplifier::CollapseEdge(SimplificationState& state, const Edge& edge) {
    int v0 = edge.v0;
    int v1 = edge.v1;
    
    // Move v0 to optimal position
    state.vertices[v0].position = edge.optimalPos;
    
    // Combine quadrics
    state.vertices[v0].q += state.vertices[v1].q;
    
    // Update all triangles that reference v1 to reference v0
    std::vector<int> trianglesToUpdate(
        state.vertices[v1].adjacentTriangles.begin(),
        state.vertices[v1].adjacentTriangles.end()
    );
    
    for (int triIdx : trianglesToUpdate) {
        Triangle& tri = state.triangles[triIdx];
        
        if (tri.deleted) continue;
        
        // Check if this triangle contains both v0 and v1 (will become degenerate)
        if (tri.HasVertex(v0) && tri.HasVertex(v1)) {
            tri.deleted = true;
            state.vertices[tri.v[0]].adjacentTriangles.erase(triIdx);
            state.vertices[tri.v[1]].adjacentTriangles.erase(triIdx);
            state.vertices[tri.v[2]].adjacentTriangles.erase(triIdx);
            continue;
        }
        
        // Replace v1 with v0
        tri.ReplaceVertex(v1, v0);
        
        // Update adjacency
        state.vertices[v1].adjacentTriangles.erase(triIdx);
        state.vertices[v0].adjacentTriangles.insert(triIdx);
    }
    
    // Mark v1 as deleted
    state.vertices[v1].deleted = true;
    
    // Update vertex mapping
    for (size_t i = 0; i < state.vertexMapping.size(); ++i) {
        if (state.vertexMapping[i] == v1) {
            state.vertexMapping[i] = v0;
        }
    }
    
    return true;
}

void ClothMeshSimplifier::UpdateEdgeCosts(SimplificationState& state, int vertex) {
    // Find all edges connected to this vertex
    std::unordered_set<int> adjacentVertices;
    
    for (int triIdx : state.vertices[vertex].adjacentTriangles) {
        const Triangle& tri = state.triangles[triIdx];
        if (tri.deleted) continue;
        
        for (int i = 0; i < 3; ++i) {
            if (tri.v[i] != vertex && !state.vertices[tri.v[i]].deleted) {
                adjacentVertices.insert(tri.v[i]);
            }
        }
    }
    
    // Recompute and add edges to queue
    for (int adjVertex : adjacentVertices) {
        Vec3 optimalPos;
        double cost = ComputeEdgeCost(state, vertex, adjVertex, optimalPos);
        
        Edge edge(vertex, adjVertex);
        edge.cost = cost;
        edge.optimalPos = optimalPos;
        
        state.edgeQueue.push(edge);
    }
}

bool ClothMeshSimplifier::IsBoundaryEdge(const SimplificationState& state, int v0, int v1) {
    // An edge is a boundary edge if it's only referenced by one triangle
    int refCount = 0;
    
    for (const Triangle& tri : state.triangles) {
        if (tri.deleted) continue;
        
        bool hasV0 = tri.HasVertex(v0);
        bool hasV1 = tri.HasVertex(v1);
        
        if (hasV0 && hasV1) {
            refCount++;
            if (refCount > 1) {
                return false;
            }
        }
    }
    
    return refCount == 1;
}

void ClothMeshSimplifier::RemoveDegenerateTriangles(SimplificationState& state) {
    int removedCount = 0;
    
    for (Triangle& tri : state.triangles) {
        if (tri.deleted) continue;
        
        // Check for duplicate vertices
        if (tri.v[0] == tri.v[1] || tri.v[1] == tri.v[2] || tri.v[2] == tri.v[0]) {
            tri.deleted = true;
            removedCount++;
            continue;
        }
        
        // Check for zero-area triangles
        Vec3 p0 = state.vertices[tri.v[0]].position;
        Vec3 p1 = state.vertices[tri.v[1]].position;
        Vec3 p2 = state.vertices[tri.v[2]].position;
        
        Vec3 edge1 = p1 - p0;
        Vec3 edge2 = p2 - p0;
        Vec3 normal = edge1.Cross(edge2);
        
        if (normal.Length() < 1e-6f) {
            tri.deleted = true;
            removedCount++;
        }
    }
    
    if (removedCount > 0) {
        std::cout << "ClothMeshSimplifier: Removed " << removedCount 
                  << " degenerate triangles" << std::endl;
    }
}

ClothMeshSimplifier::SimplificationResult ClothMeshSimplifier::ExtractResult(
    const SimplificationState& state)
{
    SimplificationResult result;
    
    // Build mapping from old vertex indices to new compact indices
    std::vector<int> oldToNew(state.vertices.size(), -1);
    int newIndex = 0;
    
    for (size_t i = 0; i < state.vertices.size(); ++i) {
        if (!state.vertices[i].deleted) {
            oldToNew[i] = newIndex++;
        }
    }
    
    result.simplifiedVertexCount = newIndex;
    
    // Extract vertex positions
    result.positions.resize(result.simplifiedVertexCount);
    for (size_t i = 0; i < state.vertices.size(); ++i) {
        if (!state.vertices[i].deleted) {
            result.positions[oldToNew[i]] = state.vertices[i].position;
        }
    }
    
    // Extract triangles
    for (const Triangle& tri : state.triangles) {
        if (tri.deleted) continue;
        
        int i0 = oldToNew[tri.v[0]];
        int i1 = oldToNew[tri.v[1]];
        int i2 = oldToNew[tri.v[2]];
        
        if (i0 >= 0 && i1 >= 0 && i2 >= 0) {
            result.indices.push_back(i0);
            result.indices.push_back(i1);
            result.indices.push_back(i2);
        }
    }
    
    result.simplifiedTriangleCount = static_cast<int>(result.indices.size()) / 3;
    
    // Build vertex mapping (original -> simplified)
    result.vertexMapping.resize(state.vertexMapping.size());
    for (size_t i = 0; i < state.vertexMapping.size(); ++i) {
        int mappedVertex = state.vertexMapping[i];
        result.vertexMapping[i] = oldToNew[mappedVertex];
    }
    
    return result;
}
