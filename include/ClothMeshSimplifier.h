#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

/**
 * @brief Mesh simplification using Quadric Error Metrics
 * 
 * Implements the algorithm from "Surface Simplification Using Quadric Error Metrics"
 * by Garland & Heckbert (SIGGRAPH 1997)
 */
class ClothMeshSimplifier {
public:
    /**
     * @brief Result of mesh simplification operation
     */
    struct SimplificationResult {
        std::vector<Vec3> positions;           // Simplified vertex positions
        std::vector<int> indices;              // Simplified triangle indices
        std::vector<int> vertexMapping;        // Maps original vertices to simplified vertices
        int originalVertexCount;
        int simplifiedVertexCount;
        int originalTriangleCount;
        int simplifiedTriangleCount;
        bool success;
        
        SimplificationResult()
            : originalVertexCount(0)
            , simplifiedVertexCount(0)
            , originalTriangleCount(0)
            , simplifiedTriangleCount(0)
            , success(false)
        {}
    };
    
    /**
     * @brief Simplify mesh to target vertex count
     * @param positions Input vertex positions
     * @param indices Input triangle indices (3 per triangle)
     * @param targetVertexCount Target number of vertices
     * @return Simplification result
     */
    static SimplificationResult Simplify(
        const std::vector<Vec3>& positions,
        const std::vector<int>& indices,
        int targetVertexCount
    );
    
    /**
     * @brief Simplify mesh by reduction ratio
     * @param positions Input vertex positions
     * @param indices Input triangle indices (3 per triangle)
     * @param reductionRatio Ratio of vertices to keep (0.0-1.0)
     * @return Simplification result
     */
    static SimplificationResult SimplifyByRatio(
        const std::vector<Vec3>& positions,
        const std::vector<int>& indices,
        float reductionRatio
    );

private:
    /**
     * @brief 4x4 Symmetric matrix for quadric error metric
     */
    struct Quadric {
        double m[10];  // Symmetric 4x4 matrix stored as 10 unique values
        
        Quadric();
        Quadric(double a, double b, double c, double d);  // Plane equation ax+by+cz+d=0
        
        Quadric operator+(const Quadric& q) const;
        Quadric& operator+=(const Quadric& q);
        
        // Evaluate quadric at point (x, y, z, 1)
        double Evaluate(const Vec3& v) const;
        
        // Find optimal vertex position that minimizes error
        bool FindOptimalPosition(Vec3& result) const;
    };
    
    /**
     * @brief Edge in the mesh
     */
    struct Edge {
        int v0, v1;           // Vertex indices
        double cost;          // Collapse cost
        Vec3 optimalPos;      // Optimal position after collapse
        
        Edge(int a, int b) : v0(a), v1(b), cost(0.0) {}
        
        bool operator>(const Edge& other) const {
            return cost > other.cost;  // Min-heap
        }
    };
    
    /**
     * @brief Triangle in the mesh
     */
    struct Triangle {
        int v[3];             // Vertex indices
        bool deleted;         // Marked for deletion
        
        Triangle(int v0, int v1, int v2) : deleted(false) {
            v[0] = v0; v[1] = v1; v[2] = v2;
        }
        
        bool HasVertex(int vertex) const {
            return v[0] == vertex || v[1] == vertex || v[2] == vertex;
        }
        
        void ReplaceVertex(int oldVertex, int newVertex) {
            for (int i = 0; i < 3; ++i) {
                if (v[i] == oldVertex) {
                    v[i] = newVertex;
                    return;
                }
            }
        }
    };
    
    /**
     * @brief Vertex in the mesh
     */
    struct Vertex {
        Vec3 position;
        Quadric q;
        bool deleted;
        std::unordered_set<int> adjacentTriangles;  // Triangle indices
        
        Vertex() : deleted(false) {}
        Vertex(const Vec3& pos) : position(pos), deleted(false) {}
    };
    
    /**
     * @brief Internal simplification state
     */
    struct SimplificationState {
        std::vector<Vertex> vertices;
        std::vector<Triangle> triangles;
        std::vector<int> vertexMapping;  // Maps original to current vertex index
        
        // Priority queue for edge collapses
        std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> edgeQueue;
        
        // Track which edges have been processed
        std::unordered_set<uint64_t> processedEdges;
    };
    
    // Helper methods
    static void InitializeState(
        SimplificationState& state,
        const std::vector<Vec3>& positions,
        const std::vector<int>& indices
    );
    
    static void ComputeQuadrics(SimplificationState& state);
    static void BuildEdgeQueue(SimplificationState& state);
    static bool CollapseEdge(SimplificationState& state, const Edge& edge);
    static void UpdateEdgeCosts(SimplificationState& state, int vertex);
    static double ComputeEdgeCost(
        const SimplificationState& state,
        int v0, int v1,
        Vec3& optimalPos
    );
    
    static bool IsBoundaryEdge(const SimplificationState& state, int v0, int v1);
    static void RemoveDegenerateTriangles(SimplificationState& state);
    
    static SimplificationResult ExtractResult(const SimplificationState& state);
    
    // Edge hash for tracking
    static uint64_t EdgeHash(int v0, int v1) {
        if (v0 > v1) std::swap(v0, v1);
        return (static_cast<uint64_t>(v0) << 32) | static_cast<uint64_t>(v1);
    }
};
