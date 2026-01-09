#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

/**
 * @brief Tetrahedral mesh simplification using Quadric Error Metrics
 * 
 * Adapts the Garland & Heckbert algorithm for volumetric tetrahedral meshes.
 * Preserves volume and surface integrity while reducing vertex count.
 */
class TetrahedralMeshSimplifier {
public:
    /**
     * @brief Result of mesh simplification operation
     */
    struct SimplificationResult {
        std::vector<Vec3> positions;           // Simplified vertex positions
        std::vector<int> indices;              // Simplified tetrahedral indices (4 per tetrahedron)
        std::vector<int> vertexMapping;        // Maps original vertices to simplified vertices
        int originalVertexCount;
        int simplifiedVertexCount;
        int originalTetrahedronCount;
        int simplifiedTetrahedronCount;
        bool success;
        
        SimplificationResult()
            : originalVertexCount(0)
            , simplifiedVertexCount(0)
            , originalTetrahedronCount(0)
            , simplifiedTetrahedronCount(0)
            , success(false)
        {}
    };
    
    /**
     * @brief Simplify mesh to target vertex count
     * @param positions Input vertex positions
     * @param indices Input tetrahedral indices (4 per tetrahedron)
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
     * @param indices Input tetrahedral indices (4 per tetrahedron)
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
     * @brief Tetrahedron in the mesh
     */
    struct Tetrahedron {
        int v[4];             // Vertex indices
        bool deleted;         // Marked for deletion
        
        Tetrahedron(int v0, int v1, int v2, int v3) : deleted(false) {
            v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
        }
        
        bool HasVertex(int vertex) const {
            return v[0] == vertex || v[1] == vertex || v[2] == vertex || v[3] == vertex;
        }
        
        void ReplaceVertex(int oldVertex, int newVertex) {
            for (int i = 0; i < 4; ++i) {
                if (v[i] == oldVertex) {
                    v[i] = newVertex;
                    return;
                }
            }
        }
        
        // Get the 4 triangular faces of the tetrahedron
        void GetFaces(int faces[4][3]) const {
            // Face 0: v0, v1, v2
            faces[0][0] = v[0]; faces[0][1] = v[1]; faces[0][2] = v[2];
            // Face 1: v0, v1, v3
            faces[1][0] = v[0]; faces[1][1] = v[1]; faces[1][2] = v[3];
            // Face 2: v0, v2, v3
            faces[2][0] = v[0]; faces[2][1] = v[2]; faces[2][2] = v[3];
            // Face 3: v1, v2, v3
            faces[3][0] = v[1]; faces[3][1] = v[2]; faces[3][2] = v[3];
        }
    };
    
    /**
     * @brief Vertex in the mesh
     */
    struct Vertex {
        Vec3 position;
        Quadric q;
        bool deleted;
        bool isSurface;  // True if vertex is on the surface
        std::unordered_set<int> adjacentTetrahedra;  // Tetrahedron indices
        
        Vertex() : deleted(false), isSurface(false) {}
        Vertex(const Vec3& pos) : position(pos), deleted(false), isSurface(false) {}
    };
    
    /**
     * @brief Internal simplification state
     */
    struct SimplificationState {
        std::vector<Vertex> vertices;
        std::vector<Tetrahedron> tetrahedra;
        std::vector<int> vertexMapping;  // Maps original to current vertex index
        
        // Priority queue for edge collapses
        std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> edgeQueue;
        
        // Track which edges have been processed
        std::unordered_set<uint64_t> processedEdges;
        
        // Surface face tracking (for boundary detection)
        std::unordered_map<uint64_t, int> faceCount;  // Face hash -> count
    };
    
    // Helper methods
    static void InitializeState(
        SimplificationState& state,
        const std::vector<Vec3>& positions,
        const std::vector<int>& indices
    );
    
    static void IdentifySurfaceVertices(SimplificationState& state);
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
    static void RemoveDegenerateTetrahedra(SimplificationState& state);
    
    static SimplificationResult ExtractResult(const SimplificationState& state);
    
    // Face hash for tracking (orders vertices consistently)
    static uint64_t FaceHash(int v0, int v1, int v2);
    
    // Edge hash for tracking
    static uint64_t EdgeHash(int v0, int v1) {
        if (v0 > v1) std::swap(v0, v1);
        return (static_cast<uint64_t>(v0) << 32) | static_cast<uint64_t>(v1);
    }
};
