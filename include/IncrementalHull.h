#pragma once

#include "Math/Vec3.h"
#include "QuickHull.h" // For ConvexHull struct
#include <vector>
#include <memory>
#include <unordered_set>

/**
 * @brief Incremental algorithm for computing 3D convex hulls
 * 
 * Implements the Randomized Incremental Construction algorithm.
 * Maintains a topological hull structure that can be updated by adding points.
 */
class IncrementalHull {
public:
    explicit IncrementalHull(float epsilon = 1e-6f);
    
    /**
     * @brief Compute convex hull for a static set of points
     * Uses randomized order for O(n log n) expected performance
     */
    ConvexHull ComputeHull(const Vec3* points, int count);
    
    /**
     * @brief Initialize with a starting simplex (tetrahedron)
     * @return true if successful
     */
    bool Initialize(const std::vector<Vec3>& points);
    
    /**
     * @brief Add a single point to the current hull
     * @return true if the hull was updated (point was outside)
     */
    bool AddPoint(const Vec3& point);
    
    /**
     * @brief Get the current hull geometry
     */
    ConvexHull GetResult() const;
    
    void SetEpsilon(float epsilon);

private:
    struct Face {
        int v0, v1, v2;
        Vec3 normal;
        float planeDistance;
        bool visible; // Marked for removal during update
        
        Face* neighbor0; // Edge v1-v2
        Face* neighbor1; // Edge v0-v2
        Face* neighbor2; // Edge v0-v1
        
        Face() : v0(-1), v1(-1), v2(-1), visible(false),
                 neighbor0(nullptr), neighbor1(nullptr), neighbor2(nullptr) {}
    };
    
    struct HalfEdge {
        int v0, v1;
        Face* face;
        HalfEdge(int a, int b, Face* f) : v0(a), v1(b), face(f) {}
    };
    
    float m_Epsilon;
    std::vector<Vec3> m_Vertices; // We store local copy of vertices as we grow
    std::vector<std::unique_ptr<Face>> m_Faces;
    
    // Helper methods
    void ComputeFaceProperties(Face* face);
    void UpdateFaceNeighbors(); // Global update (expensive)
    void ConnectNewFaces(std::vector<Face*>& newFaces); // Local neighbor update
    float PointToFaceDistance(const Vec3& point, const Face* face) const;
    bool BuildInitialSimplex();
    void FindHorizon(const Vec3& point, std::vector<HalfEdge>& horizon);
    void RemoveVisibleFaces();
};
