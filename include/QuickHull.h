#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <memory>

/**
 * @brief Result structure containing the computed convex hull
 */
struct ConvexHull {
    std::vector<Vec3> vertices;        // Hull vertices (subset of input points)
    std::vector<int> indices;          // Triangle indices (3 per face)
    std::vector<Vec3> faceNormals;     // Normal for each face
    float surfaceArea;                 // Total surface area
    int faceCount;                     // Number of faces
    
    ConvexHull() : surfaceArea(0.0f), faceCount(0) {}
};

/**
 * @brief QuickHull algorithm for computing 3D convex hulls
 * 
 * Implements the QuickHull algorithm for efficient convex hull computation
 * in 3D space. Average complexity is O(n log n), worst case O(n²).
 * 
 * Features:
 * - Robust handling of degenerate cases (coplanar, collinear points)
 * - Epsilon-based numerical tolerance
 * - Efficient point-to-face assignment
 * - Automatic surface area calculation
 */
class QuickHull {
public:
    /**
     * @brief Construct QuickHull with default epsilon tolerance
     */
    QuickHull();
    
    /**
     * @brief Construct QuickHull with custom epsilon tolerance
     * @param epsilon Tolerance for floating-point comparisons (default: 1e-6)
     */
    explicit QuickHull(float epsilon);
    
    /**
     * @brief Compute convex hull from point cloud
     * @param points Array of input points
     * @param count Number of points
     * @return ConvexHull structure containing hull geometry and surface area
     */
    ConvexHull ComputeHull(const Vec3* points, int count);
    
    /**
     * @brief Set epsilon tolerance for numerical comparisons
     * @param epsilon Tolerance value (should be > 0)
     */
    void SetEpsilon(float epsilon);
    
    /**
     * @brief Get current epsilon tolerance
     */
    float GetEpsilon() const { return m_Epsilon; }
    
    /**
     * @brief Enable/disable face merging for nearly coplanar faces
     * @param enabled Whether to merge nearly coplanar faces
     * @param angleThreshold Maximum angle difference in radians (default: 0.01)
     */
    void SetFaceMerging(bool enabled, float angleThreshold = 0.01f);

    /**
     * @brief Enable/disable parallel execution
     * @param enabled Whether to use multi-threading for intensive steps
     */
    void SetParallel(bool enabled);

private:
    // Internal face structure for hull construction
    struct Face {
        int v0, v1, v2;                    // Vertex indices
        Vec3 normal;                       // Face normal (outward)
        float planeDistance;               // Distance from origin along normal
        std::vector<int> outsidePoints;    // Points outside this face
        int furthestPoint;                 // Index of furthest outside point
        float furthestDistance;            // Distance of furthest point
        bool visible;                      // Marked for removal
        
        // Neighbor faces (for topology)
        Face* neighbor0;  // Opposite edge v1-v2
        Face* neighbor1;  // Opposite edge v0-v2
        Face* neighbor2;  // Opposite edge v0-v1
        
        Face() : v0(-1), v1(-1), v2(-1), planeDistance(0.0f),
                 furthestPoint(-1), furthestDistance(0.0f), visible(false),
                 neighbor0(nullptr), neighbor1(nullptr), neighbor2(nullptr) {}
    };
    
    // Half-edge structure for horizon detection
    struct HalfEdge {
        int v0, v1;        // Edge vertices
        Face* face;        // Adjacent face
        
        HalfEdge(int a, int b, Face* f) : v0(a), v1(b), face(f) {}
        
        bool operator==(const HalfEdge& other) const {
            return (v0 == other.v0 && v1 == other.v1) ||
                   (v0 == other.v1 && v1 == other.v0);
        }
    };
    
    // Configuration
    float m_Epsilon;
    bool m_EnableFaceMerging;
    float m_FaceMergeAngleThreshold;
    bool m_UseParallel;
    
    // Working data
    const Vec3* m_Points;
    int m_PointCount;
    std::vector<std::unique_ptr<Face>> m_Faces;
    std::vector<int> m_HullVertexIndices;
    
    // Algorithm steps
    bool BuildInitialSimplex();
    void AssignPointsToFaces();
    void ExpandHull();
    void AddPointToHull(Face* face);
    void FindHorizon(int pointIndex, std::vector<HalfEdge>& horizon);
    void CreateNewFaces(int pointIndex, const std::vector<HalfEdge>& horizon);
    void RemoveVisibleFaces();
    ConvexHull BuildResult();
    
    // Helper methods
    void ComputeFaceProperties(Face* face);
    float PointToFaceDistance(int pointIndex, const Face* face) const;
    bool IsPointOutsideFace(int pointIndex, const Face* face) const;
    void UpdateFaceNeighbors();
    void MergeCoplanarFaces();
    float CalculateFaceArea(const Face* face) const;
    
    // Degenerate case handling
    bool ArePointsCollinear(const std::vector<int>& indices) const;
    bool ArePointsCoplanar(const std::vector<int>& indices) const;
    ConvexHull HandleDegenerateCase();
    
    // Utility
    int FindExtremePoint(const Vec3& direction) const;
    void FindExtremPoints(int& minX, int& maxX, int& minY, int& maxY, int& minZ, int& maxZ) const;
};
