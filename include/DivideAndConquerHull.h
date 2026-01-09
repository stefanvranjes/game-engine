#pragma once

#include "Math/Vec3.h"
#include "QuickHull.h" // For ConvexHull struct
#include <vector>
#include <memory>
#include <list>
#include <algorithm>

/**
 * @brief Divide and Conquer algorithm for computing 3D convex hulls
 * 
 * Sorts points by X-coordinate, recursively constructs hulls for left/right subsets,
 * and merges them by wrapping a band of faces around.
 * Time Complexity: O(n log n)
 */
class DivideAndConquerHull {
public:
    explicit DivideAndConquerHull(float epsilon = 1e-6f);
    
    ConvexHull ComputeHull(const Vec3* points, int count);
    
    void SetEpsilon(float epsilon);

private:
    struct Point {
        double x, y, z;
        int id; // Original index
        
        bool operator<(const Point& other) const {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }
    };
    
    // Internal Half-Edge Data Structure needed for efficient traversal during merge
    struct HE_Edge;
    struct HE_Face;
    struct HE_Vert;
    
    struct HE_Vert {
        int id; // Index into m_SortedPoints
        HE_Edge* edge; // Outgoing edge
        double x, y, z;
        
        HE_Vert(int i, double _x, double _y, double _z) : id(i), edge(nullptr), x(_x), y(_y), z(_z) {}
    };
    
    struct HE_Edge {
        HE_Vert* vert; // Origin vertex
        HE_Edge* pair; // Twin edge
        HE_Edge* next; // Next edge in face loop
        HE_Edge* prev; // Prev edge in face loop
        HE_Face* face; // Face this edge belongs to
        
        HE_Edge() : vert(nullptr), pair(nullptr), next(nullptr), prev(nullptr), face(nullptr) {}
    };
    
    struct HE_Face {
        HE_Edge* edge;
        bool visible; // Used for deleting during merge
        
        HE_Face() : edge(nullptr), visible(false) {}
    };
    
    struct SubHull {
        std::vector<HE_Vert*> verts;
        std::vector<HE_Edge*> edges;
        std::vector<HE_Face*> faces;
        
        ~SubHull(); // Need to clean up pointers
    };
    
    float m_Epsilon;
    std::vector<Point> m_SortedPoints;
    
    // Helpers
    SubHull* Recurse(int start, int end);
    SubHull* ConstructBaseHull(int start, int end);
    SubHull* Merge(SubHull* left, SubHull* right);
    
    ConvexHull BuildResult(SubHull* hull, const Vec3* originalPoints);
};
