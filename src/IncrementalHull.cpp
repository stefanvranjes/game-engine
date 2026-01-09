#include "IncrementalHull.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <map>

IncrementalHull::IncrementalHull(float epsilon)
    : m_Epsilon(epsilon)
{
}

void IncrementalHull::SetEpsilon(float epsilon) {
    m_Epsilon = std::max(epsilon, 1e-10f);
}

ConvexHull IncrementalHull::ComputeHull(const Vec3* points, int count) {
    if (!points || count < 4) {
        return ConvexHull();
    }
    
    // Copy points
    std::vector<Vec3> pointList(points, points + count);
    
    // Randomize order to avoid worst-case scenarios
    // (e.g. points sorted along a curve)
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(pointList.begin(), pointList.end(), g);
    
    // Initialize
    if (!Initialize(pointList)) {
        return ConvexHull();
    }
    
    // Add remaining points
    // We already added the first 4 in Initialize
    for (size_t i = 4; i < pointList.size(); ++i) {
        AddPoint(pointList[i]);
    }
    
    return GetResult();
}

bool IncrementalHull::Initialize(const std::vector<Vec3>& points) {
    m_Vertices.clear();
    m_Faces.clear();
    
    if (points.size() < 4) return false;
    
    // Find initial tetrahedron from first 4 points (if non-coplanar)
    // Or scan for first non-coplanar set
    
    int p0 = 0, p1 = 1, p2 = 2, p3 = 3;
    
    // Robust check for initial simplex...
    // For now assume randomized input helps, but strict robustness requires checks
    // Similar logic to QuickHull::BuildInitialSimplex but using indices into local copy
    
    m_Vertices.push_back(points[0]);
    m_Vertices.push_back(points[1]);
    m_Vertices.push_back(points[2]);
    m_Vertices.push_back(points[3]);
    
    if (!BuildInitialSimplex()) {
        // Fallback or fail
        return false;
    }
    
    return true;
}

bool IncrementalHull::AddPoint(const Vec3& point) {
    // Find visible faces
    // Simple O(F) search: check distance to each face plane
    
    bool anyVisible = false;
    for (auto& face : m_Faces) {
        float dist = PointToFaceDistance(point, face.get());
        if (dist > m_Epsilon) {
            face->visible = true;
            anyVisible = true;
        } else {
            face->visible = false;
        }
    }
    
    if (!anyVisible) {
        return false; // Point is inside
    }
    
    // Point is outside, add it to vertices
    m_Vertices.push_back(point);
    int newPointIdx = static_cast<int>(m_Vertices.size()) - 1;
    
    // Find horizon edges
    std::vector<HalfEdge> horizon;
    
    for (auto& face : m_Faces) {
        if (!face->visible) continue;
        
        // Check edges
        auto checkEdge = [&](int v0, int v1, Face* neighbor) {
            if (neighbor && !neighbor->visible) {
                // This edge is on the horizon
                horizon.emplace_back(v0, v1, neighbor);
            }
        };
        
        checkEdge(face->v1, face->v2, face->neighbor0);
        checkEdge(face->v0, face->v2, face->neighbor1);
        checkEdge(face->v0, face->v1, face->neighbor2);
    }
    
    // Create new faces
    std::vector<Face*> newFacesPtrs;
    int firstNewFaceIdx = static_cast<int>(m_Faces.size());
    
    for (const auto& edge : horizon) {
        auto newFace = std::make_unique<Face>();
        newFace->v0 = edge.v0;
        newFace->v1 = edge.v1;
        newFace->v2 = newPointIdx;
        
        // Neighbor to existing hull
        // The edge (v0, v1) is shared with edge.face
        // We need to set up adjacency correctly later, but for now:
        // We know edge.face is the neighbor across v0-v1.
        // In newFace, edge v0-v1 is opposite to v2 (neighbor2)
        newFace->neighbor2 = edge.face;
        
        // Calculate normal
        ComputeFaceProperties(newFace.get());
        
        newFacesPtrs.push_back(newFace.get());
        m_Faces.push_back(std::move(newFace));
        
        // Update the existing neighbor to point to this new face
        // We need to find which neighbor slot in edge.face corresponds to (v1, v0)
        Face* n = edge.face;
        if (n->neighbor0 && ((n->v1 == edge.v1 && n->v2 == edge.v0) || (n->v1 == edge.v0 && n->v2 == edge.v1))) {
            n->neighbor0 = newFacesPtrs.back();
        } else if (n->neighbor1 && ((n->v0 == edge.v1 && n->v2 == edge.v0) || (n->v0 == edge.v0 && n->v2 == edge.v1))) {
            n->neighbor1 = newFacesPtrs.back();
        } else if (n->neighbor2 && ((n->v0 == edge.v1 && n->v1 == edge.v0) || (n->v0 == edge.v0 && n->v1 == edge.v1))) {
            n->neighbor2 = newFacesPtrs.back();
        }
    }
    
    // Connect new faces to each other
    ConnectNewFaces(newFacesPtrs);
    
    // Remove visible faces
    RemoveVisibleFaces();
    
    return true;
}

void IncrementalHull::ConnectNewFaces(std::vector<Face*>& newFaces) {
    // New faces are arranged in a cone.
    // Each new face shares edges (v0, newPoint) and (v1, newPoint) with other new faces.
    // We can use a map to match them.
    
    struct EdgeKey {
        int v0, v1;
        bool operator<(const EdgeKey& other) const {
            if (v0 != other.v0) return v0 < other.v0;
            return v1 < other.v1;
        }
    };
    
    std::map<EdgeKey, Face*> edgeToFace;
    
    for (Face* face : newFaces) {
        // Edge 0: v1-v2 (v1-newPoint) -> neighbor0
        // Edge 1: v0-v2 (v0-newPoint) -> neighbor1
        
        // We store directed edges from the face's perspective
        edgeToFace[{face->v1, face->v2}] = face;
        edgeToFace[{face->v0, face->v2}] = face;
    }
    
    for (Face* face : newFaces) {
        // Find neighbor for edge 0 (v1->v2)
        // We look for (v2->v1) which is (newPoint->v1)
        // Wait, v2 is newPoint.
        
        // Neighbor 0 is across edge v1-v2 (v1-newPoint).
        // We look for edge (newPoint-v1).
        auto it0 = edgeToFace.find({face->v2, face->v1});
        if (it0 != edgeToFace.end()) {
            face->neighbor0 = it0->second;
        }
        
        // Neighbor 1 is across edge v0-v2 (v0-newPoint).
        // We look for edge (newPoint-v0).
        auto it1 = edgeToFace.find({face->v2, face->v0});
        if (it1 != edgeToFace.end()) {
            face->neighbor1 = it1->second;
        }
        
        // Neighbor 2 (v0-v1) is already set to the horizon face
    }
}

void IncrementalHull::RemoveVisibleFaces() {
    // Remove faces marked visible
    m_Faces.erase(
        std::remove_if(m_Faces.begin(), m_Faces.end(),
            [](const std::unique_ptr<Face>& f) { return f->visible; }),
        m_Faces.end());
}

bool IncrementalHull::BuildInitialSimplex() {
    // Assume first 4 vertices are provided in m_Vertices
    int v0 = 0, v1 = 1, v2 = 2, v3 = 3;
    
    Vec3 center = (m_Vertices[v0] + m_Vertices[v1] + m_Vertices[v2] + m_Vertices[v3]) * 0.25f;
    
    auto createFace = [&](int a, int b, int c) {
        auto face = std::make_unique<Face>();
        face->v0 = a;
        face->v1 = b;
        face->v2 = c;
        ComputeFaceProperties(face.get());
        
        // Check normal orientation
        Vec3 faceCenter = (m_Vertices[a] + m_Vertices[b] + m_Vertices[c]) / 3.0f;
        if ((faceCenter - center).Dot(face->normal) < 0) {
            std::swap(face->v1, face->v2);
            ComputeFaceProperties(face.get());
        }
        
        m_Faces.push_back(std::move(face));
    };
    
    createFace(v0, v1, v2);
    createFace(v0, v1, v3);
    createFace(v1, v2, v3);
    createFace(v2, v0, v3);
    
    // Brute-force neighbor update for initialization
    UpdateFaceNeighbors();
    
    return true;
}

void IncrementalHull::UpdateFaceNeighbors() {
    // O(F^2) neighbor finding - only used for initialization
    for (auto& f1 : m_Faces) {
        for (auto& f2 : m_Faces) {
            if (f1 == f2) continue;
            
            // Check edges
            // f1 edge 0 (v1-v2)
            if ((f1->v1 == f2->v2 && f1->v2 == f2->v1) || (f1->v1 == f2->v1 && f1->v2 == f2->v2) || (f1->v1 == f2->v0 && f1->v2 == f2->v2)) { //... simplification
                // Ideally use a map, but for 4 faces it's fine
            }
        }
    }
    
    // Better: use map
    struct EdgeKey {
        int a, b;
        bool operator<(const EdgeKey& other) const {
            if (a != other.a) return a < other.a;
            return b < other.b;
        }
    };
    std::map<EdgeKey, Face*> edgeMap;
    
    for (auto& f : m_Faces) {
        edgeMap[{std::min(f->v1, f->v2), std::max(f->v1, f->v2)}] = f.get(); // edge 0
        edgeMap[{std::min(f->v0, f->v2), std::max(f->v0, f->v2)}] = f.get(); // edge 1
        edgeMap[{std::min(f->v0, f->v1), std::max(f->v0, f->v1)}] = f.get(); // edge 2
        // Wait, this maps an edge to *a* face. It overwrites.
        // We need to link f to the *other* face that shares the edge.
    }
    // Actually, we need to find neighbors.
    // Iterate all pairs?
    
    for (auto& f : m_Faces) {
        auto findNeighbor = [&](int a, int b) -> Face* {
            for (auto& other : m_Faces) {
                if (other == f) continue;
                if ((other->v0 == a && other->v1 == b) || (other->v0 == b && other->v1 == a) ||
                    (other->v1 == a && other->v2 == b) || (other->v1 == b && other->v2 == a) ||
                    (other->v2 == a && other->v0 == b) || (other->v2 == b && other->v0 == a)) {
                    return other.get();
                }
            }
            return nullptr;
        };
        
        f->neighbor0 = findNeighbor(f->v1, f->v2);
        f->neighbor1 = findNeighbor(f->v0, f->v2);
        f->neighbor2 = findNeighbor(f->v0, f->v1);
    }
}

void IncrementalHull::ComputeFaceProperties(Face* face) {
    const Vec3& v0 = m_Vertices[face->v0];
    const Vec3& v1 = m_Vertices[face->v1];
    const Vec3& v2 = m_Vertices[face->v2];
    
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    face->normal = edge1.Cross(edge2).Normalized();
    face->planeDistance = face->normal.Dot(v0);
}

float IncrementalHull::PointToFaceDistance(const Vec3& point, const Face* face) const {
    return face->normal.Dot(point) - face->planeDistance;
}

ConvexHull IncrementalHull::GetResult() const {
    ConvexHull result;
    result.faceCount = static_cast<int>(m_Faces.size());
    result.surfaceArea = 0.0f;
    
    for (const auto& face : m_Faces) {
        result.indices.push_back(face->v0);
        result.indices.push_back(face->v1);
        result.indices.push_back(face->v2);
        result.faceNormals.push_back(face->normal);
        
        // Area
        const Vec3& p0 = m_Vertices[face->v0];
        const Vec3& p1 = m_Vertices[face->v1];
        const Vec3& p2 = m_Vertices[face->v2];
        result.surfaceArea += 0.5f * (p1 - p0).Cross(p2 - p0).Length();
    }
    
    // Copy vertices
    result.vertices = m_Vertices;
    
    return result;
}
