#include "QuickHull.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

QuickHull::QuickHull()
    : m_Epsilon(1e-6f)
    , m_EnableFaceMerging(false)
    , m_FaceMergeAngleThreshold(0.01f)
    , m_UseParallel(false)
    , m_Points(nullptr)
    , m_PointCount(0)
{
}

QuickHull::QuickHull(float epsilon)
    : m_Epsilon(epsilon)
    , m_EnableFaceMerging(false)
    , m_FaceMergeAngleThreshold(0.01f)
    , m_UseParallel(false)
    , m_Points(nullptr)
    , m_PointCount(0)
{
}

void QuickHull::SetEpsilon(float epsilon) {
    m_Epsilon = std::max(epsilon, 1e-10f);
}

void QuickHull::SetFaceMerging(bool enabled, float angleThreshold) {
    m_EnableFaceMerging = enabled;
    m_FaceMergeAngleThreshold = angleThreshold;
}

ConvexHull QuickHull::ComputeHull(const Vec3* points, int count) {
    // Validate input
    if (!points || count < 4) {
        return ConvexHull();
    }
    
    // Initialize working data
    m_Points = points;
    m_PointCount = count;
    m_Faces.clear();
    m_HullVertexIndices.clear();
    
    // Build initial simplex (tetrahedron)
    if (!BuildInitialSimplex()) {
        return HandleDegenerateCase();
    }

    // Assign remaining points to faces
    AssignPointsToFaces();
    
    // Expand hull iteratively
    ExpandHull();
    
    // Optional: merge coplanar faces
    if (m_EnableFaceMerging) {
        MergeCoplanarFaces();
    }
    
    // Build and return result
    return BuildResult();
}

void QuickHull::SetParallel(bool enabled) {
    m_UseParallel = enabled;
}

bool QuickHull::BuildInitialSimplex() {
    // Find extreme points in each axis
    int minX, maxX, minY, maxY, minZ, maxZ;
    FindExtremPoints(minX, maxX, minY, maxY, minZ, maxZ);
    
    // Find the two most distant extreme points
    std::vector<int> extremes = {minX, maxX, minY, maxY, minZ, maxZ};
    std::sort(extremes.begin(), extremes.end());
    extremes.erase(std::unique(extremes.begin(), extremes.end()), extremes.end());
    
    if (extremes.size() < 4) {
        return false; // Degenerate case
    }
    
    // Find the two most distant points to form initial edge
    int p0 = extremes[0];
    int p1 = extremes[1];
    float maxDist = (m_Points[p0] - m_Points[p1]).LengthSquared();
    
    for (size_t i = 0; i < extremes.size(); ++i) {
        for (size_t j = i + 1; j < extremes.size(); ++j) {
            float dist = (m_Points[extremes[i]] - m_Points[extremes[j]]).LengthSquared();
            if (dist > maxDist) {
                maxDist = dist;
                p0 = extremes[i];
                p1 = extremes[j];
            }
        }
    }
    
    if (maxDist < m_Epsilon * m_Epsilon) {
        return false; // All points are coincident
    }
    
    // Find point furthest from line p0-p1 to form triangle
    Vec3 edge = m_Points[p1] - m_Points[p0];
    float edgeLen = edge.Length();
    if (edgeLen < m_Epsilon) {
        return false;
    }
    edge = edge * (1.0f / edgeLen);
    
    int p2 = -1;
    float maxDistToLine = 0.0f;
    for (int i = 0; i < m_PointCount; ++i) {
        if (i == p0 || i == p1) continue;
        
        Vec3 toPoint = m_Points[i] - m_Points[p0];
        float proj = toPoint.Dot(edge);
        Vec3 perpendicular = toPoint - edge * proj;
        float dist = perpendicular.LengthSquared();
        
        if (dist > maxDistToLine) {
            maxDistToLine = dist;
            p2 = i;
        }
    }
    
    if (p2 == -1 || maxDistToLine < m_Epsilon * m_Epsilon) {
        return false; // All points are collinear
    }
    
    // Find point furthest from plane p0-p1-p2 to form tetrahedron
    Vec3 edge1 = m_Points[p1] - m_Points[p0];
    Vec3 edge2 = m_Points[p2] - m_Points[p0];
    Vec3 normal = edge1.Cross(edge2);
    float normalLen = normal.Length();
    
    if (normalLen < m_Epsilon) {
        return false; // Triangle is degenerate
    }
    normal = normal * (1.0f / normalLen);
    
    int p3 = -1;
    float maxDistToPlane = 0.0f;
    for (int i = 0; i < m_PointCount; ++i) {
        if (i == p0 || i == p1 || i == p2) continue;
        
        Vec3 toPoint = m_Points[i] - m_Points[p0];
        float dist = std::abs(toPoint.Dot(normal));
        
        if (dist > maxDistToPlane) {
            maxDistToPlane = dist;
            p3 = i;
        }
    }
    
    if (p3 == -1 || maxDistToPlane < m_Epsilon) {
        return false; // All points are coplanar
    }
    
    // Create initial tetrahedron with 4 faces
    // Ensure normals point outward
    Vec3 center = (m_Points[p0] + m_Points[p1] + m_Points[p2] + m_Points[p3]) * 0.25f;
    
    auto createFace = [&](int v0, int v1, int v2) -> Face* {
        auto face = std::make_unique<Face>();
        face->v0 = v0;
        face->v1 = v1;
        face->v2 = v2;
        ComputeFaceProperties(face.get());
        
        // Check if normal points outward
        Vec3 faceCenter = (m_Points[v0] + m_Points[v1] + m_Points[v2]) * (1.0f / 3.0f);
        Vec3 toCenter = center - faceCenter;
        if (toCenter.Dot(face->normal) > 0) {
            // Flip winding order
            std::swap(face->v1, face->v2);
            ComputeFaceProperties(face.get());
        }
        
        Face* ptr = face.get();
        m_Faces.push_back(std::move(face));
        return ptr;
    };
    
    createFace(p0, p1, p2);
    createFace(p0, p3, p1);
    createFace(p0, p2, p3);
    createFace(p1, p3, p2);
    
    // Update neighbors
    UpdateFaceNeighbors();
    
    return true;
}

void QuickHull::ComputeFaceProperties(Face* face) {
    const Vec3& v0 = m_Points[face->v0];
    const Vec3& v1 = m_Points[face->v1];
    const Vec3& v2 = m_Points[face->v2];
    
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    face->normal = edge1.Cross(edge2);
    
    float len = face->normal.Length();
    if (len > m_Epsilon) {
        face->normal = face->normal * (1.0f / len);
    }
    
    face->planeDistance = face->normal.Dot(v0);
}

// Include necessary headers for parallelism
#include <future>
#include <mutex>

void QuickHull::AssignPointsToFaces() {
    // Clear existing assignments
    for (auto& face : m_Faces) {
        face->outsidePoints.clear();
        face->furthestPoint = -1;
        face->furthestDistance = 0.0f;
    }
    
    // Use parallel execution if enabled and we have enough points to justify it
    if (m_UseParallel && m_PointCount > 2000) {
        std::mutex faceMutex;
        
        // Lambda to process a chunk of points
        auto processChunk = [&](int start, int end) {
            // Local storage to minimize locking
            struct FaceUpdate {
                std::vector<int> points;
                int furthestPoint = -1;
                float furthestDistance = 0.0f;
            };
            std::vector<FaceUpdate> updates(m_Faces.size());
            
            for (int i = start; i < end; ++i) {
                for (size_t f = 0; f < m_Faces.size(); ++f) {
                    Face* face = m_Faces[f].get();
                    if (face->visible) continue;
                    
                    float dist = PointToFaceDistance(i, face);
                    if (dist > m_Epsilon) {
                        updates[f].points.push_back(i);
                        
                        if (dist > updates[f].furthestDistance) {
                            updates[f].furthestDistance = dist;
                            updates[f].furthestPoint = i;
                        }
                    }
                }
            }
            
            // Merge results
            std::lock_guard<std::mutex> lock(faceMutex);
            for (size_t f = 0; f < m_Faces.size(); ++f) {
                Face* face = m_Faces[f].get();
                if (!updates[f].points.empty()) {
                    face->outsidePoints.insert(face->outsidePoints.end(), 
                                             updates[f].points.begin(), 
                                             updates[f].points.end());
                    
                    if (updates[f].furthestDistance > face->furthestDistance) {
                        face->furthestDistance = updates[f].furthestDistance;
                        face->furthestPoint = updates[f].furthestPoint;
                    }
                }
            }
        };
        
        // Split work into chunks
        unsigned int threadCount = std::thread::hardware_concurrency();
        if (threadCount == 0) threadCount = 4;
        
        int chunkSize = m_PointCount / threadCount;
        std::vector<std::future<void>> futures;
        
        for (unsigned int i = 0; i < threadCount; ++i) {
            int start = i * chunkSize;
            int end = (i == threadCount - 1) ? m_PointCount : (start + chunkSize);
            futures.push_back(std::async(std::launch::async, processChunk, start, end));
        }
        
        // Wait for completion
        for (auto& f : futures) {
            f.wait();
        }
    } else {
        // Sequential implementation
        for (int i = 0; i < m_PointCount; ++i) {
            for (auto& face : m_Faces) {
                if (face->visible) continue;
                
                float dist = PointToFaceDistance(i, face.get());
                if (dist > m_Epsilon) {
                    face->outsidePoints.push_back(i);
                    
                    if (dist > face->furthestDistance) {
                        face->furthestDistance = dist;
                        face->furthestPoint = i;
                    }
                }
            }
        }
    }
}

void QuickHull::ExpandHull() {
    bool hasOutsidePoints = true;
    
    while (hasOutsidePoints) {
        hasOutsidePoints = false;
        
        // Find face with furthest outside point
        Face* selectedFace = nullptr;
        float maxDist = 0.0f;
        
        for (auto& face : m_Faces) {
            if (face->visible) continue;
            
            if (face->furthestPoint != -1 && face->furthestDistance > maxDist) {
                maxDist = face->furthestDistance;
                selectedFace = face.get();
                hasOutsidePoints = true;
            }
        }
        
        if (selectedFace) {
            AddPointToHull(selectedFace);
        }
    }
}

void QuickHull::AddPointToHull(Face* face) {
    int pointIndex = face->furthestPoint;
    
    // Find horizon edges
    std::vector<HalfEdge> horizon;
    FindHorizon(pointIndex, horizon);
    
    // Create new faces from horizon to new point
    CreateNewFaces(pointIndex, horizon);
    
    // Remove visible faces
    RemoveVisibleFaces();
    
    // Update neighbors
    UpdateFaceNeighbors();
}

void QuickHull::FindHorizon(int pointIndex, std::vector<HalfEdge>& horizon) {
    horizon.clear();
    
    // Mark all faces visible from the point
    for (auto& face : m_Faces) {
        if (face->visible) continue;
        
        float dist = PointToFaceDistance(pointIndex, face.get());
        if (dist > m_Epsilon) {
            face->visible = true;
        }
    }
    
    // Find edges between visible and non-visible faces
    for (auto& face : m_Faces) {
        if (!face->visible) continue;
        
        // Check each edge
        auto checkEdge = [&](int v0, int v1, Face* neighbor) {
            if (neighbor && !neighbor->visible) {
                // This edge is on the horizon
                horizon.push_back(HalfEdge(v0, v1, face.get()));
            }
        };
        
        checkEdge(face->v1, face->v2, face->neighbor0);
        checkEdge(face->v0, face->v2, face->neighbor1);
        checkEdge(face->v0, face->v1, face->neighbor2);
    }
}

void QuickHull::CreateNewFaces(int pointIndex, const std::vector<HalfEdge>& horizon) {
    // Collect all points that need to be reassigned
    std::vector<int> pointsToReassign;
    for (auto& oldFace : m_Faces) {
        if (!oldFace->visible) continue;
        
        pointsToReassign.insert(pointsToReassign.end(), 
                              oldFace->outsidePoints.begin(), 
                              oldFace->outsidePoints.end());
    }
    
    // Remove current point
    auto it = std::remove(pointsToReassign.begin(), pointsToReassign.end(), pointIndex);
    pointsToReassign.erase(it, pointsToReassign.end());
    
    // Create new faces
    int firstNewFaceIdx = static_cast<int>(m_Faces.size());
    for (const auto& edge : horizon) {
        auto newFace = std::make_unique<Face>();
        newFace->v0 = edge.v0;
        newFace->v1 = edge.v1;
        newFace->v2 = pointIndex;
        ComputeFaceProperties(newFace.get());
        m_Faces.push_back(std::move(newFace));
    }
    
    // Assign points to new faces
    if (m_UseParallel && pointsToReassign.size() > 2000) {
        std::mutex faceMutex;
        int newFacesCount = static_cast<int>(m_Faces.size()) - firstNewFaceIdx;
        
        auto processChunk = [&](int start, int end) {
            struct FaceUpdate {
                std::vector<int> points;
                int furthestPoint = -1;
                float furthestDistance = 0.0f;
            };
            std::vector<FaceUpdate> updates(newFacesCount);
            
            for (int i = start; i < end; ++i) {
                int p = pointsToReassign[i];
                
                for (int f = 0; f < newFacesCount; ++f) {
                    Face* face = m_Faces[firstNewFaceIdx + f].get();
                    
                    float dist = PointToFaceDistance(p, face);
                    if (dist > m_Epsilon) {
                        updates[f].points.push_back(p);
                        if (dist > updates[f].furthestDistance) {
                            updates[f].furthestDistance = dist;
                            updates[f].furthestPoint = p;
                        }
                    }
                }
            }
            
            std::lock_guard<std::mutex> lock(faceMutex);
            for (int f = 0; f < newFacesCount; ++f) {
                Face* face = m_Faces[firstNewFaceIdx + f].get();
                if (!updates[f].points.empty()) {
                    face->outsidePoints.insert(face->outsidePoints.end(), 
                                             updates[f].points.begin(), 
                                             updates[f].points.end());
                    
                    if (updates[f].furthestDistance > face->furthestDistance) {
                        face->furthestDistance = updates[f].furthestDistance;
                        face->furthestPoint = updates[f].furthestPoint;
                    }
                }
            }
        };
        
        unsigned int threadCount = std::thread::hardware_concurrency();
        if (threadCount == 0) threadCount = 4;
        
        int chunkSize = static_cast<int>(pointsToReassign.size()) / threadCount;
        std::vector<std::future<void>> futures;
        
        for (unsigned int i = 0; i < threadCount; ++i) {
            int start = i * chunkSize;
            int end = (i == threadCount - 1) ? static_cast<int>(pointsToReassign.size()) : (start + chunkSize);
            futures.push_back(std::async(std::launch::async, processChunk, start, end));
        }
        
        for (auto& f : futures) {
            f.wait();
        }
    } else {
        // Sequential reassignment
        for (int p : pointsToReassign) {
            for (size_t f = firstNewFaceIdx; f < m_Faces.size(); ++f) {
                Face* face = m_Faces[f].get();
                
                float dist = PointToFaceDistance(p, face);
                if (dist > m_Epsilon) {
                    face->outsidePoints.push_back(p);
                    
                    if (dist > face->furthestDistance) {
                        face->furthestDistance = dist;
                        face->furthestPoint = p;
                    }
                }
            }
        }
    }
}

void QuickHull::RemoveVisibleFaces() {
    m_Faces.erase(
        std::remove_if(m_Faces.begin(), m_Faces.end(),
            [](const std::unique_ptr<Face>& face) { return face->visible; }),
        m_Faces.end()
    );
}

void QuickHull::UpdateFaceNeighbors() {
    // Build edge-to-face map
    struct EdgeKey {
        int v0, v1;
        
        EdgeKey(int a, int b) {
            if (a < b) { v0 = a; v1 = b; }
            else { v0 = b; v1 = a; }
        }
        
        bool operator==(const EdgeKey& other) const {
            return v0 == other.v0 && v1 == other.v1;
        }
    };
    
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& key) const {
            return std::hash<int>()(key.v0) ^ (std::hash<int>()(key.v1) << 1);
        }
    };
    
    std::unordered_map<EdgeKey, Face*, EdgeKeyHash> edgeMap;
    
    // First pass: clear neighbors
    for (auto& face : m_Faces) {
        face->neighbor0 = nullptr;
        face->neighbor1 = nullptr;
        face->neighbor2 = nullptr;
    }
    
    // Second pass: build edge map and assign neighbors
    for (auto& face : m_Faces) {
        auto assignNeighbor = [&](int v0, int v1, Face** neighborSlot) {
            EdgeKey key(v0, v1);
            auto it = edgeMap.find(key);
            if (it != edgeMap.end()) {
                *neighborSlot = it->second;
                // Also update the neighbor's corresponding slot
                Face* neighbor = it->second;
                if (EdgeKey(neighbor->v1, neighbor->v2) == key) neighbor->neighbor0 = face.get();
                else if (EdgeKey(neighbor->v0, neighbor->v2) == key) neighbor->neighbor1 = face.get();
                else if (EdgeKey(neighbor->v0, neighbor->v1) == key) neighbor->neighbor2 = face.get();
            } else {
                edgeMap[key] = face.get();
            }
        };
        
        assignNeighbor(face->v1, face->v2, &face->neighbor0);
        assignNeighbor(face->v0, face->v2, &face->neighbor1);
        assignNeighbor(face->v0, face->v1, &face->neighbor2);
    }
}

ConvexHull QuickHull::BuildResult() {
    ConvexHull result;
    
    // Build vertex index map
    std::unordered_map<int, int> vertexMap;
    for (const auto& face : m_Faces) {
        if (vertexMap.find(face->v0) == vertexMap.end()) {
            vertexMap[face->v0] = static_cast<int>(result.vertices.size());
            result.vertices.push_back(m_Points[face->v0]);
        }
        if (vertexMap.find(face->v1) == vertexMap.end()) {
            vertexMap[face->v1] = static_cast<int>(result.vertices.size());
            result.vertices.push_back(m_Points[face->v1]);
        }
        if (vertexMap.find(face->v2) == vertexMap.end()) {
            vertexMap[face->v2] = static_cast<int>(result.vertices.size());
            result.vertices.push_back(m_Points[face->v2]);
        }
    }
    
    // Build face indices and calculate surface area
    result.surfaceArea = 0.0f;
    result.faceCount = static_cast<int>(m_Faces.size());
    
    for (const auto& face : m_Faces) {
        result.indices.push_back(vertexMap[face->v0]);
        result.indices.push_back(vertexMap[face->v1]);
        result.indices.push_back(vertexMap[face->v2]);
        result.faceNormals.push_back(face->normal);
        
        result.surfaceArea += CalculateFaceArea(face.get());
    }
    
    return result;
}

float QuickHull::PointToFaceDistance(int pointIndex, const Face* face) const {
    return m_Points[pointIndex].Dot(face->normal) - face->planeDistance;
}

bool QuickHull::IsPointOutsideFace(int pointIndex, const Face* face) const {
    return PointToFaceDistance(pointIndex, face) > m_Epsilon;
}

float QuickHull::CalculateFaceArea(const Face* face) const {
    const Vec3& v0 = m_Points[face->v0];
    const Vec3& v1 = m_Points[face->v1];
    const Vec3& v2 = m_Points[face->v2];
    
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    return 0.5f * edge1.Cross(edge2).Length();
}

void QuickHull::MergeCoplanarFaces() {
    // TODO: Implement face merging for nearly coplanar faces
    // This is an optional optimization
}

void QuickHull::FindExtremPoints(int& minX, int& maxX, int& minY, int& maxY, int& minZ, int& maxZ) const {
    minX = maxX = minY = maxY = minZ = maxZ = 0;
    
    for (int i = 1; i < m_PointCount; ++i) {
        if (m_Points[i].x < m_Points[minX].x) minX = i;
        if (m_Points[i].x > m_Points[maxX].x) maxX = i;
        if (m_Points[i].y < m_Points[minY].y) minY = i;
        if (m_Points[i].y > m_Points[maxY].y) maxY = i;
        if (m_Points[i].z < m_Points[minZ].z) minZ = i;
        if (m_Points[i].z > m_Points[maxZ].z) maxZ = i;
    }
}

int QuickHull::FindExtremePoint(const Vec3& direction) const {
    int bestIdx = 0;
    float bestDot = m_Points[0].Dot(direction);
    
    for (int i = 1; i < m_PointCount; ++i) {
        float dot = m_Points[i].Dot(direction);
        if (dot > bestDot) {
            bestDot = dot;
            bestIdx = i;
        }
    }
    
    return bestIdx;
}

bool QuickHull::ArePointsCollinear(const std::vector<int>& indices) const {
    if (indices.size() < 3) return true;
    
    Vec3 edge = m_Points[indices[1]] - m_Points[indices[0]];
    float edgeLen = edge.Length();
    if (edgeLen < m_Epsilon) return true;
    edge = edge * (1.0f / edgeLen);
    
    for (size_t i = 2; i < indices.size(); ++i) {
        Vec3 toPoint = m_Points[indices[i]] - m_Points[indices[0]];
        float proj = toPoint.Dot(edge);
        Vec3 perpendicular = toPoint - edge * proj;
        if (perpendicular.Length() > m_Epsilon) {
            return false;
        }
    }
    
    return true;
}

bool QuickHull::ArePointsCoplanar(const std::vector<int>& indices) const {
    if (indices.size() < 4) return true;
    
    Vec3 edge1 = m_Points[indices[1]] - m_Points[indices[0]];
    Vec3 edge2 = m_Points[indices[2]] - m_Points[indices[0]];
    Vec3 normal = edge1.Cross(edge2);
    float normalLen = normal.Length();
    
    if (normalLen < m_Epsilon) return true;
    normal = normal * (1.0f / normalLen);
    
    float planeDist = normal.Dot(m_Points[indices[0]]);
    
    for (size_t i = 3; i < indices.size(); ++i) {
        float dist = std::abs(normal.Dot(m_Points[indices[i]]) - planeDist);
        if (dist > m_Epsilon) {
            return false;
        }
    }
    
    return true;
}

ConvexHull QuickHull::HandleDegenerateCase() {
    // For degenerate cases, return empty hull
    // Could be extended to handle 2D hulls or line segments
    return ConvexHull();
}
