#include "GiftWrapping.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <iostream>

GiftWrapping::GiftWrapping(float epsilon)
    : m_Epsilon(epsilon)
    , m_Points(nullptr)
    , m_PointCount(0)
{
}

void GiftWrapping::SetEpsilon(float epsilon) {
    m_Epsilon = std::max(epsilon, 1e-10f);
}

ConvexHull GiftWrapping::ComputeHull(const Vec3* points, int count) {
    // Validate input
    if (!points || count < 4) {
        return ConvexHull();
    }
    
    m_Points = points;
    m_PointCount = count;
    
    // Step 1: Find the first point (pivot)
    // Use the point with minimum Y coordinate (and min X, Z for tie-breaking)
    int p0 = FindExtremePoint();
    
    // Step 2: Find the second point to form an initial edge
    int p1 = FindSecondPoint(p0);
    if (p1 == -1) return ConvexHull(); // Degenerate
    
    // Step 3: Find the third point to form the first face
    // We assume a virtual normal for the initial edge to orient the first face search
    // Using simple axis vector perpendicular to edge p0-p1
    Vec3 edge0 = m_Points[p1] - m_Points[p0];
    Vec3 axis = Vec3(1, 0, 0);
    if (std::abs(edge0.Dot(axis) / edge0.Length()) > 0.9f) {
        axis = Vec3(0, 1, 0);
    }
    Vec3 virtualNormal = edge0.Cross(axis);
    
    int p2 = FindBestPointForEdge(p0, p1, virtualNormal);
    if (p2 == -1) return ConvexHull(); // Degenerate
    
    // We now have the first face (p0, p1, p2)
    // Ensure correct winding (CCW from outside)
    // The FindBestPointForEdge should already give consistent winding relative to reference normal
    // but the first face is arbitrary.
    // Let's ensure normal points "outward" from the bulk of points?
    // Actually, simple Gift Wrapping maintains "hull on the right" or similar invariant.
    
    // Let's store faces and manage open edges
    std::vector<Vec3> hullVertices;
    std::vector<int> hullIndices;
    std::vector<Vec3> faceNormals;
    
    // Use a set to track processed edges to avoid duplicating faces
    // We store directed edges. Attempting to add (u,v) when (u,v) exists means we are doing something wrong?
    // No, we process edges. If we processed (u,v), we found its neighbor face.
    // We need to process every hull edge exactly once.
    // An edge (u,v) on a face means we need to find the face attached to (v,u).
    
    std::unordered_set<DirectedEdge, DirectedEdgeHash> openEdges;
    std::unordered_set<DirectedEdge, DirectedEdgeHash> processedEdges;
    
    // Add initial edges from first face.
    // Note: If face is (p0, p1, p2), edges are (p0,p1), (p1,p2), (p2,p0).
    // The "open" side of (p0,p1) is (p1,p0).
    
    auto addFace = [&](int a, int b, int c) {
        hullIndices.push_back(a);
        hullIndices.push_back(b);
        hullIndices.push_back(c);
        
        Vec3 vA = m_Points[a];
        Vec3 vB = m_Points[b];
        Vec3 vC = m_Points[c];
        Vec3 normal = (vB - vA).Cross(vC - vA).Normalized();
        faceNormals.push_back(normal);
        
        // Add opposite edges to open set if not already processed
        auto handleEdge = [&](int u, int v) {
            DirectedEdge opposite(v, u);
            if (processedEdges.count(opposite)) {
                // We've already closed this edge from the other side.
                // It should currently be in openEdges? No, processedEdges tracks strictly completed adjacencies.
                // Actually, simply:
                // If (v,u) is in openEdges, remove it (we just successfully matched it).
                // Else, add (u,v) to openEdges (it needs a mate).
                // Wait, standard logic:
                // Edge (u,v) on current face. We need to find face for (v,u).
                // So adding (u,v) face implies we generated boundary edges (u,v), (v,w), (w,u).
                // The "need to solve" edges are the reversed ones: (v,u), (w,v), (u,w).
            }
        };
        
        // Simpler queue-based logic:
        // Push (p1, p0), (p2, p1), (p0, p2) to queue.
        // If an edge is already in "done" set, skip.
    };

    // Correct logic for 3D hull wrapping:
    // 1. Queue of active edges (directed).
    // 2. Set of visited edges to avoid reprocessing.
    // Initial face (p0, p1, p2).
    // Push (p1, p0), (p2, p1), (p0, p2) to queue of "open edges".
    // Mark (p0, p1), (p1, p2), (p2, p0) as visited?
    
    std::vector<DirectedEdge> workQueue;
    
    // Add first face
    hullIndices.push_back(p0);
    hullIndices.push_back(p1);
    hullIndices.push_back(p2);
    
    Vec3 n0 = (m_Points[p1] - m_Points[p0]).Cross(m_Points[p2] - m_Points[p0]).Normalized();
    faceNormals.push_back(n0);
    
    workQueue.emplace_back(p1, p0);
    workQueue.emplace_back(p2, p1);
    workQueue.emplace_back(p0, p2);
    
    processedEdges.insert(DirectedEdge(p0, p1));
    processedEdges.insert(DirectedEdge(p1, p2));
    processedEdges.insert(DirectedEdge(p2, p0));
    
    while (!workQueue.empty()) {
        DirectedEdge edge = workQueue.back();
        workQueue.pop_back();
        
        if (processedEdges.count(edge)) continue;
        
        int u = edge.v0;
        int v = edge.v1;
        
        // Find best point w for edge (u, v)
        // We need the normal of the face attached to (v, u) to measure angle from.
        // But we don't store it easily in this structure.
        // Alternative: Recompute it, or store it with the edge.
        // Recomputing is fine since we know the old face was "on the right"?
        // No, we need the "previous face" normal to fold "around".
        // Actually, we can just find the point that makes the most convex turn from ANY valid previous plane.
        // But we need to ensure we pick the correct side.
        // Simplest: Find point w such that all other points are on the "inside" of plane (u, v, w).
        // This is O(n^2) per face, brute force gift wrapping.
        // Optimization: We know (v, u) belongs to a face we already built.
        
        // Let's iterate all points and find the one that forms the "most right" turn.
        // Since we don't track the previous normal easily here, let's just use the robust check:
        // Find w such that all points x are (w-u) x (v-u) . (x-u) <= 0.
        // i.e., plane normal points OUT.
        
        int w = -1;
        for (int i = 0; i < m_PointCount; ++i) {
            if (i == u || i == v) continue;
            
            if (w == -1) {
                w = i;
                continue;
            }
            
            // Compare w and i
            // Which one is "more outside"?
            // Plane (u, v, w) normal: N_w = (v-u) x (w-u)
            // Check if point i is in front of or behind plane (u, v, w).
            // If i is in front (positive dot), then i is "more outside" than w.
            
            Vec3 edgeVec = m_Points[v] - m_Points[u];
            Vec3 toW = m_Points[w] - m_Points[u];
            Vec3 normalW = edgeVec.Cross(toW);
            
            Vec3 toI = m_Points[i] - m_Points[u];
            
            if (normalW.Dot(toI) > m_Epsilon) {
                w = i;
            }
        }
        
        if (w != -1) {
            // Found new face (u, v, w)
            hullIndices.push_back(u);
            hullIndices.push_back(v);
            hullIndices.push_back(w);
            
            Vec3 n = (m_Points[v] - m_Points[u]).Cross(m_Points[w] - m_Points[u]).Normalized();
            faceNormals.push_back(n);
            
            processedEdges.insert(edge); // Mark (u, v) as done
            
            // Add new edges (v, w) and (w, u) to queue if not processed
            workQueue.emplace_back(v, w);
            workQueue.emplace_back(w, u);
        }
    }
    
    // Construct Result
    ConvexHull result;
    result.indices = hullIndices;
    result.faceNormals = faceNormals;
    result.faceCount = static_cast<int>(hullIndices.size() / 3);
    
    // Collect unique vertices
    std::vector<int> uniqueIndices = hullIndices;
    std::sort(uniqueIndices.begin(), uniqueIndices.end());
    uniqueIndices.erase(std::unique(uniqueIndices.begin(), uniqueIndices.end()), uniqueIndices.end());
    
    // Remap indices to 0..k
    // This is optional if we just want the hull mesh as subset of original points.
    // But usually ConvexHull expects result.vertices to be the compact list.
    
    // For now, let's just copy exactly the used vertices
    std::unordered_map<int, int> oldToNew;
    for (int idx : uniqueIndices) {
        oldToNew[idx] = static_cast<int>(result.vertices.size());
        result.vertices.push_back(m_Points[idx]);
    }
    
    for (size_t i = 0; i < result.indices.size(); ++i) {
        result.indices[i] = oldToNew[result.indices[i]];
    }
    
    // Calculate area
    result.surfaceArea = 0.0f;
    for (size_t i = 0; i < result.indices.size(); i += 3) {
        const Vec3& p0 = result.vertices[result.indices[i]];
        const Vec3& p1 = result.vertices[result.indices[i+1]];
        const Vec3& p2 = result.vertices[result.indices[i+2]];
        result.surfaceArea += CalculateTriangleArea(p0, p1, p2);
    }
    
    return result;
}

int GiftWrapping::FindExtremePoint() const {
    int best = 0;
    for (int i = 1; i < m_PointCount; ++i) {
        if (m_Points[i].y < m_Points[best].y) {
            best = i;
        } else if (m_Points[i].y == m_Points[best].y) {
            if (m_Points[i].x < m_Points[best].x) best = i;
            else if (m_Points[i].x == m_Points[best].x) {
                if (m_Points[i].z < m_Points[best].z) best = i;
            }
        }
    }
    return best;
}

int GiftWrapping::FindSecondPoint(int p0) const {
    // We want the edge that has the maximum angle against the horizontal plane,
    // or just any edge on the hull. 
    // Actually, any point that makes the "steepest" edge relative to a plane passing through p0 works.
    // But simplest for convex hull:
    // Just finding the point that has the smallest angle relative to an arbitrary vector won't work in 3D easily without a plane.
    
    // Robust method: Pick a support plane at p0 (e.g. y=p0.y).
    // Find p1 that minimizes angle with that plane.
    // Or simply: Find p1 such that all other points are on one side of line p0-p1? No, 3D lines don't partition space.
    
    // Let's assume an arbitrary axis through p0 (e.g. X axis).
     // Project all points onto the plane perpendicular to X. Find hull edge there.
     // That corresponds to a 3D hull edge.
     
     // Let's use the logic:
     // "Pivot" around an axis.
     // Let's pivot around Z axis passing through p0.
     // Find p1 such that p0-p1 makes smallest angle with X axis in XY plane?
     // This guarantees p0-p1 is an edge of the SHADOW, hence an edge of the 3D hull.
     
     int best = -1;
     float maxDot = -2.0f; // Cosine of angle
     
     // Reference axis: X axis (1, 0, 0)
     Vec3 ref = Vec3(1, 0, 0);
     
     for (int i = 0; i < m_PointCount; ++i) {
         if (i == p0) continue;
         
         Vec3 dir = (m_Points[i] - m_Points[p0]).Normalized();
         float dot = dir.Dot(ref);
         
         // We essentially want the point that is "rightmost" relative to axis.
         // Actually, just finding *any* extreme vertex p1 is valid if (p0, p1) is on hull.
         // Since p0 is extreme in -Y, any vertex p1 that minimizes the angle with horizontal plane (XZ) is an edge.
         
         // Let's assume p0 is bottom-most.
         // The edge (p0, p1) with the smallest slope (closest to horizontal) is on the hull.
         // Slope = dy / horizontal_dist.
         // Minimize slope (since dy is positive).
         
         if (best == -1) {
             best = i;
             continue;
         }
         
         // Compare i with best
         // We want minimal slope dy/len
         // actually minimal angle with plane y = p0.y
         // angle = asin( (p.y - p0.y) / dist )
         // Minimize this value.
         
         Vec3 vBest = m_Points[best] - m_Points[p0];
         float angleBest = vBest.y / vBest.Length();
         
         Vec3 vI = m_Points[i] - m_Points[p0];
         float angleI = vI.y / vI.Length();
         
         if (angleI < angleBest) {
             best = i;
         }
     }
     return best;
}

int GiftWrapping::FindBestPointForEdge(int p0, int p1, const Vec3& refNormal) const {
    // This isn't used in the main loop above (we integrated the logic directly),
    // but useful for initialization step 3.
    // Find p2 such that plane (p0, p1, p2) has minimal angle with reference plane.
    
    int best = -1;
    
    for (int i = 0; i < m_PointCount; ++i) {
        if (i == p0 || i == p1) continue;
        
        if (best == -1) {
            best = i;
            continue;
        }
        
        // Compare i vs best
        // Current best normal
        Vec3 edge = m_Points[p1] - m_Points[p0];
        Vec3 nBest = edge.Cross(m_Points[best] - m_Points[p0]);
        Vec3 nI = edge.Cross(m_Points[i] - m_Points[p0]);
        
        // We want point that is "most right" (or most consistent side).
        // Check if i is in front of plane (p0, p1, best)
        if (nBest.Dot(m_Points[i] - m_Points[p0]) > m_Epsilon) {
            best = i;
        }
    }
    return best;
}

float GiftWrapping::CalculateTriangleArea(const Vec3& p0, const Vec3& p1, const Vec3& p2) const {
    return 0.5f * (p1 - p0).Cross(p2 - p0).Length();
}
