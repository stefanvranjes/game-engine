#include "BlendTree.h"
#include "Animation.h"
#include "Bone.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// ============================================================================
// Helper Functions
// ============================================================================

namespace {
    float SmoothDamp(float current, float target, float& currentVelocity, float smoothTime, float maxSpeed, float deltaTime) {
        smoothTime = std::max(0.0001f, smoothTime);
        float omega = 2.0f / smoothTime;
        
        float x = omega * deltaTime;
        float exp = 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
        float change = current - target;
        float originalTo = target;
        
        // Clamp maximum speed
        float maxChange = maxSpeed * smoothTime;
        change = std::max(-maxChange, std::min(maxChange, change));
        target = current - change;
        
        float temp = (currentVelocity + omega * change) * deltaTime;
        currentVelocity = (currentVelocity - omega * temp) * exp;
        float output = target + (change + temp) * exp;
        
        // Prevent overshooting
        if (originalTo - current > 0.0f == output > originalTo) {
            output = originalTo;
            currentVelocity = (output - originalTo) / deltaTime;
        }
        
        return output;
    }
    
    Vec2 SmoothDamp(Vec2 current, Vec2 target, Vec2& currentVelocity, float smoothTime, float maxSpeed, float deltaTime) {
        smoothTime = std::max(0.0001f, smoothTime);
        float omega = 2.0f / smoothTime;
        
        float x = omega * deltaTime;
        float exp = 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
        Vec2 change = current - target;
        Vec2 originalTo = target;
        
        // Clamp maximum speed
        float maxChange = maxSpeed * smoothTime;
        float changeLen = change.Length();
        if (changeLen > maxChange) {
            change = change * (maxChange / changeLen);
        }
        
        target = current - change;
        
        Vec2 temp = (currentVelocity + change * omega) * deltaTime;
        currentVelocity = (currentVelocity - temp * omega) * exp;
        Vec2 output = target + (change + temp) * exp;
        
        // Prevent overshooting check omitted for vectors for simplicity, 
        // usually strictly component-wise is fine, or just trusting the damp.
        
        return output;
    }
    
    Vec3 SmoothDamp(Vec3 current, Vec3 target, Vec3& currentVelocity, float smoothTime, float maxSpeed, float deltaTime) {
        smoothTime = std::max(0.0001f, smoothTime);
        float omega = 2.0f / smoothTime;
        
        float x = omega * deltaTime;
        float exp = 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
        Vec3 change = current - target;
        Vec3 originalTo = target;
        
        // Clamp maximum speed
        float maxChange = maxSpeed * smoothTime;
        float changeLen = change.Length();
        if (changeLen > maxChange) {
            change = change * (maxChange / changeLen);
        }
        
        target = current - change;
        
        Vec3 temp = (currentVelocity + change * omega) * deltaTime;
        currentVelocity = (currentVelocity - temp * omega) * exp;
        Vec3 output = target + (change + temp) * exp;
        
        return output;
    }
}

// ============================================================================
// BlendTree1D Implementation
// ============================================================================

BlendTree1D::BlendTree1D() 
    : m_CurrentParameter(0.0f)
    , m_TargetParameter(0.0f)
    , m_CurrentVelocity(0.0f)
    , m_SmoothTime(0.0f)
    , m_NeedsSorting(false) {
}

void BlendTree1D::AddNode(int animationIndex, float parameter) {
    m_Nodes.push_back(BlendNode1D(animationIndex, parameter));
    m_NeedsSorting = true;
}

void BlendTree1D::SetParameter(float value) {
    m_TargetParameter = value;
    m_SmoothTime = 0.0f;
    m_CurrentParameter = value;
    m_CurrentVelocity = 0.0f;
}

void BlendTree1D::SetParameterSmooth(float value, float smoothTime) {
    m_TargetParameter = value;
    m_SmoothTime = smoothTime;
}

void BlendTree1D::SortNodes() {
    if (!m_NeedsSorting) return;
    
    std::sort(m_Nodes.begin(), m_Nodes.end(), 
              [](const BlendNode1D& a, const BlendNode1D& b) {
                  return a.parameter < b.parameter;
              });
    
    m_NeedsSorting = false;
}

void BlendTree1D::FindBlendPair(int& outIndex1, int& outIndex2, float& outBlendWeight) {
    if (m_Nodes.empty()) {
        outIndex1 = outIndex2 = -1;
        outBlendWeight = 0.0f;
        return;
    }
    
    if (m_Nodes.size() == 1) {
        outIndex1 = outIndex2 = 0;
        outBlendWeight = 0.0f;
        return;
    }
    
    // Find nodes that bracket current parameter
    for (size_t i = 0; i < m_Nodes.size() - 1; ++i) {
        if (m_CurrentParameter >= m_Nodes[i].parameter && 
            m_CurrentParameter <= m_Nodes[i+1].parameter) {
            outIndex1 = static_cast<int>(i);
            outIndex2 = static_cast<int>(i + 1);
            
            float range = m_Nodes[i+1].parameter - m_Nodes[i].parameter;
            if (range > 0.0001f) {
                outBlendWeight = (m_CurrentParameter - m_Nodes[i].parameter) / range;
            } else {
                outBlendWeight = 0.0f;
            }
            return;
        }
    }
    
    // Clamp to edges
    if (m_CurrentParameter <= m_Nodes[0].parameter) {
        outIndex1 = outIndex2 = 0;
        outBlendWeight = 0.0f;
    } else {
        outIndex1 = outIndex2 = static_cast<int>(m_Nodes.size() - 1);
        outBlendWeight = 0.0f;
    }
}

void BlendTree1D::CalculateNodeBoneMatrices(int nodeIndex, 
                                            const std::vector<std::shared_ptr<Animation>>& animations,
                                            const Skeleton* skeleton, 
                                            std::vector<Mat4>& outMatrices) {
    if (!skeleton || nodeIndex < 0 || nodeIndex >= static_cast<int>(m_Nodes.size())) {
        return;
    }
    
    BlendNode1D& node = m_Nodes[nodeIndex];
    if (node.animationIndex < 0 || node.animationIndex >= static_cast<int>(animations.size())) {
        return;
    }
    
    Animation* anim = animations[node.animationIndex].get();
    int boneCount = skeleton->GetBoneCount();
    
    std::vector<Mat4> localTransforms(boneCount);
    std::vector<Mat4> globalTransforms(boneCount);
    
    // Update local transforms from animation
    for (int i = 0; i < boneCount; ++i) {
        const AnimationChannel* channel = anim->GetChannelForBone(i);
        
        if (channel) {
            Vec3 position;
            Quaternion rotation;
            Vec3 scale;
            channel->GetTransform(node.currentTime, position, rotation, scale);
            
            Mat4 translation = Mat4::Translate(position);
            Mat4 rotationMat = rotation.ToMatrix();
            Mat4 scaleMat = Mat4::Scale(scale);
            
            localTransforms[i] = translation * rotationMat * scaleMat;
        } else {
            localTransforms[i] = skeleton->GetBone(i).localTransform;
        }
    }
    
    // Calculate global transforms
    skeleton->CalculateGlobalTransforms(localTransforms, globalTransforms);
    
    // Calculate final bone matrices
    outMatrices.resize(boneCount);
    for (int i = 0; i < boneCount; ++i) {
        outMatrices[i] = globalTransforms[i] * skeleton->GetBone(i).inverseBindMatrix;
    }
}

void BlendTree1D::Update(float deltaTime, 
                         const std::vector<std::shared_ptr<Animation>>& animations,
                         const Skeleton* skeleton, 
                         std::vector<Mat4>& outBoneMatrices) {
    if (!skeleton || m_Nodes.empty()) {
        return;
    }
    
    // Sort nodes if needed
    SortNodes();
    
    // Apply smoothing
    if (m_SmoothTime > 0.0001f) {
        m_CurrentParameter = SmoothDamp(m_CurrentParameter, m_TargetParameter, m_CurrentVelocity, m_SmoothTime, 1000.0f, deltaTime);
    } else {
        m_CurrentParameter = m_TargetParameter;
    }
    
    // Find blend pair first to determine dominant node for sync
    int index1, index2;
    float blendWeight;
    FindBlendPair(index1, index2, blendWeight);
    
    // Determine dominant node
    int dominantIndex = -1;
    if (index1 >= 0) {
        dominantIndex = (blendWeight < 0.5f) ? index1 : index2;
    }
    
    std::string dominantSyncGroup = "";
    float dominantNormalizedTime = 0.0f;
    
    // Update dominant node time first
    if (dominantIndex >= 0 && dominantIndex < static_cast<int>(m_Nodes.size())) {
        BlendNode1D& domNode = m_Nodes[dominantIndex];
        if (domNode.animationIndex >= 0 && domNode.animationIndex < static_cast<int>(animations.size())) {
            Animation* domAnim = animations[domNode.animationIndex].get();
            dominantSyncGroup = domAnim->GetSyncGroup();
            
            domNode.currentTime += deltaTime;
            float duration = domAnim->GetDuration();
            if (duration > 0.0f) {
                domNode.currentTime = fmod(domNode.currentTime, duration);
                dominantNormalizedTime = domNode.currentTime / duration;
            }
        }
    }
    
    // Update other nodes
    for (int i = 0; i < static_cast<int>(m_Nodes.size()); ++i) {
        if (i == dominantIndex) continue;
        
        BlendNode1D& node = m_Nodes[i];
        if (node.animationIndex >= 0 && node.animationIndex < static_cast<int>(animations.size())) {
            Animation* anim = animations[node.animationIndex].get();
            float duration = anim->GetDuration();
            
            // Sync if in same group, otherwise update independently
            if (!dominantSyncGroup.empty() && anim->GetSyncGroup() == dominantSyncGroup && duration > 0.0f) {
                 node.currentTime = dominantNormalizedTime * duration;
            } else {
                 node.currentTime += deltaTime;
                 if (duration > 0.0f) {
                     node.currentTime = fmod(node.currentTime, duration);
                 }
            }
        }
    }
    
    if (index1 < 0) {
        return;
    }
    
    int boneCount = skeleton->GetBoneCount();
    
    if (index1 == index2 || blendWeight <= 0.0001f) {
        // No blending needed, use single animation
        CalculateNodeBoneMatrices(index1, animations, skeleton, outBoneMatrices);
    } else {
        // Blend between two animations
        std::vector<Mat4> matrices1, matrices2;
        CalculateNodeBoneMatrices(index1, animations, skeleton, matrices1);
        CalculateNodeBoneMatrices(index2, animations, skeleton, matrices2);
        
        // Blend matrices
        outBoneMatrices.resize(boneCount);
        for (int i = 0; i < boneCount; ++i) {
            // Decompose matrices
            Vec3 pos1, pos2, scale1, scale2;
            Quaternion rot1, rot2;
            
            DecomposeMatrix(matrices1[i], pos1, rot1, scale1);
            DecomposeMatrix(matrices2[i], pos2, rot2, scale2);
            
            // Interpolate
            Vec3 blendPos;
            blendPos.x = pos1.x + blendWeight * (pos2.x - pos1.x);
            blendPos.y = pos1.y + blendWeight * (pos2.y - pos1.y);
            blendPos.z = pos1.z + blendWeight * (pos2.z - pos1.z);
            
            Quaternion blendRot = Quaternion::Slerp(rot1, rot2, blendWeight);
            
            Vec3 blendScale;
            blendScale.x = scale1.x + blendWeight * (scale2.x - scale1.x);
            blendScale.y = scale1.y + blendWeight * (scale2.y - scale1.y);
            blendScale.z = scale1.z + blendWeight * (scale2.z - scale1.z);
            
            outBoneMatrices[i] = ComposeMatrix(blendPos, blendRot, blendScale);
        }
    }
}

// ============================================================================
// BlendTree2D Implementation
// ============================================================================

BlendTree2D::BlendTree2D() 
    : m_CurrentParameter(0, 0)
    , m_TargetParameter(0, 0)
    , m_CurrentVelocity(0, 0)
    , m_SmoothTime(0.0f)
    , m_NeedsTriangulation(false) {
}

void BlendTree2D::AddNode(int animationIndex, Vec2 parameter) {
    m_Nodes.push_back(BlendNode2D(animationIndex, parameter));
    m_NeedsTriangulation = true;
}

void BlendTree2D::SetParameter(Vec2 value) {
    m_TargetParameter = value;
    m_SmoothTime = 0.0f;
    m_CurrentParameter = value;
    m_CurrentVelocity = Vec2(0, 0);
}

void BlendTree2D::SetParameterSmooth(Vec2 value, float smoothTime) {
    m_TargetParameter = value;
    m_SmoothTime = smoothTime;
}

void BlendTree2D::Triangulate() {
    if (!m_NeedsTriangulation || m_Nodes.size() < 3) {
        return;
    }
    
    m_Triangles.clear();
    
    // Simple fan triangulation from first node
    // For production, use Delaunay triangulation
    for (size_t i = 1; i < m_Nodes.size() - 1; ++i) {
        Triangle tri;
        tri.indices[0] = 0;
        tri.indices[1] = static_cast<int>(i);
        tri.indices[2] = static_cast<int>(i + 1);
        m_Triangles.push_back(tri);
    }
    
    m_NeedsTriangulation = false;
}

Vec3 BlendTree2D::CalculateBarycentricCoordinates(Vec2 p, Vec2 a, Vec2 b, Vec2 c) {
    Vec2 v0 = b - a;
    Vec2 v1 = c - a;
    Vec2 v2 = p - a;
    
    float d00 = v0.x * v0.x + v0.y * v0.y;
    float d01 = v0.x * v1.x + v0.y * v1.y;
    float d11 = v1.x * v1.x + v1.y * v1.y;
    float d20 = v2.x * v0.x + v2.y * v0.y;
    float d21 = v2.x * v1.x + v2.y * v1.y;
    
    float denom = d00 * d11 - d01 * d01;
    if (fabs(denom) < 0.0001f) {
        return Vec3(1, 0, 0);
    }
    
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;
    
    return Vec3(u, v, w);
}

bool BlendTree2D::FindContainingTriangle(Vec2 point, int& outTriIndex, Vec3& outBarycentricCoords) {
    for (size_t i = 0; i < m_Triangles.size(); ++i) {
        const Triangle& tri = m_Triangles[i];
        
        Vec2 a = m_Nodes[tri.indices[0]].parameter;
        Vec2 b = m_Nodes[tri.indices[1]].parameter;
        Vec2 c = m_Nodes[tri.indices[2]].parameter;
        
        Vec3 bary = CalculateBarycentricCoordinates(point, a, b, c);
        
        // Check if point is inside triangle
        if (bary.x >= -0.001f && bary.y >= -0.001f && bary.z >= -0.001f) {
            outTriIndex = static_cast<int>(i);
            outBarycentricCoords = bary;
            return true;
        }
    }
    
    return false;
}

int BlendTree2D::FindNearestNode(Vec2 point) {
    if (m_Nodes.empty()) return -1;
    
    int nearestIndex = 0;
    float nearestDist = (m_Nodes[0].parameter - point).Length();
    
    for (size_t i = 1; i < m_Nodes.size(); ++i) {
        float dist = (m_Nodes[i].parameter - point).Length();
        if (dist < nearestDist) {
            nearestDist = dist;
            nearestIndex = static_cast<int>(i);
        }
    }
    
    return nearestIndex;
}

void BlendTree2D::CalculateNodeBoneMatrices(int nodeIndex,
                                            const std::vector<std::shared_ptr<Animation>>& animations,
                                            const Skeleton* skeleton,
                                            std::vector<Mat4>& outMatrices) {
    if (!skeleton || nodeIndex < 0 || nodeIndex >= static_cast<int>(m_Nodes.size())) {
        return;
    }
    
    BlendNode2D& node = m_Nodes[nodeIndex];
    if (node.animationIndex < 0 || node.animationIndex >= static_cast<int>(animations.size())) {
        return;
    }
    
    Animation* anim = animations[node.animationIndex].get();
    int boneCount = skeleton->GetBoneCount();
    
    std::vector<Mat4> localTransforms(boneCount);
    std::vector<Mat4> globalTransforms(boneCount);
    
    // Update local transforms from animation
    for (int i = 0; i < boneCount; ++i) {
        const AnimationChannel* channel = anim->GetChannelForBone(i);
        
        if (channel) {
            Vec3 position;
            Quaternion rotation;
            Vec3 scale;
            channel->GetTransform(node.currentTime, position, rotation, scale);
            
            Mat4 translation = Mat4::Translate(position);
            Mat4 rotationMat = rotation.ToMatrix();
            Mat4 scaleMat = Mat4::Scale(scale);
            
            localTransforms[i] = translation * rotationMat * scaleMat;
        } else {
            localTransforms[i] = skeleton->GetBone(i).localTransform;
        }
    }
    
    // Calculate global transforms
    skeleton->CalculateGlobalTransforms(localTransforms, globalTransforms);
    
    // Calculate final bone matrices
    outMatrices.resize(boneCount);
    for (int i = 0; i < boneCount; ++i) {
        outMatrices[i] = globalTransforms[i] * skeleton->GetBone(i).inverseBindMatrix;
    }
}

void BlendTree2D::Update(float deltaTime,
                         const std::vector<std::shared_ptr<Animation>>& animations,
                         const Skeleton* skeleton,
                         std::vector<Mat4>& outBoneMatrices) {
    if (!skeleton || m_Nodes.empty()) {
        return;
    }
    
    // Triangulate if needed
    if (m_NeedsTriangulation) {
        Triangulate();
    }
    
    // Apply smoothing
    if (m_SmoothTime > 0.0001f) {
        m_CurrentParameter = SmoothDamp(m_CurrentParameter, m_TargetParameter, m_CurrentVelocity, m_SmoothTime, 1000.0f, deltaTime);
    } else {
        m_CurrentParameter = m_TargetParameter;
    }
    

    int boneCount = skeleton->GetBoneCount();
    
    // Find containing triangle first determines weights and active nodes
    int triIndex = -1;
    Vec3 baryCoords;
    bool hasTriangle = !m_Triangles.empty() && FindContainingTriangle(m_CurrentParameter, triIndex, baryCoords);
    
    int dominantIndex = -1;
    
    if (hasTriangle) {
        // Find dominant node in triangle (max barycentric weight)
        const Triangle& tri = m_Triangles[triIndex];
        if (baryCoords.x >= baryCoords.y && baryCoords.x >= baryCoords.z) dominantIndex = tri.indices[0];
        else if (baryCoords.y >= baryCoords.z) dominantIndex = tri.indices[1];
        else dominantIndex = tri.indices[2];
    } else {
        // Use nearest node
        dominantIndex = FindNearestNode(m_CurrentParameter);
    }
    
    // Update times with sync logic
    std::string dominantSyncGroup = "";
    float dominantNormalizedTime = 0.0f;
    
    // Update dominant node
    if (dominantIndex >= 0 && dominantIndex < static_cast<int>(m_Nodes.size())) {
        BlendNode2D& domNode = m_Nodes[dominantIndex];
        if (domNode.animationIndex >= 0 && domNode.animationIndex < static_cast<int>(animations.size())) {
            Animation* domAnim = animations[domNode.animationIndex].get();
            dominantSyncGroup = domAnim->GetSyncGroup();
            
            domNode.currentTime += deltaTime;
            float duration = domAnim->GetDuration();
            if (duration > 0.0f) {
                domNode.currentTime = fmod(domNode.currentTime, duration);
                dominantNormalizedTime = domNode.currentTime / duration;
            }
        }
    }
    
    // Update others
    for (int i = 0; i < static_cast<int>(m_Nodes.size()); ++i) {
        if (i == dominantIndex) continue;
        
        BlendNode2D& node = m_Nodes[i];
         if (node.animationIndex >= 0 && node.animationIndex < static_cast<int>(animations.size())) {
            Animation* anim = animations[node.animationIndex].get();
            float duration = anim->GetDuration();
            
            if (!dominantSyncGroup.empty() && anim->GetSyncGroup() == dominantSyncGroup && duration > 0.0f) {
                 node.currentTime = dominantNormalizedTime * duration;
            } else {
                 node.currentTime += deltaTime;
                 if (duration > 0.0f) {
                     node.currentTime = fmod(node.currentTime, duration);
                 }
            }
        }
    }
    
    if (!hasTriangle) {
        // Use nearest node if outside or no triangles
        if (dominantIndex >= 0) {
            CalculateNodeBoneMatrices(dominantIndex, animations, skeleton, outBoneMatrices);
        }
        return;
    }
    
    // Blend three animations using barycentric coordinates
    const Triangle& tri = m_Triangles[triIndex];
    std::vector<Mat4> matrices[3];
    
    for (int i = 0; i < 3; ++i) {
        CalculateNodeBoneMatrices(tri.indices[i], animations, skeleton, matrices[i]);
    }
    
    // Blend using barycentric coordinates
    outBoneMatrices.resize(boneCount);
    for (int bone = 0; bone < boneCount; ++bone) {
        // Decompose all three matrices
        Vec3 pos[3], scale[3];
        Quaternion rot[3];
        
        for (int i = 0; i < 3; ++i) {
            DecomposeMatrix(matrices[i][bone], pos[i], rot[i], scale[i]);
        }
        
        // Blend position
        Vec3 blendPos;
        blendPos.x = pos[0].x * baryCoords.x + pos[1].x * baryCoords.y + pos[2].x * baryCoords.z;
        blendPos.y = pos[0].y * baryCoords.x + pos[1].y * baryCoords.y + pos[2].y * baryCoords.z;
        blendPos.z = pos[0].z * baryCoords.x + pos[1].z * baryCoords.y + pos[2].z * baryCoords.z;
        
        // Blend rotation (sequential slerp)
        Quaternion blendRot = Quaternion::Slerp(rot[0], rot[1], baryCoords.y / (baryCoords.x + baryCoords.y + 0.0001f));
        blendRot = Quaternion::Slerp(blendRot, rot[2], baryCoords.z);
        
        // Blend scale
        Vec3 blendScale;
        blendScale.x = scale[0].x * baryCoords.x + scale[1].x * baryCoords.y + scale[2].x * baryCoords.z;
        blendScale.y = scale[0].y * baryCoords.x + scale[1].y * baryCoords.y + scale[2].y * baryCoords.z;
        blendScale.z = scale[0].z * baryCoords.x + scale[1].z * baryCoords.y + scale[2].z * baryCoords.z;
        
        outBoneMatrices[bone] = ComposeMatrix(blendPos, blendRot, blendScale);
    }
}

// ============================================================================
// BlendTree3D Implementation
// ============================================================================

BlendTree3D::BlendTree3D() 
    : m_CurrentParameter(0, 0, 0)
    , m_TargetParameter(0, 0, 0)
    , m_CurrentVelocity(0, 0, 0)
    , m_SmoothTime(0.0f)
    , m_NeedsTetrahedralization(false) {
}

void BlendTree3D::AddNode(int animationIndex, Vec3 parameter) {
    m_Nodes.push_back(BlendNode3D(animationIndex, parameter));
    m_NeedsTetrahedralization = true;
}

void BlendTree3D::SetParameter(Vec3 value) {
    m_TargetParameter = value;
    m_SmoothTime = 0.0f;
    m_CurrentParameter = value;
    m_CurrentVelocity = Vec3(0, 0, 0);
}

void BlendTree3D::SetParameterSmooth(Vec3 value, float smoothTime) {
    m_TargetParameter = value;
    m_SmoothTime = smoothTime;
}

void BlendTree3D::Tetrahedralize() {
    if (!m_NeedsTetrahedralization || m_Nodes.size() < 4) {
        return;
    }
    
    m_Tetrahedra.clear();
    
    // Simple fan tetrahedralization from first node and second node
    // For production, use Delaunay tetrahedralization
    for (size_t i = 2; i < m_Nodes.size() - 1; ++i) {
        Tetrahedron tet;
        tet.indices[0] = 0;
        tet.indices[1] = 1;
        tet.indices[2] = static_cast<int>(i);
        tet.indices[3] = static_cast<int>(i + 1);
        m_Tetrahedra.push_back(tet);
    }
    
    m_NeedsTetrahedralization = false;
}

Vec4 BlendTree3D::CalculateBarycentricCoordinates(Vec3 p, Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    Vec3 vap = p - a;
    
    Vec3 vab = b - a;
    Vec3 vac = c - a;
    Vec3 vad = d - a;
    
    // Scalar triple product for volume calculation
    // Volume of tetrahedron = 1/6 * |(b-a) . ((c-a) x (d-a))|
    float va6 = vab.Dot(vac.Cross(vad));
    
    if (fabs(va6) < 0.0001f) {
        // Degenerate tetrahedron
        return Vec4(1, 0, 0, 0);
    }
    
    float invVol = 1.0f / va6;
    
    // Weight for D (VolABCP)
    float wD = vap.Dot(vab.Cross(vac)) * invVol;
    
    // Weight for C (VolABPD)
    float wC = vap.Dot(vad.Cross(vab)) * invVol;
    
    // Weight for B (VolAPCD)
    // Note: The order of cross product matters for signed volume
    float wB = vap.Dot(vac.Cross(vad)) * invVol;
    
    // Weight for A = 1 - wB - wC - wD
    float wA = 1.0f - wB - wC - wD;
    
    return Vec4(wA, wB, wC, wD);
}

bool BlendTree3D::FindContainingTetrahedron(Vec3 point, int& outTetIndex, Vec4& outBarycentricCoords) {
    for (size_t i = 0; i < m_Tetrahedra.size(); ++i) {
        const Tetrahedron& tet = m_Tetrahedra[i];
        
        Vec3 a = m_Nodes[tet.indices[0]].parameter;
        Vec3 b = m_Nodes[tet.indices[1]].parameter;
        Vec3 c = m_Nodes[tet.indices[2]].parameter;
        Vec3 d = m_Nodes[tet.indices[3]].parameter;
        
        Vec4 bary = CalculateBarycentricCoordinates(point, a, b, c, d);
        
        // Check if point is inside tetrahedron (all weights >= 0)
        // Add small epsilon for floating point errors
        if (bary.x >= -0.001f && bary.y >= -0.001f && bary.z >= -0.001f && bary.w >= -0.001f) {
            outTetIndex = static_cast<int>(i);
            outBarycentricCoords = bary;
            return true;
        }
    }
    
    return false;
}

int BlendTree3D::FindNearestNode(Vec3 point) {
    if (m_Nodes.empty()) return -1;
    
    int nearestIndex = 0;
    float nearestDist = (m_Nodes[0].parameter - point).Length();
    
    for (size_t i = 1; i < m_Nodes.size(); ++i) {
        float dist = (m_Nodes[i].parameter - point).Length();
        if (dist < nearestDist) {
            nearestDist = dist;
            nearestIndex = static_cast<int>(i);
        }
    }
    
    return nearestIndex;
}

void BlendTree3D::CalculateNodeBoneMatrices(int nodeIndex,
                                            const std::vector<std::shared_ptr<Animation>>& animations,
                                            const Skeleton* skeleton,
                                            std::vector<Mat4>& outMatrices) {
    if (!skeleton || nodeIndex < 0 || nodeIndex >= static_cast<int>(m_Nodes.size())) {
        return;
    }
    
    BlendNode3D& node = m_Nodes[nodeIndex];
    if (node.animationIndex < 0 || node.animationIndex >= static_cast<int>(animations.size())) {
        return;
    }
    
    Animation* anim = animations[node.animationIndex].get();
    int boneCount = skeleton->GetBoneCount();
    
    std::vector<Mat4> localTransforms(boneCount);
    std::vector<Mat4> globalTransforms(boneCount);
    
    // Update local transforms from animation
    for (int i = 0; i < boneCount; ++i) {
        const AnimationChannel* channel = anim->GetChannelForBone(i);
        
        if (channel) {
            Vec3 position;
            Quaternion rotation;
            Vec3 scale;
            channel->GetTransform(node.currentTime, position, rotation, scale);
            
            Mat4 translation = Mat4::Translate(position);
            Mat4 rotationMat = rotation.ToMatrix();
            Mat4 scaleMat = Mat4::Scale(scale);
            
            localTransforms[i] = translation * rotationMat * scaleMat;
        } else {
            localTransforms[i] = skeleton->GetBone(i).localTransform;
        }
    }
    
    // Calculate global transforms
    skeleton->CalculateGlobalTransforms(localTransforms, globalTransforms);
    
    // Calculate final bone matrices
    outMatrices.resize(boneCount);
    for (int i = 0; i < boneCount; ++i) {
        outMatrices[i] = globalTransforms[i] * skeleton->GetBone(i).inverseBindMatrix;
    }
}

void BlendTree3D::Update(float deltaTime,
                         const std::vector<std::shared_ptr<Animation>>& animations,
                         const Skeleton* skeleton,
                         std::vector<Mat4>& outBoneMatrices) {
    if (!skeleton || m_Nodes.empty()) {
        return;
    }
    
    // Tetrahedralize if needed
    if (m_NeedsTetrahedralization) {
        Tetrahedralize();
    }
    
    // Apply smoothing
    if (m_SmoothTime > 0.0001f) {
        m_CurrentParameter = SmoothDamp(m_CurrentParameter, m_TargetParameter, m_CurrentVelocity, m_SmoothTime, 1000.0f, deltaTime);
    } else {
        m_CurrentParameter = m_TargetParameter;
    }
    
    int boneCount = skeleton->GetBoneCount();
    
    // Find containing tetrahedron first
    int tetIndex = -1;
    Vec4 baryCoords;
    bool hasTetra = !m_Tetrahedra.empty() && FindContainingTetrahedron(m_CurrentParameter, tetIndex, baryCoords);
    
    int dominantIndex = -1;
    if (hasTetra) {
        // Find dominant node
        const Tetrahedron& tet = m_Tetrahedra[tetIndex];
        float maxW = baryCoords.x;
        dominantIndex = tet.indices[0];
        
        if (baryCoords.y > maxW) { maxW = baryCoords.y; dominantIndex = tet.indices[1]; }
        if (baryCoords.z > maxW) { maxW = baryCoords.z; dominantIndex = tet.indices[2]; }
        if (baryCoords.w > maxW) { maxW = baryCoords.w; dominantIndex = tet.indices[3]; }
    } else {
        dominantIndex = FindNearestNode(m_CurrentParameter);
    }
    
    // Update times with sync logic
    std::string dominantSyncGroup = "";
    float dominantNormalizedTime = 0.0f;
    
     // Update dominant node
    if (dominantIndex >= 0 && dominantIndex < static_cast<int>(m_Nodes.size())) {
        BlendNode3D& domNode = m_Nodes[dominantIndex];
        if (domNode.animationIndex >= 0 && domNode.animationIndex < static_cast<int>(animations.size())) {
            Animation* domAnim = animations[domNode.animationIndex].get();
            dominantSyncGroup = domAnim->GetSyncGroup();
            
            domNode.currentTime += deltaTime;
            float duration = domAnim->GetDuration();
            if (duration > 0.0f) {
                domNode.currentTime = fmod(domNode.currentTime, duration);
                dominantNormalizedTime = domNode.currentTime / duration;
            }
        }
    }
      
    // Update others
    for (int i = 0; i < static_cast<int>(m_Nodes.size()); ++i) {
        if (i == dominantIndex) continue;
        
        BlendNode3D& node = m_Nodes[i];
         if (node.animationIndex >= 0 && node.animationIndex < static_cast<int>(animations.size())) {
            Animation* anim = animations[node.animationIndex].get();
            float duration = anim->GetDuration();
            
            if (!dominantSyncGroup.empty() && anim->GetSyncGroup() == dominantSyncGroup && duration > 0.0f) {
                 node.currentTime = dominantNormalizedTime * duration;
            } else {
                 node.currentTime += deltaTime;
                 if (duration > 0.0f) {
                     node.currentTime = fmod(node.currentTime, duration);
                 }
            }
        }
    }

    if (!hasTetra) {
        // Use nearest node if outside or no tetrahedra
        if (dominantIndex >= 0) {
            CalculateNodeBoneMatrices(dominantIndex, animations, skeleton, outBoneMatrices);
        }
        return;
    }
    
    // Blend four animations using barycentric coordinates
    const Tetrahedron& tet = m_Tetrahedra[tetIndex];
    std::vector<Mat4> matrices[4];
    
    for (int i = 0; i < 4; ++i) {
        CalculateNodeBoneMatrices(tet.indices[i], animations, skeleton, matrices[i]);
    }
    
    // Blend using barycentric coordinates
    outBoneMatrices.resize(boneCount);
    for (int bone = 0; bone < boneCount; ++bone) {
        // Decompose all four matrices
        Vec3 pos[4], scale[4];
        Quaternion rot[4];
        
        for (int i = 0; i < 4; ++i) {
            DecomposeMatrix(matrices[i][bone], pos[i], rot[i], scale[i]);
        }
        
        // Blend position
        Vec3 blendPos;
        blendPos.x = pos[0].x * baryCoords.x + pos[1].x * baryCoords.y + pos[2].x * baryCoords.z + pos[3].x * baryCoords.w;
        blendPos.y = pos[0].y * baryCoords.x + pos[1].y * baryCoords.y + pos[2].y * baryCoords.z + pos[3].y * baryCoords.w;
        blendPos.z = pos[0].z * baryCoords.x + pos[1].z * baryCoords.y + pos[2].z * baryCoords.z + pos[3].z * baryCoords.w;
        
        // Blend rotation (sequential slerp)
        Quaternion blendRot = Quaternion::Slerp(rot[0], rot[1], baryCoords.y / (baryCoords.x + baryCoords.y + 0.0001f));
        float combinedWeight = baryCoords.x + baryCoords.y;
        blendRot = Quaternion::Slerp(blendRot, rot[2], baryCoords.z / (combinedWeight + baryCoords.z + 0.0001f));
        blendRot = Quaternion::Slerp(blendRot, rot[3], baryCoords.w);
        
        // Blend scale
        Vec3 blendScale;
        blendScale.x = scale[0].x * baryCoords.x + scale[1].x * baryCoords.y + scale[2].x * baryCoords.z + scale[3].x * baryCoords.w;
        blendScale.y = scale[0].y * baryCoords.x + scale[1].y * baryCoords.y + scale[2].y * baryCoords.z + scale[3].y * baryCoords.w;
        blendScale.z = scale[0].z * baryCoords.x + scale[1].z * baryCoords.y + scale[2].z * baryCoords.z + scale[3].z * baryCoords.w;
        
        outBoneMatrices[bone] = ComposeMatrix(blendPos, blendRot, blendScale);
    }
}
