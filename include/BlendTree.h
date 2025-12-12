#pragma once

#include "Math/Vec2.h"
#include "Math/Vec3.h"
#include "Math/Mat4.h"
#include <vector>
#include <memory>

class Animation;
class Skeleton;

// 1D blend space entry
struct BlendNode1D {
    int animationIndex;
    float parameter;        // Position on 1D axis
    float currentTime;
    
    BlendNode1D() : animationIndex(-1), parameter(0.0f), currentTime(0.0f) {}
    BlendNode1D(int anim, float param) : animationIndex(anim), parameter(param), currentTime(0.0f) {}
};

// 2D blend space entry
struct BlendNode2D {
    int animationIndex;
    Vec2 parameter;        // Position in 2D space (x, y)
    float currentTime;
    
    BlendNode2D() : animationIndex(-1), parameter(0, 0), currentTime(0.0f) {}
    BlendNode2D(int anim, Vec2 param) : animationIndex(anim), parameter(param), currentTime(0.0f) {}
};

// 1D Blend Tree (linear blending)
class BlendTree1D {
public:
    BlendTree1D();
    
    void AddNode(int animationIndex, float parameter);
    void SetParameter(float value);
    void SetParameterSmooth(float value, float smoothTime);
    float GetParameter() const { return m_CurrentParameter; }
    
    // Calculate blended bone matrices
    void Update(float deltaTime, const std::vector<std::shared_ptr<Animation>>& animations,
                const Skeleton* skeleton, std::vector<Mat4>& outBoneMatrices);
    
    int GetNodeCount() const { return static_cast<int>(m_Nodes.size()); }
    
private:
    std::vector<BlendNode1D> m_Nodes;
    float m_CurrentParameter;
    float m_TargetParameter;
    float m_CurrentVelocity;
    float m_SmoothTime;
    
    bool m_NeedsSorting;
    
    void SortNodes(); // Sort by parameter value
    void FindBlendPair(int& outIndex1, int& outIndex2, float& outBlendWeight);
    void CalculateNodeBoneMatrices(int nodeIndex, const std::vector<std::shared_ptr<Animation>>& animations,
                                   const Skeleton* skeleton, std::vector<Mat4>& outMatrices);
};

// 2D Blend Tree (triangulation + barycentric blending)
class BlendTree2D {
public:
    BlendTree2D();
    
    void AddNode(int animationIndex, Vec2 parameter);
    void SetParameter(Vec2 value);
    void SetParameterSmooth(Vec2 value, float smoothTime);
    Vec2 GetParameter() const { return m_CurrentParameter; }
    
    // Calculate blended bone matrices
    void Update(float deltaTime, const std::vector<std::shared_ptr<Animation>>& animations,
                const Skeleton* skeleton, std::vector<Mat4>& outBoneMatrices);
    
    int GetNodeCount() const { return static_cast<int>(m_Nodes.size()); }
    
private:
    std::vector<BlendNode2D> m_Nodes;
    Vec2 m_CurrentParameter;
    Vec2 m_TargetParameter;
    Vec2 m_CurrentVelocity;
    float m_SmoothTime;
    
    struct Triangle {
        int indices[3];
    };
    
    std::vector<Triangle> m_Triangles;
    bool m_NeedsTriangulation;
    
    void Triangulate(); // Simple triangulation
    bool FindContainingTriangle(Vec2 point, int& outTriIndex, Vec3& outBarycentricCoords);
    Vec3 CalculateBarycentricCoordinates(Vec2 point, Vec2 a, Vec2 b, Vec2 c);
    void CalculateNodeBoneMatrices(int nodeIndex, const std::vector<std::shared_ptr<Animation>>& animations,
                                   const Skeleton* skeleton, std::vector<Mat4>& outMatrices);
    int FindNearestNode(Vec2 point);
};

// 3D blend space entry
struct BlendNode3D {
    int animationIndex;
    Vec3 parameter;        // Position in 3D space (x, y, z)
    float currentTime;
    
    BlendNode3D() : animationIndex(-1), parameter(0, 0, 0), currentTime(0.0f) {}
    BlendNode3D(int anim, Vec3 param) : animationIndex(anim), parameter(param), currentTime(0.0f) {}
};

// 3D Blend Tree (tetrahedralization + barycentric blending)
class BlendTree3D {
public:
    BlendTree3D();
    
    void AddNode(int animationIndex, Vec3 parameter);
    void SetParameter(Vec3 value);
    void SetParameterSmooth(Vec3 value, float smoothTime);
    Vec3 GetParameter() const { return m_CurrentParameter; }
    
    // Calculate blended bone matrices
    void Update(float deltaTime, const std::vector<std::shared_ptr<Animation>>& animations,
                const Skeleton* skeleton, std::vector<Mat4>& outBoneMatrices);
    
    int GetNodeCount() const { return static_cast<int>(m_Nodes.size()); }
    
private:
    std::vector<BlendNode3D> m_Nodes;
    Vec3 m_CurrentParameter;
    Vec3 m_TargetParameter;
    Vec3 m_CurrentVelocity;
    float m_SmoothTime;
    
    struct Tetrahedron {
        int indices[4];
    };
    
    std::vector<Tetrahedron> m_Tetrahedra;
    bool m_NeedsTetrahedralization;
    
    void Tetrahedralize(); // Simple tetrahedralization
    bool FindContainingTetrahedron(Vec3 point, int& outTetIndex, Vec4& outBarycentricCoords);
    Vec4 CalculateBarycentricCoordinates(Vec3 p, Vec3 a, Vec3 b, Vec3 c, Vec3 d);
    void CalculateNodeBoneMatrices(int nodeIndex, const std::vector<std::shared_ptr<Animation>>& animations,
                                   const Skeleton* skeleton, std::vector<Mat4>& outMatrices);
    int FindNearestNode(Vec3 point);
};
