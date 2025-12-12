#pragma once

#include "Math/Mat4.h"
#include "Math/Vec3.h"
#include "Animation.h"
#include <string>
#include <vector>

// Bone in skeletal hierarchy
struct Bone {
    std::string name;
    int parentIndex;           // -1 for root bones
    Mat4 inverseBindMatrix;    // Transforms from mesh space to bone space
    Mat4 localTransform;       // Local transform relative to parent
    
    Bone() : parentIndex(-1), inverseBindMatrix(Mat4::Identity()), localTransform(Mat4::Identity()) {}
    Bone(const std::string& name, int parent) 
        : name(name), parentIndex(parent), inverseBindMatrix(Mat4::Identity()), localTransform(Mat4::Identity()) {}
};

// Skeleton - collection of bones
class Skeleton {
public:
    Skeleton();
    ~Skeleton();
    
    void AddBone(const Bone& bone);
    int GetBoneCount() const { return static_cast<int>(m_Bones.size()); }
    const Bone& GetBone(int index) const { return m_Bones[index]; }
    Bone& GetBone(int index) { return m_Bones[index]; }
    
    // Find bone by name
    int FindBoneIndex(const std::string& name) const;
    
    // Calculate global transforms for all bones
    void CalculateGlobalTransforms(const std::vector<Mat4>& localTransforms, std::vector<Mat4>& outGlobalTransforms) const;
    
private:
    std::vector<Bone> m_Bones;
};
