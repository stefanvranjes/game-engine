#include "Bone.h"
#include <algorithm>

// Skeleton implementation
Skeleton::Skeleton() {
}

Skeleton::~Skeleton() {
}

void Skeleton::AddBone(const Bone& bone) {
    m_Bones.push_back(bone);
}

int Skeleton::FindBoneIndex(const std::string& name) const {
    for (size_t i = 0; i < m_Bones.size(); ++i) {
        if (m_Bones[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void Skeleton::CalculateGlobalTransforms(const std::vector<Mat4>& localTransforms, std::vector<Mat4>& outGlobalTransforms) const {
    outGlobalTransforms.resize(m_Bones.size());
    
    for (size_t i = 0; i < m_Bones.size(); ++i) {
        const Bone& bone = m_Bones[i];
        
        if (bone.parentIndex < 0) {
            // Root bone - global transform is just local transform
            outGlobalTransforms[i] = localTransforms[i];
        } else {
            // Child bone - multiply parent's global transform by local transform
            outGlobalTransforms[i] = outGlobalTransforms[bone.parentIndex] * localTransforms[i];
        }
    }
}
