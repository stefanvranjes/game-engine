#include "BoneMask.h"
#include "Bone.h"
#include <algorithm>
#include <iostream>

BoneMask::BoneMask() {
}

BoneMask::BoneMask(int boneCount) {
    m_Weights.resize(boneCount, 1.0f);
}

void BoneMask::SetBoneWeight(int boneIndex, float weight) {
    if (boneIndex >= 0 && boneIndex < static_cast<int>(m_Weights.size())) {
        m_Weights[boneIndex] = std::max(0.0f, std::min(1.0f, weight));
    }
}

float BoneMask::GetBoneWeight(int boneIndex) const {
    if (boneIndex >= 0 && boneIndex < static_cast<int>(m_Weights.size())) {
        return m_Weights[boneIndex];
    }
    return 0.0f;
}

void BoneMask::SetBoneWeightRecursive(int boneIndex, float weight, const Skeleton* skeleton) {
    if (!skeleton || boneIndex < 0 || boneIndex >= skeleton->GetBoneCount()) {
        return;
    }
    
    // Set weight for this bone
    SetBoneWeight(boneIndex, weight);
    
    // Recursively set weight for all children
    int boneCount = skeleton->GetBoneCount();
    for (int i = 0; i < boneCount; ++i) {
        const Bone& bone = skeleton->GetBone(i);
        if (bone.parentIndex == boneIndex) {
            SetBoneWeightRecursive(i, weight, skeleton);
        }
    }
}

BoneMask BoneMask::CreateFullBody(int boneCount) {
    BoneMask mask(boneCount);
    // All weights already initialized to 1.0
    return mask;
}

BoneMask BoneMask::CreateUpperBody(const Skeleton* skeleton, const std::string& rootBoneName) {
    if (!skeleton) {
        return BoneMask();
    }
    
    BoneMask mask(skeleton->GetBoneCount());
    mask.Clear(0.0f); // Start with all bones disabled
    
    // Find the root bone (e.g., "Spine", "Spine1", etc.)
    int rootBoneIndex = skeleton->FindBoneIndex(rootBoneName);
    if (rootBoneIndex >= 0) {
        mask.SetBoneWeightRecursive(rootBoneIndex, 1.0f, skeleton);
        std::cout << "Created upper body mask from bone: " << rootBoneName << std::endl;
    } else {
        std::cerr << "Upper body root bone not found: " << rootBoneName << std::endl;
    }
    
    return mask;
}

BoneMask BoneMask::CreateLowerBody(const Skeleton* skeleton, const std::string& rootBoneName) {
    if (!skeleton) {
        return BoneMask();
    }
    
    // Create upper body mask and invert it
    BoneMask mask = CreateUpperBody(skeleton, rootBoneName);
    mask.Invert();
    
    return mask;
}

BoneMask BoneMask::CreateFromBone(const Skeleton* skeleton, const std::string& boneName, bool includeChildren) {
    if (!skeleton) {
        return BoneMask();
    }
    
    BoneMask mask(skeleton->GetBoneCount());
    mask.Clear(0.0f);
    
    int boneIndex = skeleton->FindBoneIndex(boneName);
    if (boneIndex >= 0) {
        if (includeChildren) {
            mask.SetBoneWeightRecursive(boneIndex, 1.0f, skeleton);
        } else {
            mask.SetBoneWeight(boneIndex, 1.0f);
        }
        std::cout << "Created bone mask from: " << boneName 
                  << (includeChildren ? " (with children)" : " (single bone)") << std::endl;
    } else {
        std::cerr << "Bone not found for mask: " << boneName << std::endl;
    }
    
    return mask;
}

void BoneMask::Clear(float value) {
    float clampedValue = std::max(0.0f, std::min(1.0f, value));
    std::fill(m_Weights.begin(), m_Weights.end(), clampedValue);
}

void BoneMask::Invert() {
    for (float& weight : m_Weights) {
        weight = 1.0f - weight;
    }
}

void BoneMask::Resize(int boneCount) {
    m_Weights.resize(boneCount, 1.0f);
}
