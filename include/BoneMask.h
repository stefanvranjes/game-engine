#pragma once

#include <vector>
#include <string>

class Skeleton;

// Bone mask - per-bone weights for partial blending
// Allows selective animation of bone subtrees (e.g., upper body only)
class BoneMask {
public:
    BoneMask();
    explicit BoneMask(int boneCount);
    
    // Set weight for specific bone
    void SetBoneWeight(int boneIndex, float weight);
    float GetBoneWeight(int boneIndex) const;
    
    // Set weight for bone and all children recursively
    void SetBoneWeightRecursive(int boneIndex, float weight, const Skeleton* skeleton);
    
    // Preset masks for common use cases
    static BoneMask CreateFullBody(int boneCount);
    static BoneMask CreateUpperBody(const Skeleton* skeleton, const std::string& rootBoneName);
    static BoneMask CreateLowerBody(const Skeleton* skeleton, const std::string& rootBoneName);
    static BoneMask CreateFromBone(const Skeleton* skeleton, const std::string& boneName, bool includeChildren = true);
    
    // Utilities
    void Clear(float value = 0.0f);
    void Invert(); // 1.0 - weight for all bones
    void Resize(int boneCount);
    int GetBoneCount() const { return static_cast<int>(m_Weights.size()); }
    const std::vector<float>& GetWeights() const { return m_Weights; }
    
private:
    std::vector<float> m_Weights; // Per-bone weights [0,1]
};
