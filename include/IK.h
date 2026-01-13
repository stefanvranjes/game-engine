#pragma once

#include "Math/Vec3.h"
#include "Math/Mat4.h"
#include "Animation.h"
#include <vector>
#include <string>

class Skeleton;

struct IKChain {
    std::string rootBoneName;
    std::string effectorBoneName;
    
    int rootIndex = -1;
    int effectorIndex = -1;
    
    Vec3 targetPosition;
    float weight = 0.0f; // 0.0 = Animation, 1.0 = IK
    
    int iterations = 10;
    float tolerance = 0.01f;
    
    // Internal chain indices (Efector -> Parent -> ... -> Root)
    std::vector<int> chainIndices;
    // Lengths between joints
    std::vector<float> lengths;
};

class IKSolver {
public:
    static void SolveFABRIK(IKChain& chain, const Skeleton* skeleton, std::vector<Mat4>& globalTransforms);
};
