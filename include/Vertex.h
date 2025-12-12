#pragma once

#include "Math/Vec2.h"
#include "Math/Vec3.h"

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 texCoord;
    int boneIDs[4];        // Bone indices (up to 4 bones per vertex)
    float boneWeights[4];  // Bone weights (must sum to 1.0)
    
    Vertex() : position(0, 0, 0), normal(0, 1, 0), texCoord(0, 0) {
        // Default: no bone influence
        for (int i = 0; i < 4; ++i) {
            boneIDs[i] = 0;
            boneWeights[i] = 0.0f;
        }
    }
};
