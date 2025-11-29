#pragma once

#include "Math/Vec3.h"

enum class LightType {
    Directional,
    Point,
    Spot
};

struct Light {
    LightType type;
    Vec3 position;
    Vec3 direction;
    Vec3 color;
    float intensity;
    
    // Attenuation
    float constant;
    float linear;
    float quadratic;
    
    // Spot light
    float cutOff;
    float outerCutOff;
    
    bool castsShadows;
    float range;
    float shadowSoftness;
    float lightSize; // For PCSS - affects shadow softness

    Light(Vec3 pos = Vec3(0,0,0), Vec3 col = Vec3(1,1,1), float inten = 1.0f, LightType t = LightType::Point)
        : type(t)
        , position(pos)
        , direction(0, -1, 0)
        , color(col)
        , intensity(inten)
        , constant(1.0f)
        , linear(0.09f)
        , quadratic(0.032f)
        , cutOff(12.5f)
        , outerCutOff(17.5f)
        , castsShadows(false)
        , range(20.0f)
        , shadowSoftness(1.0f)
        , lightSize(0.5f)
    {}
};
