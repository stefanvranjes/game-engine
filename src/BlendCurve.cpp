#include "BlendCurve.h"
#include <algorithm>

float EasingFunctions::Apply(float t, BlendCurve curve) {
    // Clamp to [0,1]
    t = std::max(0.0f, std::min(1.0f, t));
    
    switch (curve) {
        case BlendCurve::Linear:
            return Linear(t);
        case BlendCurve::EaseIn:
            return EaseIn(t);
        case BlendCurve::EaseOut:
            return EaseOut(t);
        case BlendCurve::EaseInOut:
            return EaseInOut(t);
        case BlendCurve::SmoothStep:
            return SmoothStep(t);
        case BlendCurve::SmootherStep:
            return SmootherStep(t);
        default:
            return Linear(t);
    }
}

float EasingFunctions::Linear(float t) {
    return t;
}

float EasingFunctions::EaseIn(float t) {
    // Quadratic ease-in: t^2
    return t * t;
}

float EasingFunctions::EaseOut(float t) {
    // Quadratic ease-out: 1 - (1-t)^2
    float inv = 1.0f - t;
    return 1.0f - (inv * inv);
}

float EasingFunctions::EaseInOut(float t) {
    // Quadratic ease-in-out
    if (t < 0.5f) {
        return 2.0f * t * t;
    } else {
        float inv = 1.0f - t;
        return 1.0f - 2.0f * inv * inv;
    }
}

float EasingFunctions::SmoothStep(float t) {
    // Hermite interpolation: 3t^2 - 2t^3
    return t * t * (3.0f - 2.0f * t);
}

float EasingFunctions::SmootherStep(float t) {
    // Quintic interpolation: 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
