#pragma once

namespace Easing {
    // Easing function types
    enum class EaseType {
        Linear,
        EaseInQuad,
        EaseOutQuad,
        EaseInOutQuad,
        EaseInCubic,
        EaseOutCubic,
        EaseInOutCubic
    };

    // Linear interpolation - no easing, constant speed
    inline float Linear(float t) {
        return t;
    }

    // Quadratic easing functions
    inline float EaseInQuad(float t) {
        return t * t;
    }

    inline float EaseOutQuad(float t) {
        return t * (2.0f - t);
    }

    inline float EaseInOutQuad(float t) {
        if (t < 0.5f) {
            return 2.0f * t * t;
        } else {
            return -1.0f + (4.0f - 2.0f * t) * t;
        }
    }

    // Cubic easing functions
    inline float EaseInCubic(float t) {
        return t * t * t;
    }

    inline float EaseOutCubic(float t) {
        float f = t - 1.0f;
        return f * f * f + 1.0f;
    }

    inline float EaseInOutCubic(float t) {
        if (t < 0.5f) {
            return 4.0f * t * t * t;
        } else {
            float f = (2.0f * t) - 2.0f;
            return 0.5f * f * f * f + 1.0f;
        }
    }

    // Main easing function - dispatches to appropriate curve
    inline float Ease(float t, EaseType type) {
        // Clamp t to [0, 1] range
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;

        switch (type) {
            case EaseType::Linear:
                return Linear(t);
            case EaseType::EaseInQuad:
                return EaseInQuad(t);
            case EaseType::EaseOutQuad:
                return EaseOutQuad(t);
            case EaseType::EaseInOutQuad:
                return EaseInOutQuad(t);
            case EaseType::EaseInCubic:
                return EaseInCubic(t);
            case EaseType::EaseOutCubic:
                return EaseOutCubic(t);
            case EaseType::EaseInOutCubic:
                return EaseInOutCubic(t);
            default:
                return Linear(t);
        }
    }
}
