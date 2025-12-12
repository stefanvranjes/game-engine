#pragma once

// Blend curve types for animation transitions
enum class BlendCurve {
    Linear,        // Constant speed (default)
    EaseIn,        // Slow start, fast end (quadratic)
    EaseOut,       // Fast start, slow end (quadratic)
    EaseInOut,     // Slow start and end, fast middle (quadratic)
    SmoothStep,    // Smooth hermite interpolation
    SmootherStep   // Even smoother (quintic)
};

// Easing function implementations
class EasingFunctions {
public:
    // Apply easing curve to normalized time [0,1]
    static float Apply(float t, BlendCurve curve);
    
    // Individual easing functions
    static float Linear(float t);
    static float EaseIn(float t);
    static float EaseOut(float t);
    static float EaseInOut(float t);
    static float SmoothStep(float t);
    static float SmootherStep(float t);
};
