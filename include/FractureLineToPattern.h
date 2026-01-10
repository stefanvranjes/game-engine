#pragma once

#include "FractureLine.h"
#include "SoftBodyTearPattern.h"
#include "StraightTearPattern.h"
#include "CurvedTearPattern.h"
#include <memory>
#include <vector>

/**
 * @brief Utility class for converting fracture lines to tear patterns
 */
class FractureLineToPattern {
public:
    /**
     * @brief Convert fracture line to straight tear pattern
     * Uses first and last points of the fracture line
     */
    static std::unique_ptr<StraightTearPattern> ToStraightPattern(
        const FractureLine& line
    );
    
    /**
     * @brief Convert fracture line to curved tear pattern
     * Uses middle points as Bezier control points
     * @param curvature Curve intensity (0 = straight, 1 = max curve)
     */
    static std::unique_ptr<CurvedTearPattern> ToCurvedPattern(
        const FractureLine& line,
        float curvature = 0.5f
    );
    
    /**
     * @brief Convert multi-segment fracture line to pattern sequence
     * Creates one pattern per line segment
     */
    static std::vector<std::unique_ptr<SoftBodyTearPattern>> ToPatternSequence(
        const FractureLine& line
    );
    
    /**
     * @brief Estimate best pattern type for fracture line
     * Analyzes point configuration to determine optimal pattern
     */
    static SoftBodyTearPattern::PatternType EstimatePatternType(
        const FractureLine& line
    );
    
    /**
     * @brief Calculate curvature from fracture line points
     * Returns 0 for straight lines, higher values for curved
     */
    static float CalculateCurvature(const FractureLine& line);
    
private:
    /**
     * @brief Check if points form a straight line
     */
    static bool IsLineStraight(const std::vector<Vec3>& points, float threshold = 0.1f);
    
    /**
     * @brief Calculate deviation from straight line
     */
    static float CalculateDeviation(const std::vector<Vec3>& points);
};
