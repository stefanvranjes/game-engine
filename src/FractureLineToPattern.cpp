#include "FractureLineToPattern.h"
#include <cmath>
#include <algorithm>

std::unique_ptr<StraightTearPattern> FractureLineToPattern::ToStraightPattern(
    const FractureLine& line)
{
    const auto& points = line.GetPoints();
    if (points.size() < 2) {
        return nullptr;
    }
    
    // Create straight pattern with fracture line width
    auto pattern = std::make_unique<StraightTearPattern>(line.GetWidth());
    
    return pattern;
}

std::unique_ptr<CurvedTearPattern> FractureLineToPattern::ToCurvedPattern(
    const FractureLine& line,
    float curvature)
{
    const auto& points = line.GetPoints();
    if (points.size() < 2) {
        return nullptr;
    }
    
    // Create curved pattern with fracture line width
    auto pattern = std::make_unique<CurvedTearPattern>(line.GetWidth(), curvature);
    
    // If we have 3 or more points, use the middle point(s) as control point
    if (points.size() >= 3) {
        // Use the middle point as explicit control point
        int midIndex = points.size() / 2;
        pattern->SetControlPoint(points[midIndex]);
    }
    
    return pattern;
}

std::vector<std::unique_ptr<SoftBodyTearPattern>> FractureLineToPattern::ToPatternSequence(
    const FractureLine& line)
{
    std::vector<std::unique_ptr<SoftBodyTearPattern>> patterns;
    const auto& points = line.GetPoints();
    
    if (points.size() < 2) {
        return patterns;
    }
    
    // Create one straight pattern for each segment
    for (size_t i = 0; i < points.size() - 1; ++i) {
        auto pattern = std::make_unique<StraightTearPattern>(line.GetWidth());
        patterns.push_back(std::move(pattern));
    }
    
    return patterns;
}

SoftBodyTearPattern::PatternType FractureLineToPattern::EstimatePatternType(
    const FractureLine& line)
{
    const auto& points = line.GetPoints();
    
    if (points.size() < 2) {
        return SoftBodyTearPattern::PatternType::Custom;
    }
    
    if (points.size() == 2) {
        // Two points always form a straight line
        return SoftBodyTearPattern::PatternType::Straight;
    }
    
    // Check if points form a straight line
    if (IsLineStraight(points)) {
        return SoftBodyTearPattern::PatternType::Straight;
    }
    
    // Otherwise, it's curved
    return SoftBodyTearPattern::PatternType::Curved;
}

float FractureLineToPattern::CalculateCurvature(const FractureLine& line)
{
    const auto& points = line.GetPoints();
    
    if (points.size() < 3) {
        return 0.0f; // Straight line
    }
    
    float deviation = CalculateDeviation(points);
    
    // Normalize deviation to 0-1 range
    // Assume max deviation of 1.0 units = max curvature
    float curvature = std::min(deviation, 1.0f);
    
    return curvature;
}

bool FractureLineToPattern::IsLineStraight(const std::vector<Vec3>& points, float threshold)
{
    if (points.size() < 3) {
        return true;
    }
    
    float deviation = CalculateDeviation(points);
    return deviation < threshold;
}

float FractureLineToPattern::CalculateDeviation(const std::vector<Vec3>& points)
{
    if (points.size() < 3) {
        return 0.0f;
    }
    
    // Calculate deviation of middle points from straight line
    Vec3 start = points.front();
    Vec3 end = points.back();
    Vec3 lineDir = end - start;
    float lineLength = lineDir.Length();
    
    if (lineLength < 0.0001f) {
        return 0.0f;
    }
    
    lineDir = lineDir * (1.0f / lineLength);
    
    float maxDeviation = 0.0f;
    
    // Check each middle point
    for (size_t i = 1; i < points.size() - 1; ++i) {
        Vec3 toPoint = points[i] - start;
        float projection = toPoint.Dot(lineDir);
        
        // Clamp to line segment
        projection = std::max(0.0f, std::min(lineLength, projection));
        
        Vec3 closestOnLine = start + lineDir * projection;
        float distance = (points[i] - closestOnLine).Length();
        
        maxDeviation = std::max(maxDeviation, distance);
    }
    
    return maxDeviation;
}
