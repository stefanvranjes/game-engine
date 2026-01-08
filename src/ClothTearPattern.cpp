#include "ClothTearPattern.h"
#include <sstream>
#include <cmath>
#include <algorithm>

// ============================================================================
// Base ClothTearPattern
// ============================================================================

std::map<std::string, std::string> ClothTearPattern::Serialize() const {
    std::map<std::string, std::string> data;
    data["name"] = m_Name;
    data["description"] = m_Description;
    return data;
}

void ClothTearPattern::Deserialize(const std::map<std::string, std::string>& data) {
    auto it = data.find("name");
    if (it != data.end()) {
        m_Name = it->second;
    }
    
    it = data.find("description");
    if (it != data.end()) {
        m_Description = it->second;
    }
}

// ============================================================================
// LinearTearPattern
// ============================================================================

LinearTearPattern::LinearTearPattern()
    : m_Length(0.3f)  // 30cm default
    , m_Width(0.02f)  // 2cm default
{
    m_Name = "Linear Tear";
    m_Description = "Straight line tear pattern";
}

LinearTearPattern::LinearTearPattern(float length, float width)
    : m_Length(length)
    , m_Width(width)
{
    m_Name = "Linear Tear";
    m_Description = "Straight line tear pattern";
}

std::vector<int> LinearTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<int> affectedParticles;
    
    float scaledLength = m_Length * scale;
    float scaledWidth = m_Width * scale;
    
    Vec3 normalizedDir = direction;
    normalizedDir.Normalize();
    
    // Calculate line start and end
    Vec3 lineStart = position - normalizedDir * (scaledLength * 0.5f);
    Vec3 lineEnd = position + normalizedDir * (scaledLength * 0.5f);
    
    // Find particles within width of the line
    for (size_t i = 0; i < particlePositions.size(); ++i) {
        const Vec3& particlePos = particlePositions[i];
        
        // Project particle onto line
        Vec3 toParticle = particlePos - lineStart;
        float projection = toParticle.Dot(normalizedDir);
        
        // Check if projection is within line segment
        if (projection >= 0.0f && projection <= scaledLength) {
            Vec3 closestPoint = lineStart + normalizedDir * projection;
            float distance = (particlePos - closestPoint).Length();
            
            if (distance <= scaledWidth) {
                affectedParticles.push_back(static_cast<int>(i));
            }
        }
    }
    
    return affectedParticles;
}

std::vector<Vec3> LinearTearPattern::GetVisualizationPoints(
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<Vec3> points;
    
    float scaledLength = m_Length * scale;
    Vec3 normalizedDir = direction;
    normalizedDir.Normalize();
    
    Vec3 start = position - normalizedDir * (scaledLength * 0.5f);
    Vec3 end = position + normalizedDir * (scaledLength * 0.5f);
    
    points.push_back(start);
    points.push_back(end);
    
    return points;
}

std::map<std::string, std::string> LinearTearPattern::Serialize() const {
    auto data = ClothTearPattern::Serialize();
    data["type"] = "Linear";
    data["length"] = std::to_string(m_Length);
    data["width"] = std::to_string(m_Width);
    return data;
}

void LinearTearPattern::Deserialize(const std::map<std::string, std::string>& data) {
    ClothTearPattern::Deserialize(data);
    
    auto it = data.find("length");
    if (it != data.end()) {
        m_Length = std::stof(it->second);
    }
    
    it = data.find("width");
    if (it != data.end()) {
        m_Width = std::stof(it->second);
    }
}

std::shared_ptr<ClothTearPattern> LinearTearPattern::Clone() const {
    auto clone = std::make_shared<LinearTearPattern>(m_Length, m_Width);
    clone->SetName(m_Name);
    clone->SetDescription(m_Description);
    return clone;
}

// ============================================================================
// RadialTearPattern
// ============================================================================

RadialTearPattern::RadialTearPattern()
    : m_Radius(0.15f)   // 15cm default
    , m_RayCount(6)      // 6 rays default
    , m_RayWidth(0.01f)  // 1cm default
{
    m_Name = "Radial Tear";
    m_Description = "Radial/star-shaped tear pattern";
}

RadialTearPattern::RadialTearPattern(float radius, int rayCount)
    : m_Radius(radius)
    , m_RayCount(rayCount)
    , m_RayWidth(0.01f)
{
    m_Name = "Radial Tear";
    m_Description = "Radial/star-shaped tear pattern";
}

std::vector<int> RadialTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<int> affectedParticles;
    
    float scaledRadius = m_Radius * scale;
    float scaledRayWidth = m_RayWidth * scale;
    
    // Use direction to determine plane orientation
    Vec3 normalizedDir = direction;
    normalizedDir.Normalize();
    
    // Create perpendicular vectors for the plane
    Vec3 right;
    if (std::abs(normalizedDir.y) < 0.9f) {
        right = Vec3(0, 1, 0).Cross(normalizedDir);
    } else {
        right = Vec3(1, 0, 0).Cross(normalizedDir);
    }
    right.Normalize();
    
    Vec3 up = normalizedDir.Cross(right);
    up.Normalize();
    
    // Generate rays
    float angleStep = 2.0f * 3.14159265f / m_RayCount;
    
    for (size_t i = 0; i < particlePositions.size(); ++i) {
        const Vec3& particlePos = particlePositions[i];
        Vec3 toParticle = particlePos - position;
        
        // Project onto plane
        float distAlongNormal = toParticle.Dot(normalizedDir);
        Vec3 planeProjection = toParticle - normalizedDir * distAlongNormal;
        float distFromCenter = planeProjection.Length();
        
        if (distFromCenter > scaledRadius || distFromCenter < 0.001f) {
            continue;
        }
        
        // Calculate angle in plane
        float x = planeProjection.Dot(right);
        float y = planeProjection.Dot(up);
        float angle = std::atan2(y, x);
        if (angle < 0) angle += 2.0f * 3.14159265f;
        
        // Check if particle is near any ray
        for (int ray = 0; ray < m_RayCount; ++ray) {
            float rayAngle = ray * angleStep;
            float angleDiff = std::abs(angle - rayAngle);
            
            // Handle wrap-around
            if (angleDiff > 3.14159265f) {
                angleDiff = 2.0f * 3.14159265f - angleDiff;
            }
            
            // Calculate angular width at this distance
            float angularWidth = scaledRayWidth / distFromCenter;
            
            if (angleDiff <= angularWidth) {
                affectedParticles.push_back(static_cast<int>(i));
                break;
            }
        }
    }
    
    return affectedParticles;
}

std::vector<Vec3> RadialTearPattern::GetVisualizationPoints(
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<Vec3> points;
    
    float scaledRadius = m_Radius * scale;
    Vec3 normalizedDir = direction;
    normalizedDir.Normalize();
    
    // Create perpendicular vectors
    Vec3 right;
    if (std::abs(normalizedDir.y) < 0.9f) {
        right = Vec3(0, 1, 0).Cross(normalizedDir);
    } else {
        right = Vec3(1, 0, 0).Cross(normalizedDir);
    }
    right.Normalize();
    
    Vec3 up = normalizedDir.Cross(right);
    up.Normalize();
    
    // Generate ray endpoints
    float angleStep = 2.0f * 3.14159265f / m_RayCount;
    for (int i = 0; i < m_RayCount; ++i) {
        float angle = i * angleStep;
        Vec3 rayDir = right * std::cos(angle) + up * std::sin(angle);
        
        points.push_back(position);
        points.push_back(position + rayDir * scaledRadius);
    }
    
    return points;
}

std::map<std::string, std::string> RadialTearPattern::Serialize() const {
    auto data = ClothTearPattern::Serialize();
    data["type"] = "Radial";
    data["radius"] = std::to_string(m_Radius);
    data["rayCount"] = std::to_string(m_RayCount);
    data["rayWidth"] = std::to_string(m_RayWidth);
    return data;
}

void RadialTearPattern::Deserialize(const std::map<std::string, std::string>& data) {
    ClothTearPattern::Deserialize(data);
    
    auto it = data.find("radius");
    if (it != data.end()) {
        m_Radius = std::stof(it->second);
    }
    
    it = data.find("rayCount");
    if (it != data.end()) {
        m_RayCount = std::stoi(it->second);
    }
    
    it = data.find("rayWidth");
    if (it != data.end()) {
        m_RayWidth = std::stof(it->second);
    }
}

std::shared_ptr<ClothTearPattern> RadialTearPattern::Clone() const {
    auto clone = std::make_shared<RadialTearPattern>(m_Radius, m_RayCount);
    clone->SetRayWidth(m_RayWidth);
    clone->SetName(m_Name);
    clone->SetDescription(m_Description);
    return clone;
}

// ============================================================================
// CrossTearPattern
// ============================================================================

CrossTearPattern::CrossTearPattern()
    : m_Length(0.2f)   // 20cm default
    , m_Width(0.015f)  // 1.5cm default
    , m_Angle(90.0f)   // 90 degrees default
{
    m_Name = "Cross Tear";
    m_Description = "X-shaped tear pattern";
}

CrossTearPattern::CrossTearPattern(float length, float width, float angle)
    : m_Length(length)
    , m_Width(width)
    , m_Angle(angle)
{
    m_Name = "Cross Tear";
    m_Description = "X-shaped tear pattern";
}

std::vector<int> CrossTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<int> affectedParticles;
    
    float scaledLength = m_Length * scale;
    float scaledWidth = m_Width * scale;
    
    Vec3 normalizedDir = direction;
    normalizedDir.Normalize();
    
    // Create perpendicular vector for second arm
    Vec3 perpendicular;
    if (std::abs(normalizedDir.y) < 0.9f) {
        perpendicular = Vec3(0, 1, 0).Cross(normalizedDir);
    } else {
        perpendicular = Vec3(1, 0, 0).Cross(normalizedDir);
    }
    perpendicular.Normalize();
    
    // Rotate perpendicular by angle/2
    float halfAngleRad = (m_Angle * 0.5f) * 3.14159265f / 180.0f;
    Vec3 arm1Dir = normalizedDir * std::cos(halfAngleRad) + perpendicular * std::sin(halfAngleRad);
    Vec3 arm2Dir = normalizedDir * std::cos(halfAngleRad) - perpendicular * std::sin(halfAngleRad);
    
    // Check particles against both arms
    for (size_t i = 0; i < particlePositions.size(); ++i) {
        const Vec3& particlePos = particlePositions[i];
        bool affected = false;
        
        // Check arm 1
        Vec3 lineStart1 = position - arm1Dir * (scaledLength * 0.5f);
        Vec3 lineEnd1 = position + arm1Dir * (scaledLength * 0.5f);
        Vec3 toParticle1 = particlePos - lineStart1;
        float projection1 = toParticle1.Dot(arm1Dir);
        
        if (projection1 >= 0.0f && projection1 <= scaledLength) {
            Vec3 closestPoint1 = lineStart1 + arm1Dir * projection1;
            float distance1 = (particlePos - closestPoint1).Length();
            if (distance1 <= scaledWidth) {
                affected = true;
            }
        }
        
        // Check arm 2
        if (!affected) {
            Vec3 lineStart2 = position - arm2Dir * (scaledLength * 0.5f);
            Vec3 lineEnd2 = position + arm2Dir * (scaledLength * 0.5f);
            Vec3 toParticle2 = particlePos - lineStart2;
            float projection2 = toParticle2.Dot(arm2Dir);
            
            if (projection2 >= 0.0f && projection2 <= scaledLength) {
                Vec3 closestPoint2 = lineStart2 + arm2Dir * projection2;
                float distance2 = (particlePos - closestPoint2).Length();
                if (distance2 <= scaledWidth) {
                    affected = true;
                }
            }
        }
        
        if (affected) {
            affectedParticles.push_back(static_cast<int>(i));
        }
    }
    
    return affectedParticles;
}

std::vector<Vec3> CrossTearPattern::GetVisualizationPoints(
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<Vec3> points;
    
    float scaledLength = m_Length * scale;
    Vec3 normalizedDir = direction;
    normalizedDir.Normalize();
    
    Vec3 perpendicular;
    if (std::abs(normalizedDir.y) < 0.9f) {
        perpendicular = Vec3(0, 1, 0).Cross(normalizedDir);
    } else {
        perpendicular = Vec3(1, 0, 0).Cross(normalizedDir);
    }
    perpendicular.Normalize();
    
    float halfAngleRad = (m_Angle * 0.5f) * 3.14159265f / 180.0f;
    Vec3 arm1Dir = normalizedDir * std::cos(halfAngleRad) + perpendicular * std::sin(halfAngleRad);
    Vec3 arm2Dir = normalizedDir * std::cos(halfAngleRad) - perpendicular * std::sin(halfAngleRad);
    
    // Arm 1
    points.push_back(position - arm1Dir * (scaledLength * 0.5f));
    points.push_back(position + arm1Dir * (scaledLength * 0.5f));
    
    // Arm 2
    points.push_back(position - arm2Dir * (scaledLength * 0.5f));
    points.push_back(position + arm2Dir * (scaledLength * 0.5f));
    
    return points;
}

std::map<std::string, std::string> CrossTearPattern::Serialize() const {
    auto data = ClothTearPattern::Serialize();
    data["type"] = "Cross";
    data["length"] = std::to_string(m_Length);
    data["width"] = std::to_string(m_Width);
    data["angle"] = std::to_string(m_Angle);
    return data;
}

void CrossTearPattern::Deserialize(const std::map<std::string, std::string>& data) {
    ClothTearPattern::Deserialize(data);
    
    auto it = data.find("length");
    if (it != data.end()) {
        m_Length = std::stof(it->second);
    }
    
    it = data.find("width");
    if (it != data.end()) {
        m_Width = std::stof(it->second);
    }
    
    it = data.find("angle");
    if (it != data.end()) {
        m_Angle = std::stof(it->second);
    }
}

std::shared_ptr<ClothTearPattern> CrossTearPattern::Clone() const {
    auto clone = std::make_shared<CrossTearPattern>(m_Length, m_Width, m_Angle);
    clone->SetName(m_Name);
    clone->SetDescription(m_Description);
    return clone;
}

// ============================================================================
// CustomPathTearPattern
// ============================================================================

CustomPathTearPattern::CustomPathTearPattern()
    : m_Width(0.02f)
    , m_SampleCount(20)
{
    m_Name = "Custom Path Tear";
    m_Description = "Custom bezier curve tear pattern";
}

void CustomPathTearPattern::AddControlPoint(const Vec3& point) {
    m_ControlPoints.push_back(point);
}

void CustomPathTearPattern::ClearControlPoints() {
    m_ControlPoints.clear();
}

Vec3 CustomPathTearPattern::EvaluateBezier(float t) const {
    if (m_ControlPoints.empty()) {
        return Vec3(0, 0, 0);
    }
    
    if (m_ControlPoints.size() == 1) {
        return m_ControlPoints[0];
    }
    
    // Linear interpolation for 2 points
    if (m_ControlPoints.size() == 2) {
        return m_ControlPoints[0] * (1.0f - t) + m_ControlPoints[1] * t;
    }
    
    // Quadratic bezier for 3 points
    if (m_ControlPoints.size() == 3) {
        float u = 1.0f - t;
        return m_ControlPoints[0] * (u * u) +
               m_ControlPoints[1] * (2.0f * u * t) +
               m_ControlPoints[2] * (t * t);
    }
    
    // Cubic bezier for 4+ points (use first 4)
    float u = 1.0f - t;
    return m_ControlPoints[0] * (u * u * u) +
           m_ControlPoints[1] * (3.0f * u * u * t) +
           m_ControlPoints[2] * (3.0f * u * t * t) +
           m_ControlPoints[3] * (t * t * t);
}

std::vector<int> CustomPathTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<int> affectedParticles;
    
    if (m_ControlPoints.size() < 2) {
        return affectedParticles;
    }
    
    float scaledWidth = m_Width * scale;
    
    // Sample curve
    std::vector<Vec3> curvePoints;
    for (int i = 0; i <= m_SampleCount; ++i) {
        float t = static_cast<float>(i) / m_SampleCount;
        Vec3 localPoint = EvaluateBezier(t);
        
        // Transform to world space
        Vec3 worldPoint = position + localPoint * scale;
        curvePoints.push_back(worldPoint);
    }
    
    // Check particles against curve segments
    for (size_t i = 0; i < particlePositions.size(); ++i) {
        const Vec3& particlePos = particlePositions[i];
        
        for (size_t j = 0; j < curvePoints.size() - 1; ++j) {
            Vec3 segmentStart = curvePoints[j];
            Vec3 segmentEnd = curvePoints[j + 1];
            Vec3 segmentDir = segmentEnd - segmentStart;
            float segmentLength = segmentDir.Length();
            
            if (segmentLength < 0.001f) continue;
            
            segmentDir = segmentDir * (1.0f / segmentLength);
            
            Vec3 toParticle = particlePos - segmentStart;
            float projection = toParticle.Dot(segmentDir);
            
            if (projection >= 0.0f && projection <= segmentLength) {
                Vec3 closestPoint = segmentStart + segmentDir * projection;
                float distance = (particlePos - closestPoint).Length();
                
                if (distance <= scaledWidth) {
                    affectedParticles.push_back(static_cast<int>(i));
                    break;
                }
            }
        }
    }
    
    return affectedParticles;
}

std::vector<Vec3> CustomPathTearPattern::GetVisualizationPoints(
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<Vec3> points;
    
    if (m_ControlPoints.size() < 2) {
        return points;
    }
    
    for (int i = 0; i <= m_SampleCount; ++i) {
        float t = static_cast<float>(i) / m_SampleCount;
        Vec3 localPoint = EvaluateBezier(t);
        Vec3 worldPoint = position + localPoint * scale;
        points.push_back(worldPoint);
    }
    
    return points;
}

std::map<std::string, std::string> CustomPathTearPattern::Serialize() const {
    auto data = ClothTearPattern::Serialize();
    data["type"] = "CustomPath";
    data["width"] = std::to_string(m_Width);
    data["sampleCount"] = std::to_string(m_SampleCount);
    
    // Serialize control points
    std::stringstream ss;
    for (size_t i = 0; i < m_ControlPoints.size(); ++i) {
        if (i > 0) ss << ";";
        ss << m_ControlPoints[i].x << "," 
           << m_ControlPoints[i].y << "," 
           << m_ControlPoints[i].z;
    }
    data["controlPoints"] = ss.str();
    
    return data;
}

void CustomPathTearPattern::Deserialize(const std::map<std::string, std::string>& data) {
    ClothTearPattern::Deserialize(data);
    
    auto it = data.find("width");
    if (it != data.end()) {
        m_Width = std::stof(it->second);
    }
    
    it = data.find("sampleCount");
    if (it != data.end()) {
        m_SampleCount = std::stoi(it->second);
    }
    
    it = data.find("controlPoints");
    if (it != data.end()) {
        m_ControlPoints.clear();
        std::stringstream ss(it->second);
        std::string point;
        
        while (std::getline(ss, point, ';')) {
            std::stringstream pointStream(point);
            std::string coord;
            Vec3 p;
            
            if (std::getline(pointStream, coord, ',')) p.x = std::stof(coord);
            if (std::getline(pointStream, coord, ',')) p.y = std::stof(coord);
            if (std::getline(pointStream, coord, ',')) p.z = std::stof(coord);
            
            m_ControlPoints.push_back(p);
        }
    }
}

std::shared_ptr<ClothTearPattern> CustomPathTearPattern::Clone() const {
    auto clone = std::make_shared<CustomPathTearPattern>();
    clone->SetWidth(m_Width);
    clone->SetSampleCount(m_SampleCount);
    clone->m_ControlPoints = m_ControlPoints;
    clone->SetName(m_Name);
    clone->SetDescription(m_Description);
    return clone;
}

// ============================================================================
// StressBasedTearPattern
// ============================================================================

StressBasedTearPattern::StressBasedTearPattern()
    : m_StressThreshold(1.5f)
    , m_PropagationRadius(0.1f)
{
    m_Name = "Stress-Based Tear";
    m_Description = "Automatic tear based on physics stress";
}

void StressBasedTearPattern::SetEdgeData(const std::vector<int>& indices, const std::vector<float>& restLengths) {
    m_EdgeIndices = indices;
    m_EdgeRestLengths = restLengths;
}

std::vector<int> StressBasedTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<int> affectedParticles;
    
    // Find particles near the stress point
    float scaledRadius = m_PropagationRadius * scale;
    
    for (size_t i = 0; i < particlePositions.size(); ++i) {
        float dist = (particlePositions[i] - position).Length();
        if (dist <= scaledRadius) {
            affectedParticles.push_back(static_cast<int>(i));
        }
    }
    
    return affectedParticles;
}

std::vector<Vec3> StressBasedTearPattern::GetVisualizationPoints(
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<Vec3> points;
    
    float scaledRadius = m_PropagationRadius * scale;
    
    // Draw a circle to represent stress area
    int segments = 16;
    for (int i = 0; i <= segments; ++i) {
        float angle = (static_cast<float>(i) / segments) * 2.0f * 3.14159265f;
        Vec3 offset(std::cos(angle) * scaledRadius, 0, std::sin(angle) * scaledRadius);
        points.push_back(position + offset);
    }
    
    return points;
}

std::map<std::string, std::string> StressBasedTearPattern::Serialize() const {
    auto data = ClothTearPattern::Serialize();
    data["type"] = "StressBased";
    data["stressThreshold"] = std::to_string(m_StressThreshold);
    data["propagationRadius"] = std::to_string(m_PropagationRadius);
    return data;
}

void StressBasedTearPattern::Deserialize(const std::map<std::string, std::string>& data) {
    ClothTearPattern::Deserialize(data);
    
    auto it = data.find("stressThreshold");
    if (it != data.end()) {
        m_StressThreshold = std::stof(it->second);
    }
    
    it = data.find("propagationRadius");
    if (it != data.end()) {
        m_PropagationRadius = std::stof(it->second);
    }
}

std::shared_ptr<ClothTearPattern> StressBasedTearPattern::Clone() const {
    auto clone = std::make_shared<StressBasedTearPattern>();
    clone->SetStressThreshold(m_StressThreshold);
    clone->SetPropagationRadius(m_PropagationRadius);
    clone->SetName(m_Name);
    clone->SetDescription(m_Description);
    return clone;
}
