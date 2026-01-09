#include "ClothTearPattern.h"
#include "SpatialHashGrid.h"
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
    // Call optimized version with nullptr (brute force fallback)
    return GetAffectedParticles(particlePositions, position, direction, scale, nullptr);
}

std::vector<int> LinearTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale,
    const SpatialHashGrid* spatialGrid
) const {
    std::vector<int> affectedParticles;
    
    float scaledLength = m_Length * scale;
    float scaledWidth = m_Width * scale;
    
    Vec3 normalizedDir = direction;
    normalizedDir.Normalize();
    
    // Calculate line start and end
    Vec3 lineStart = position - normalizedDir * (scaledLength * 0.5f);
    Vec3 lineEnd = position + normalizedDir * (scaledLength * 0.5f);
    
    // Get candidate particles
    std::vector<int> candidates;
    if (spatialGrid && !spatialGrid->IsEmpty()) {
        // Use spatial grid for acceleration
        candidates = spatialGrid->QueryLineSegment(lineStart, lineEnd, scaledWidth);
    } else {
        // Fallback: check all particles
        candidates.reserve(particlePositions.size());
        for (size_t i = 0; i < particlePositions.size(); ++i) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    
    // Filter candidates with precise distance check
    for (int particleIdx : candidates) {
        const Vec3& particlePos = particlePositions[particleIdx];
        
        // Project particle onto line
        Vec3 toParticle = particlePos - lineStart;
        float projection = toParticle.Dot(normalizedDir);
        
        // Check if projection is within line segment
        if (projection >= 0.0f && projection <= scaledLength) {
            Vec3 closestPoint = lineStart + normalizedDir * projection;
            float distance = (particlePos - closestPoint).Length();
            
            if (distance <= scaledWidth) {
                affectedParticles.push_back(particleIdx);
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
    // Call optimized version with nullptr (brute force fallback)
    return GetAffectedParticles(particlePositions, position, direction, scale, nullptr);
}

std::vector<int> RadialTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale,
    const SpatialHashGrid* spatialGrid
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
    
    // Get candidate particles
    std::vector<int> candidates;
    if (spatialGrid && !spatialGrid->IsEmpty()) {
        // Use spatial grid for acceleration
        candidates = spatialGrid->QuerySphere(position, scaledRadius);
    } else {
        // Fallback: check all particles
        candidates.reserve(particlePositions.size());
        for (size_t i = 0; i < particlePositions.size(); ++i) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    
    // Filter candidates with precise ray checks
    for (int particleIdx : candidates) {
        const Vec3& particlePos = particlePositions[particleIdx];
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
                affectedParticles.push_back(particleIdx);
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
    return GetAffectedParticles(particlePositions, position, direction, scale, nullptr);
}

std::vector<int> CrossTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale,
    const SpatialHashGrid* spatialGrid
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
    
    // Calculate line segments for both arms
    Vec3 lineStart1 = position - arm1Dir * (scaledLength * 0.5f);
    Vec3 lineEnd1 = position + arm1Dir * (scaledLength * 0.5f);
    Vec3 lineStart2 = position - arm2Dir * (scaledLength * 0.5f);
    Vec3 lineEnd2 = position + arm2Dir * (scaledLength * 0.5f);
    
    // Get candidate particles
    std::vector<int> candidates;
    if (spatialGrid && !spatialGrid->IsEmpty()) {
        // Query both arms and merge results
        std::vector<int> candidates1 = spatialGrid->QueryLineSegment(lineStart1, lineEnd1, scaledWidth);
        std::vector<int> candidates2 = spatialGrid->QueryLineSegment(lineStart2, lineEnd2, scaledWidth);
        
        // Merge and deduplicate
        candidates = candidates1;
        for (int idx : candidates2) {
            if (std::find(candidates.begin(), candidates.end(), idx) == candidates.end()) {
                candidates.push_back(idx);
            }
        }
    } else {
        // Fallback: check all particles
        candidates.reserve(particlePositions.size());
        for (size_t i = 0; i < particlePositions.size(); ++i) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    
    // Check particles against both arms
    for (int particleIdx : candidates) {
        const Vec3& particlePos = particlePositions[particleIdx];
        bool affected = false;
        
        // Check arm 1
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
            affectedParticles.push_back(particleIdx);
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
    return GetAffectedParticles(particlePositions, position, direction, scale, nullptr);
}

std::vector<int> CustomPathTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale,
    const SpatialHashGrid* spatialGrid
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
    
    // Get candidate particles
    std::vector<int> candidates;
    if (spatialGrid && !spatialGrid->IsEmpty()) {
        // Query each curve segment and merge results
        std::vector<bool> included(particlePositions.size(), false);
        
        for (size_t j = 0; j < curvePoints.size() - 1; ++j) {
            std::vector<int> segmentCandidates = spatialGrid->QueryLineSegment(
                curvePoints[j], curvePoints[j + 1], scaledWidth
            );
            
            for (int idx : segmentCandidates) {
                if (!included[idx]) {
                    included[idx] = true;
                    candidates.push_back(idx);
                }
            }
        }
    } else {
        // Fallback: check all particles
        candidates.reserve(particlePositions.size());
        for (size_t i = 0; i < particlePositions.size(); ++i) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    
    // Check particles against curve segments
    for (int particleIdx : candidates) {
        const Vec3& particlePos = particlePositions[particleIdx];
        
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
                    affectedParticles.push_back(particleIdx);
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
    return GetAffectedParticles(particlePositions, position, direction, scale, nullptr);
}

std::vector<int> StressBasedTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale,
    const SpatialHashGrid* spatialGrid
) const {
    std::vector<int> affectedParticles;
    
    // Find particles near the stress point
    float scaledRadius = m_PropagationRadius * scale;
    
    if (spatialGrid && !spatialGrid->IsEmpty()) {
        // Use spatial grid for acceleration
        affectedParticles = spatialGrid->QuerySphere(position, scaledRadius);
    } else {
        // Fallback: check all particles
        for (size_t i = 0; i < particlePositions.size(); ++i) {
            float dist = (particlePositions[i] - position).Length();
            if (dist <= scaledRadius) {
                affectedParticles.push_back(static_cast<int>(i));
            }
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

// ============================================================================
// SpiralTearPattern
// ============================================================================

SpiralTearPattern::SpiralTearPattern()
    : m_Radius(0.25f)         // 25cm default
    , m_Turns(2.5f)           // 2.5 rotations default
    , m_Width(0.015f)         // 1.5cm default
    , m_TightnessFactor(0.7f) // Moderately tight
{
    m_Name = "Spiral Tear";
    m_Description = "Spiral/vortex tear pattern";
}

SpiralTearPattern::SpiralTearPattern(float radius, float turns, float width)
    : m_Radius(radius)
    , m_Turns(turns)
    , m_Width(width)
    , m_TightnessFactor(0.7f)
{
    m_Name = "Spiral Tear";
    m_Description = "Spiral/vortex tear pattern";
}

std::vector<int> SpiralTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    return GetAffectedParticles(particlePositions, position, direction, scale, nullptr);
}

std::vector<int> SpiralTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale,
    const SpatialHashGrid* spatialGrid
) const {
    std::vector<int> affectedParticles;
    
    float scaledRadius = m_Radius * scale;
    float scaledWidth = m_Width * scale;
    
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
    
    // Get candidate particles
    std::vector<int> candidates;
    if (spatialGrid && !spatialGrid->IsEmpty()) {
        // Use spatial grid for acceleration
        candidates = spatialGrid->QuerySphere(position, scaledRadius);
    } else {
        // Fallback: check all particles
        candidates.reserve(particlePositions.size());
        for (size_t i = 0; i < particlePositions.size(); ++i) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    
    // Archimedean spiral: r = a + b*theta
    // where a controls starting radius, b controls spacing
    float maxTheta = m_Turns * 2.0f * 3.14159265f;
    float b = (scaledRadius * m_TightnessFactor) / maxTheta;
    float a = scaledRadius * (1.0f - m_TightnessFactor);
    
    // Filter candidates with precise spiral checks
    for (int particleIdx : candidates) {
        const Vec3& particlePos = particlePositions[particleIdx];
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
        
        // For each possible theta value (spiral can wrap), check distance to spiral
        bool onSpiral = false;
        for (float theta = 0; theta <= maxTheta; theta += 0.1f) {
            float spiralRadius = a + b * theta;
            float spiralAngle = theta;
            
            // Normalize angle to [0, 2π]
            while (spiralAngle > 2.0f * 3.14159265f) {
                spiralAngle -= 2.0f * 3.14159265f;
            }
            
            // Check if particle is near this point on the spiral
            float angleDiff = std::abs(angle - spiralAngle);
            if (angleDiff > 3.14159265f) {
                angleDiff = 2.0f * 3.14159265f - angleDiff;
            }
            
            float radiusDiff = std::abs(distFromCenter - spiralRadius);
            
            // Calculate distance to spiral curve
            float angularDist = angleDiff * distFromCenter;
            float totalDist = std::sqrt(radiusDiff * radiusDiff + angularDist * angularDist);
            
            if (totalDist <= scaledWidth) {
                onSpiral = true;
                break;
            }
        }
        
        if (onSpiral) {
            affectedParticles.push_back(particleIdx);
        }
    }
    
    return affectedParticles;
}

std::vector<Vec3> SpiralTearPattern::GetVisualizationPoints(
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
    
    // Generate spiral curve
    float maxTheta = m_Turns * 2.0f * 3.14159265f;
    float b = (scaledRadius * m_TightnessFactor) / maxTheta;
    float a = scaledRadius * (1.0f - m_TightnessFactor);
    
    int segments = static_cast<int>(m_Turns * 32); // 32 points per turn
    for (int i = 0; i <= segments; ++i) {
        float t = static_cast<float>(i) / segments;
        float theta = t * maxTheta;
        float r = a + b * theta;
        
        Vec3 spiralDir = right * std::cos(theta) + up * std::sin(theta);
        points.push_back(position + spiralDir * r);
    }
    
    return points;
}

std::map<std::string, std::string> SpiralTearPattern::Serialize() const {
    auto data = ClothTearPattern::Serialize();
    data["type"] = "Spiral";
    data["radius"] = std::to_string(m_Radius);
    data["turns"] = std::to_string(m_Turns);
    data["width"] = std::to_string(m_Width);
    data["tightnessFactor"] = std::to_string(m_TightnessFactor);
    return data;
}

void SpiralTearPattern::Deserialize(const std::map<std::string, std::string>& data) {
    ClothTearPattern::Deserialize(data);
    
    auto it = data.find("radius");
    if (it != data.end()) {
        m_Radius = std::stof(it->second);
    }
    
    it = data.find("turns");
    if (it != data.end()) {
        m_Turns = std::stof(it->second);
    }
    
    it = data.find("width");
    if (it != data.end()) {
        m_Width = std::stof(it->second);
    }
    
    it = data.find("tightnessFactor");
    if (it != data.end()) {
        m_TightnessFactor = std::stof(it->second);
    }
}

std::shared_ptr<ClothTearPattern> SpiralTearPattern::Clone() const {
    auto clone = std::make_shared<SpiralTearPattern>(m_Radius, m_Turns, m_Width);
    clone->SetTightnessFactor(m_TightnessFactor);
    clone->SetName(m_Name);
    clone->SetDescription(m_Description);
    return clone;
}

// ============================================================================
// GridTearPattern
// ============================================================================

GridTearPattern::GridTearPattern()
    : m_GridSize(0.4f)     // 40cm default
    , m_CellSize(0.05f)    // 5cm cells default
    , m_LineWidth(0.01f)   // 1cm lines default
    , m_Orientation(0.0f)  // No rotation default
{
    m_Name = "Grid Tear";
    m_Description = "Regular grid-based tear pattern";
}

GridTearPattern::GridTearPattern(float gridSize, float cellSize, float lineWidth)
    : m_GridSize(gridSize)
    , m_CellSize(cellSize)
    , m_LineWidth(lineWidth)
    , m_Orientation(0.0f)
{
    m_Name = "Grid Tear";
    m_Description = "Regular grid-based tear pattern";
}

std::vector<int> GridTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    return GetAffectedParticles(particlePositions, position, direction, scale, nullptr);
}

std::vector<int> GridTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale,
    const SpatialHashGrid* spatialGrid
) const {
    std::vector<int> affectedParticles;
    
    float scaledGridSize = m_GridSize * scale;
    float scaledCellSize = m_CellSize * scale;
    float scaledLineWidth = m_LineWidth * scale;
    
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
    
    // Apply orientation rotation
    if (m_Orientation != 0.0f) {
        float angleRad = m_Orientation * 3.14159265f / 180.0f;
        float cosA = std::cos(angleRad);
        float sinA = std::sin(angleRad);
        Vec3 newRight = right * cosA + up * sinA;
        Vec3 newUp = up * cosA - right * sinA;
        right = newRight;
        up = newUp;
    }
    
    // Get candidate particles
    std::vector<int> candidates;
    if (spatialGrid && !spatialGrid->IsEmpty()) {
        // Query bounding box of grid
        float halfSize = scaledGridSize * 0.5f;
        candidates = spatialGrid->QuerySphere(position, halfSize * 1.415f); // sqrt(2) for diagonal
    } else {
        // Fallback: check all particles
        candidates.reserve(particlePositions.size());
        for (size_t i = 0; i < particlePositions.size(); ++i) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    
    float halfSize = scaledGridSize * 0.5f;
    
    // Filter candidates
    for (int particleIdx : candidates) {
        const Vec3& particlePos = particlePositions[particleIdx];
        Vec3 toParticle = particlePos - position;
        
        // Project onto plane
        float x = toParticle.Dot(right);
        float y = toParticle.Dot(up);
        
        // Check if within grid bounds
        if (std::abs(x) > halfSize || std::abs(y) > halfSize) {
            continue;
        }
        
        // Offset to grid origin
        x += halfSize;
        y += halfSize;
        
        // Check distance to nearest grid line
        float distToVerticalLine = std::fmod(x, scaledCellSize);
        if (distToVerticalLine > scaledCellSize * 0.5f) {
            distToVerticalLine = scaledCellSize - distToVerticalLine;
        }
        
        float distToHorizontalLine = std::fmod(y, scaledCellSize);
        if (distToHorizontalLine > scaledCellSize * 0.5f) {
            distToHorizontalLine = scaledCellSize - distToHorizontalLine;
        }
        
        float minDist = std::min(distToVerticalLine, distToHorizontalLine);
        
        if (minDist <= scaledLineWidth * 0.5f) {
            affectedParticles.push_back(particleIdx);
        }
    }
    
    return affectedParticles;
}

std::vector<Vec3> GridTearPattern::GetVisualizationPoints(
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<Vec3> points;
    
    float scaledGridSize = m_GridSize * scale;
    float scaledCellSize = m_CellSize * scale;
    
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
    
    // Apply orientation rotation
    if (m_Orientation != 0.0f) {
        float angleRad = m_Orientation * 3.14159265f / 180.0f;
        float cosA = std::cos(angleRad);
        float sinA = std::sin(angleRad);
        Vec3 newRight = right * cosA + up * sinA;
        Vec3 newUp = up * cosA - right * sinA;
        right = newRight;
        up = newUp;
    }
    
    float halfSize = scaledGridSize * 0.5f;
    int numLines = static_cast<int>(scaledGridSize / scaledCellSize) + 1;
    
    // Vertical lines
    for (int i = 0; i <= numLines; ++i) {
        float x = -halfSize + i * scaledCellSize;
        if (std::abs(x) > halfSize) continue;
        
        Vec3 start = position + right * x - up * halfSize;
        Vec3 end = position + right * x + up * halfSize;
        points.push_back(start);
        points.push_back(end);
    }
    
    // Horizontal lines
    for (int i = 0; i <= numLines; ++i) {
        float y = -halfSize + i * scaledCellSize;
        if (std::abs(y) > halfSize) continue;
        
        Vec3 start = position - right * halfSize + up * y;
        Vec3 end = position + right * halfSize + up * y;
        points.push_back(start);
        points.push_back(end);
    }
    
    return points;
}

std::map<std::string, std::string> GridTearPattern::Serialize() const {
    auto data = ClothTearPattern::Serialize();
    data["type"] = "Grid";
    data["gridSize"] = std::to_string(m_GridSize);
    data["cellSize"] = std::to_string(m_CellSize);
    data["lineWidth"] = std::to_string(m_LineWidth);
    data["orientation"] = std::to_string(m_Orientation);
    return data;
}

void GridTearPattern::Deserialize(const std::map<std::string, std::string>& data) {
    ClothTearPattern::Deserialize(data);
    
    auto it = data.find("gridSize");
    if (it != data.end()) {
        m_GridSize = std::stof(it->second);
    }
    
    it = data.find("cellSize");
    if (it != data.end()) {
        m_CellSize = std::stof(it->second);
    }
    
    it = data.find("lineWidth");
    if (it != data.end()) {
        m_LineWidth = std::stof(it->second);
    }
    
    it = data.find("orientation");
    if (it != data.end()) {
        m_Orientation = std::stof(it->second);
    }
}

std::shared_ptr<ClothTearPattern> GridTearPattern::Clone() const {
    auto clone = std::make_shared<GridTearPattern>(m_GridSize, m_CellSize, m_LineWidth);
    clone->SetOrientation(m_Orientation);
    clone->SetName(m_Name);
    clone->SetDescription(m_Description);
    return clone;
}

// ============================================================================
// ProceduralTearPattern
// ============================================================================

ProceduralTearPattern::ProceduralTearPattern()
    : m_Radius(0.3f)       // 30cm default
    , m_NoiseScale(5.0f)   // Medium frequency
    , m_Threshold(0.5f)    // 50% density
    , m_Seed(12345)        // Default seed
    , m_Octaves(3)         // 3 octaves for detail
{
    m_Name = "Procedural Tear";
    m_Description = "Noise-based organic tear pattern";
}

ProceduralTearPattern::ProceduralTearPattern(float radius, float noiseScale, float threshold)
    : m_Radius(radius)
    , m_NoiseScale(noiseScale)
    , m_Threshold(threshold)
    , m_Seed(12345)
    , m_Octaves(3)
{
    m_Name = "Procedural Tear";
    m_Description = "Noise-based organic tear pattern";
}

float ProceduralTearPattern::Fade(float t) const {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float ProceduralTearPattern::Lerp(float t, float a, float b) const {
    return a + t * (b - a);
}

float ProceduralTearPattern::Grad(int hash, float x, float y, float z) const {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

float ProceduralTearPattern::PerlinNoise(float x, float y, float z) const {
    // Simple permutation table based on seed
    static int p[512];
    static bool initialized = false;
    static int lastSeed = 0;
    
    if (!initialized || lastSeed != m_Seed) {
        // Initialize permutation table
        for (int i = 0; i < 256; ++i) {
            p[i] = i;
        }
        
        // Shuffle based on seed
        unsigned int seed = static_cast<unsigned int>(m_Seed);
        for (int i = 255; i > 0; --i) {
            seed = seed * 1103515245 + 12345; // LCG
            int j = seed % (i + 1);
            std::swap(p[i], p[j]);
        }
        
        // Duplicate for overflow
        for (int i = 0; i < 256; ++i) {
            p[256 + i] = p[i];
        }
        
        initialized = true;
        lastSeed = m_Seed;
    }
    
    // Find unit cube containing point
    int X = static_cast<int>(std::floor(x)) & 255;
    int Y = static_cast<int>(std::floor(y)) & 255;
    int Z = static_cast<int>(std::floor(z)) & 255;
    
    // Find relative position in cube
    x -= std::floor(x);
    y -= std::floor(y);
    z -= std::floor(z);
    
    // Compute fade curves
    float u = Fade(x);
    float v = Fade(y);
    float w = Fade(z);
    
    // Hash coordinates of cube corners
    int A = p[X] + Y;
    int AA = p[A] + Z;
    int AB = p[A + 1] + Z;
    int B = p[X + 1] + Y;
    int BA = p[B] + Z;
    int BB = p[B + 1] + Z;
    
    // Blend results from 8 corners
    return Lerp(w, 
        Lerp(v, 
            Lerp(u, Grad(p[AA], x, y, z), Grad(p[BA], x - 1, y, z)),
            Lerp(u, Grad(p[AB], x, y - 1, z), Grad(p[BB], x - 1, y - 1, z))
        ),
        Lerp(v,
            Lerp(u, Grad(p[AA + 1], x, y, z - 1), Grad(p[BA + 1], x - 1, y, z - 1)),
            Lerp(u, Grad(p[AB + 1], x, y - 1, z - 1), Grad(p[BB + 1], x - 1, y - 1, z - 1))
        )
    );
}

std::vector<int> ProceduralTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    return GetAffectedParticles(particlePositions, position, direction, scale, nullptr);
}

std::vector<int> ProceduralTearPattern::GetAffectedParticles(
    const std::vector<Vec3>& particlePositions,
    const Vec3& position,
    const Vec3& direction,
    float scale,
    const SpatialHashGrid* spatialGrid
) const {
    std::vector<int> affectedParticles;
    
    float scaledRadius = m_Radius * scale;
    
    // Get candidate particles
    std::vector<int> candidates;
    if (spatialGrid && !spatialGrid->IsEmpty()) {
        candidates = spatialGrid->QuerySphere(position, scaledRadius);
    } else {
        candidates.reserve(particlePositions.size());
        for (size_t i = 0; i < particlePositions.size(); ++i) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    
    // Filter candidates using noise
    for (int particleIdx : candidates) {
        const Vec3& particlePos = particlePositions[particleIdx];
        Vec3 toParticle = particlePos - position;
        float dist = toParticle.Length();
        
        if (dist > scaledRadius) {
            continue;
        }
        
        // Multi-octave noise
        float noiseValue = 0.0f;
        float amplitude = 1.0f;
        float frequency = m_NoiseScale;
        float maxValue = 0.0f;
        
        for (int octave = 0; octave < m_Octaves; ++octave) {
            noiseValue += PerlinNoise(
                particlePos.x * frequency,
                particlePos.y * frequency,
                particlePos.z * frequency
            ) * amplitude;
            
            maxValue += amplitude;
            amplitude *= 0.5f;
            frequency *= 2.0f;
        }
        
        // Normalize to [0, 1]
        noiseValue = (noiseValue / maxValue) * 0.5f + 0.5f;
        
        // Apply radial falloff
        float falloff = 1.0f - (dist / scaledRadius);
        noiseValue *= falloff;
        
        if (noiseValue >= m_Threshold) {
            affectedParticles.push_back(particleIdx);
        }
    }
    
    return affectedParticles;
}

std::vector<Vec3> ProceduralTearPattern::GetVisualizationPoints(
    const Vec3& position,
    const Vec3& direction,
    float scale
) const {
    std::vector<Vec3> points;
    
    float scaledRadius = m_Radius * scale;
    
    // Sample points in a grid and visualize threshold boundary
    int samples = 20;
    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < samples; ++j) {
            float u = (static_cast<float>(i) / (samples - 1)) * 2.0f - 1.0f;
            float v = (static_cast<float>(j) / (samples - 1)) * 2.0f - 1.0f;
            
            float dist = std::sqrt(u * u + v * v);
            if (dist > 1.0f) continue;
            
            Vec3 offset(u * scaledRadius, 0, v * scaledRadius);
            Vec3 samplePos = position + offset;
            
            // Evaluate noise
            float noiseValue = 0.0f;
            float amplitude = 1.0f;
            float frequency = m_NoiseScale;
            float maxValue = 0.0f;
            
            for (int octave = 0; octave < m_Octaves; ++octave) {
                noiseValue += PerlinNoise(
                    samplePos.x * frequency,
                    samplePos.y * frequency,
                    samplePos.z * frequency
                ) * amplitude;
                
                maxValue += amplitude;
                amplitude *= 0.5f;
                frequency *= 2.0f;
            }
            
            noiseValue = (noiseValue / maxValue) * 0.5f + 0.5f;
            float falloff = 1.0f - dist;
            noiseValue *= falloff;
            
            if (noiseValue >= m_Threshold) {
                points.push_back(samplePos);
            }
        }
    }
    
    return points;
}

std::map<std::string, std::string> ProceduralTearPattern::Serialize() const {
    auto data = ClothTearPattern::Serialize();
    data["type"] = "Procedural";
    data["radius"] = std::to_string(m_Radius);
    data["noiseScale"] = std::to_string(m_NoiseScale);
    data["threshold"] = std::to_string(m_Threshold);
    data["seed"] = std::to_string(m_Seed);
    data["octaves"] = std::to_string(m_Octaves);
    return data;
}

void ProceduralTearPattern::Deserialize(const std::map<std::string, std::string>& data) {
    ClothTearPattern::Deserialize(data);
    
    auto it = data.find("radius");
    if (it != data.end()) {
        m_Radius = std::stof(it->second);
    }
    
    it = data.find("noiseScale");
    if (it != data.end()) {
        m_NoiseScale = std::stof(it->second);
    }
    
    it = data.find("threshold");
    if (it != data.end()) {
        m_Threshold = std::stof(it->second);
    }
    
    it = data.find("seed");
    if (it != data.end()) {
        m_Seed = std::stoi(it->second);
    }
    
    it = data.find("octaves");
    if (it != data.end()) {
        m_Octaves = std::stoi(it->second);
    }
}

std::shared_ptr<ClothTearPattern> ProceduralTearPattern::Clone() const {
    auto clone = std::make_shared<ProceduralTearPattern>(m_Radius, m_NoiseScale, m_Threshold);
    clone->SetSeed(m_Seed);
    clone->SetOctaves(m_Octaves);
    clone->SetName(m_Name);
    clone->SetDescription(m_Description);
    return clone;
}

