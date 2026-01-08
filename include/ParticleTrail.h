#pragma once

#include "Math/Vec3.h"
#include "Math/Vec4.h"
#include <vector>
#include <memory>

// Represents a single point in a particle trail
struct TrailPoint {
    Vec3 position;
    Vec4 color;
    float width;
    float age;          // Time since this point was created
    float lifeRatio;    // 0.0 (just created) to 1.0 (about to die)
    
    TrailPoint()
        : position(0, 0, 0)
        , color(1, 1, 1, 1)
        , width(1.0f)
        , age(0.0f)
        , lifeRatio(0.0f)
    {}
    
    TrailPoint(const Vec3& pos, const Vec4& col, float w)
        : position(pos)
        , color(col)
        , width(w)
        , age(0.0f)
        , lifeRatio(0.0f)
    {}
};

// Manages trail data for a single particle
class ParticleTrail {
public:
    ParticleTrail(int maxPoints = 50);
    ~ParticleTrail();
    
    // Add a new point to the trail
    void AddPoint(const Vec3& position, const Vec4& color, float width);
    
    // Update trail points (age them, remove old ones)
    void Update(float deltaTime);
    
    // Clear all trail points
    void Clear();
    void Reset() { Clear(); }
    
    // Get all trail points
    const std::vector<TrailPoint>& GetPoints() const { return m_Points; }
    
    // Check if trail has any points
    bool IsEmpty() const { return m_Points.empty(); }
    
    // Get number of points
    int GetPointCount() const { return static_cast<int>(m_Points.size()); }
    
    // Configuration
    void SetMaxPoints(int maxPoints) { m_MaxPoints = maxPoints; }
    int GetMaxPoints() const { return m_MaxPoints; }
    
    void SetPointLifetime(float lifetime) { m_PointLifetime = lifetime; }
    float GetPointLifetime() const { return m_PointLifetime; }
    
    void SetMinDistance(float distance) { m_MinDistance = distance; }
    float GetMinDistance() const { return m_MinDistance; }

private:
    std::vector<TrailPoint> m_Points;
    int m_MaxPoints;
    float m_PointLifetime;  // How long each point lives
    float m_MinDistance;    // Minimum distance before adding new point
    Vec3 m_LastPosition;    // Last position where we added a point
};
