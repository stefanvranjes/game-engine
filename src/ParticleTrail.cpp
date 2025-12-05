#include "ParticleTrail.h"
#include <cmath>
#include <algorithm>

ParticleTrail::ParticleTrail(int maxPoints)
    : m_MaxPoints(maxPoints)
    , m_PointLifetime(1.0f)
    , m_MinDistance(0.1f)
    , m_LastPosition(0, 0, 0)
{
    m_Points.reserve(maxPoints);
}

ParticleTrail::~ParticleTrail() {
}

void ParticleTrail::AddPoint(const Vec3& position, const Vec4& color, float width) {
    // Check if we should add a point based on distance
    if (!m_Points.empty()) {
        Vec3 delta = position - m_LastPosition;
        float distSq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        
        // Don't add point if too close to last one
        if (distSq < m_MinDistance * m_MinDistance) {
            return;
        }
    }
    
    // Add new point
    TrailPoint point(position, color, width);
    m_Points.push_back(point);
    m_LastPosition = position;
    
    // Remove oldest point if we exceeded max
    if (static_cast<int>(m_Points.size()) > m_MaxPoints) {
        m_Points.erase(m_Points.begin());
    }
}

void ParticleTrail::Update(float deltaTime) {
    // Update all points
    for (auto& point : m_Points) {
        point.age += deltaTime;
        point.lifeRatio = point.age / m_PointLifetime;
    }
    
    // Remove dead points (those that exceeded lifetime)
    m_Points.erase(
        std::remove_if(m_Points.begin(), m_Points.end(),
            [this](const TrailPoint& p) { return p.age >= m_PointLifetime; }),
        m_Points.end()
    );
}

void ParticleTrail::Clear() {
    m_Points.clear();
    m_LastPosition = Vec3(0, 0, 0);
}
