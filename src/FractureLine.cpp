#include "FractureLine.h"
#include "TearResistanceMap.h"
#include <algorithm>
#include <iostream>

FractureLine::FractureLine(float weaknessMultiplier)
    : m_WeaknessMultiplier(weaknessMultiplier)
    , m_Width(0.1f)
{
}

void FractureLine::AddPoint(const Vec3& point) {
    m_Points.push_back(point);
}

void FractureLine::ApplyToResistanceMap(
    TearResistanceMap& resistanceMap,
    const Vec3* tetCenters,
    int tetCount) const
{
    if (m_Points.size() < 2 || !tetCenters) {
        return;
    }
    
    int affectedCount = 0;
    float halfWidth = m_Width * 0.5f;
    
    // For each tetrahedron
    for (int tetIdx = 0; tetIdx < tetCount; ++tetIdx) {
        Vec3 tetCenter = tetCenters[tetIdx];
        
        // Find closest distance to fracture line
        float minDistance = std::numeric_limits<float>::max();
        
        // Check distance to each line segment
        for (size_t i = 0; i < m_Points.size() - 1; ++i) {
            Vec3 p0 = m_Points[i];
            Vec3 p1 = m_Points[i + 1];
            
            // Calculate distance to line segment
            Vec3 lineDir = p1 - p0;
            float lineLength = lineDir.Length();
            
            if (lineLength < 0.0001f) continue;
            
            lineDir = lineDir * (1.0f / lineLength);
            
            Vec3 toPoint = tetCenter - p0;
            float projection = toPoint.Dot(lineDir);
            
            // Clamp to segment
            projection = std::max(0.0f, std::min(lineLength, projection));
            
            Vec3 closestPoint = p0 + lineDir * projection;
            float distance = (tetCenter - closestPoint).Length();
            
            minDistance = std::min(minDistance, distance);
        }
        
        // If within width, apply weakness
        if (minDistance <= halfWidth) {
            // Get current resistance
            float currentResistance = resistanceMap.GetResistance(tetIdx);
            
            // Apply weakness multiplier
            float newResistance = currentResistance * m_WeaknessMultiplier;
            resistanceMap.SetTetrahedronResistance(tetIdx, newResistance);
            
            affectedCount++;
        }
    }
    
    std::cout << "Fracture line applied: " << affectedCount << " tetrahedra weakened" << std::endl;
}

void FractureLine::RemovePoint(int index) {
    if (index >= 0 && index < static_cast<int>(m_Points.size())) {
        m_Points.erase(m_Points.begin() + index);
    }
}

void FractureLine::InsertPoint(int index, const Vec3& point) {
    if (index >= 0 && index <= static_cast<int>(m_Points.size())) {
        m_Points.insert(m_Points.begin() + index, point);
    }
}

void FractureLine::SetPoint(int index, const Vec3& point) {
    if (index >= 0 && index < static_cast<int>(m_Points.size())) {
        m_Points[index] = point;
    }
}

Vec3 FractureLine::GetPoint(int index) const {
    if (index >= 0 && index < static_cast<int>(m_Points.size())) {
        return m_Points[index];
    }
    return Vec3(0, 0, 0);
}

int FractureLine::GetPointCount() const {
    return static_cast<int>(m_Points.size());
}

void FractureLine::ClearPoints() {
    m_Points.clear();
}

