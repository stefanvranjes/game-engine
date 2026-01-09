#include "TearResistanceMap.h"
#include <algorithm>
#include <iostream>

TearResistanceMap::TearResistanceMap()
    : m_DefaultResistance(1.0f)
{
}

TearResistanceMap::~TearResistanceMap() {
}

void TearResistanceMap::Initialize(int tetrahedronCount, float defaultResistance) {
    m_DefaultResistance = defaultResistance;
    m_Resistances.resize(tetrahedronCount, defaultResistance);
    
    std::cout << "TearResistanceMap initialized with " << tetrahedronCount 
              << " tetrahedra (default resistance: " << defaultResistance << ")" << std::endl;
}

void TearResistanceMap::SetTetrahedronResistance(int tetIndex, float resistance) {
    if (tetIndex >= 0 && tetIndex < static_cast<int>(m_Resistances.size())) {
        m_Resistances[tetIndex] = resistance;
    }
}

void TearResistanceMap::SetSphereResistance(
    const Vec3* tetCenters,
    int tetCount,
    const Vec3& center,
    float radius,
    float resistance)
{
    if (!tetCenters || tetCount <= 0) {
        return;
    }

    float radiusSq = radius * radius;
    int affectedCount = 0;

    for (int i = 0; i < tetCount && i < static_cast<int>(m_Resistances.size()); ++i) {
        float distSq = (tetCenters[i] - center).LengthSquared();
        if (distSq <= radiusSq) {
            m_Resistances[i] = resistance;
            affectedCount++;
        }
    }

    std::cout << "Set sphere resistance: " << affectedCount << " tetrahedra affected" << std::endl;
}

void TearResistanceMap::SetGradient(
    const Vec3* tetCenters,
    int tetCount,
    const Vec3& start,
    const Vec3& end,
    float startResistance,
    float endResistance)
{
    if (!tetCenters || tetCount <= 0) {
        return;
    }

    Vec3 dir = end - start;
    float length = dir.Length();

    if (length < 0.0001f) {
        return;
    }

    dir = dir * (1.0f / length);

    for (int i = 0; i < tetCount && i < static_cast<int>(m_Resistances.size()); ++i) {
        Vec3 toTet = tetCenters[i] - start;
        float projection = toTet.Dot(dir);
        float t = std::clamp(projection / length, 0.0f, 1.0f);

        float resistance = startResistance + t * (endResistance - startResistance);
        m_Resistances[i] = resistance;
    }

    std::cout << "Set resistance gradient from " << startResistance 
              << " to " << endResistance << std::endl;
}

float TearResistanceMap::GetResistance(int tetIndex) const {
    if (tetIndex >= 0 && tetIndex < static_cast<int>(m_Resistances.size())) {
        return m_Resistances[tetIndex];
    }
    return m_DefaultResistance;
}

void TearResistanceMap::Reset() {
    std::fill(m_Resistances.begin(), m_Resistances.end(), m_DefaultResistance);
    std::cout << "Reset all resistances to " << m_DefaultResistance << std::endl;
}
