#include "AnisotropicMaterial.h"
#include <algorithm>

AnisotropicMaterial::AnisotropicMaterial() {
}

void AnisotropicMaterial::Initialize(int numTetrahedra) {
    m_FiberData.resize(numTetrahedra);
    
    // Default: fibers along X axis
    for (int i = 0; i < numTetrahedra; ++i) {
        m_FiberData[i] = FiberData(Vec3(1, 0, 0), 1.0f, 0.3f);
    }
}

void AnisotropicMaterial::SetFiberDirection(
    int tetIndex,
    const Vec3& direction,
    float longitudinalStiffness,
    float transverseStiffness
) {
    if (tetIndex < 0 || tetIndex >= static_cast<int>(m_FiberData.size())) {
        return;
    }
    
    m_FiberData[tetIndex] = FiberData(direction, longitudinalStiffness, transverseStiffness);
}

float AnisotropicMaterial::GetStiffnessMultiplier(int tetIndex, const Vec3& strainDir) const {
    if (tetIndex < 0 || tetIndex >= static_cast<int>(m_FiberData.size())) {
        return 1.0f; // Isotropic fallback
    }
    
    const FiberData& fiber = m_FiberData[tetIndex];
    
    // Normalize strain direction
    Vec3 normalizedStrain = strainDir;
    float len = normalizedStrain.Length();
    if (len < 1e-6f) {
        return 1.0f;
    }
    normalizedStrain = normalizedStrain * (1.0f / len);
    
    // Calculate alignment with fiber direction
    // alignment = cos(theta) where theta is angle between strain and fiber
    float alignment = normalizedStrain.Dot(fiber.direction);
    float alignmentSq = alignment * alignment; // cos^2(theta)
    
    // Interpolate between transverse and longitudinal stiffness
    // When aligned (alignmentSq = 1): use longitudinal stiffness
    // When perpendicular (alignmentSq = 0): use transverse stiffness
    float stiffness = fiber.transverseStiffness + 
                     (fiber.longitudinalStiffness - fiber.transverseStiffness) * alignmentSq;
    
    return stiffness;
}

const FiberData& AnisotropicMaterial::GetFiberData(int tetIndex) const {
    static FiberData defaultFiber;
    
    if (tetIndex < 0 || tetIndex >= static_cast<int>(m_FiberData.size())) {
        return defaultFiber;
    }
    
    return m_FiberData[tetIndex];
}

void AnisotropicMaterial::SetUniformFiberDirection(
    const Vec3& direction,
    float longitudinalStiffness,
    float transverseStiffness
) {
    FiberData uniformFiber(direction, longitudinalStiffness, transverseStiffness);
    
    for (auto& fiber : m_FiberData) {
        fiber = uniformFiber;
    }
}
