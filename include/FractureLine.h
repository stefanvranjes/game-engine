#pragma once

#include "Math/Vec3.h"
#include <vector>

/**
 * @brief Defines a fracture line for controlled tearing
 */
class FractureLine {
public:
    /**
     * @brief Constructor
     * @param weaknessMultiplier How much weaker the line is (0.5 = 50% resistance)
     */
    explicit FractureLine(float weaknessMultiplier = 0.5f);
    
    /**
     * @brief Add point to fracture line
     */
    void AddPoint(const Vec3& point);
    
    /**
     * @brief Set weakness multiplier
     */
    void SetWeaknessMultiplier(float multiplier) { m_WeaknessMultiplier = multiplier; }
    
    /**
     * @brief Get weakness multiplier
     */
    float GetWeaknessMultiplier() const { return m_WeaknessMultiplier; }
    
    /**
     * @brief Get points
     */
    const std::vector<Vec3>& GetPoints() const { return m_Points; }
    
    /**
     * @brief Set line width
     */
    void SetWidth(float width) { m_Width = width; }
    
    /**
     * @brief Get line width
     */
    float GetWidth() const { return m_Width; }
    
    /**
     * @brief Apply fracture line to resistance map
     */
    void ApplyToResistanceMap(
        class TearResistanceMap& resistanceMap,
        const Vec3* tetCenters,
        int tetCount
    ) const;
    
    /**
     * @brief Remove point at index
     */
    void RemovePoint(int index);
    
    /**
     * @brief Insert point at index
     */
    void InsertPoint(int index, const Vec3& point);
    
    /**
     * @brief Update point position
     */
    void SetPoint(int index, const Vec3& point);
    
    /**
     * @brief Get point at index
     */
    Vec3 GetPoint(int index) const;
    
    /**
     * @brief Get number of points
     */
    int GetPointCount() const;
    
    /**
     * @brief Clear all points
     */
    void ClearPoints();

private:
    std::vector<Vec3> m_Points;
    float m_WeaknessMultiplier;  // 0.0 = no resistance, 1.0 = normal resistance
    float m_Width;                // Width of fracture line
};
