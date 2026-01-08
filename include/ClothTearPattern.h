#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <string>
#include <memory>
#include <map>

/**
 * @brief Tear pattern types
 */
enum class TearPatternType {
    Linear,        // Straight line tears (knife cuts, slashes)
    Radial,        // Circular/star-shaped tears (bullet impacts, explosions)
    Cross,         // X-shaped tears (dual-direction impacts)
    CustomPath,    // User-defined bezier curve paths
    StressBased    // Automatic tears based on physics stress analysis
};

/**
 * @brief Base class for cloth tear patterns
 * 
 * Defines reusable tear patterns that can be applied to cloth meshes.
 * Patterns generate lists of particles to tear based on position, direction, and scale.
 */
class ClothTearPattern {
public:
    virtual ~ClothTearPattern() = default;

    /**
     * @brief Get pattern type
     */
    virtual TearPatternType GetType() const = 0;

    /**
     * @brief Get pattern name
     */
    const std::string& GetName() const { return m_Name; }

    /**
     * @brief Set pattern name
     */
    void SetName(const std::string& name) { m_Name = name; }

    /**
     * @brief Get pattern description
     */
    const std::string& GetDescription() const { return m_Description; }

    /**
     * @brief Set pattern description
     */
    void SetDescription(const std::string& desc) { m_Description = desc; }

    /**
     * @brief Get particles affected by this pattern
     * @param particlePositions All particle positions in the cloth
     * @param position Pattern application position (world space)
     * @param direction Pattern direction/orientation (normalized)
     * @param scale Pattern scale multiplier
     * @return Indices of particles to tear
     */
    virtual std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const = 0;

    /**
     * @brief Get visualization points for pattern preview
     * @param position Pattern position
     * @param direction Pattern direction
     * @param scale Pattern scale
     * @return List of points defining the pattern shape
     */
    virtual std::vector<Vec3> GetVisualizationPoints(
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const = 0;

    /**
     * @brief Serialize pattern to JSON-like map
     */
    virtual std::map<std::string, std::string> Serialize() const;

    /**
     * @brief Deserialize pattern from JSON-like map
     */
    virtual void Deserialize(const std::map<std::string, std::string>& data);

    /**
     * @brief Clone this pattern
     */
    virtual std::shared_ptr<ClothTearPattern> Clone() const = 0;

protected:
    std::string m_Name;
    std::string m_Description;
};

/**
 * @brief Linear tear pattern (straight line cuts)
 */
class LinearTearPattern : public ClothTearPattern {
public:
    LinearTearPattern();
    LinearTearPattern(float length, float width);

    TearPatternType GetType() const override { return TearPatternType::Linear; }

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::vector<Vec3> GetVisualizationPoints(
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::map<std::string, std::string> Serialize() const override;
    void Deserialize(const std::map<std::string, std::string>& data) override;
    std::shared_ptr<ClothTearPattern> Clone() const override;

    // Parameters
    float GetLength() const { return m_Length; }
    void SetLength(float length) { m_Length = length; }

    float GetWidth() const { return m_Width; }
    void SetWidth(float width) { m_Width = width; }

private:
    float m_Length;  // Length of the tear
    float m_Width;   // Width/thickness of the tear
};

/**
 * @brief Radial tear pattern (explosion/impact tears)
 */
class RadialTearPattern : public ClothTearPattern {
public:
    RadialTearPattern();
    RadialTearPattern(float radius, int rayCount);

    TearPatternType GetType() const override { return TearPatternType::Radial; }

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::vector<Vec3> GetVisualizationPoints(
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::map<std::string, std::string> Serialize() const override;
    void Deserialize(const std::map<std::string, std::string>& data) override;
    std::shared_ptr<ClothTearPattern> Clone() const override;

    // Parameters
    float GetRadius() const { return m_Radius; }
    void SetRadius(float radius) { m_Radius = radius; }

    int GetRayCount() const { return m_RayCount; }
    void SetRayCount(int count) { m_RayCount = count; }

    float GetRayWidth() const { return m_RayWidth; }
    void SetRayWidth(float width) { m_RayWidth = width; }

private:
    float m_Radius;     // Radius of the radial pattern
    int m_RayCount;     // Number of rays emanating from center
    float m_RayWidth;   // Width of each ray
};

/**
 * @brief Cross tear pattern (X-shaped tears)
 */
class CrossTearPattern : public ClothTearPattern {
public:
    CrossTearPattern();
    CrossTearPattern(float length, float width, float angle);

    TearPatternType GetType() const override { return TearPatternType::Cross; }

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::vector<Vec3> GetVisualizationPoints(
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::map<std::string, std::string> Serialize() const override;
    void Deserialize(const std::map<std::string, std::string>& data) override;
    std::shared_ptr<ClothTearPattern> Clone() const override;

    // Parameters
    float GetLength() const { return m_Length; }
    void SetLength(float length) { m_Length = length; }

    float GetWidth() const { return m_Width; }
    void SetWidth(float width) { m_Width = width; }

    float GetAngle() const { return m_Angle; }
    void SetAngle(float angle) { m_Angle = angle; }

private:
    float m_Length;  // Length of each arm
    float m_Width;   // Width of each arm
    float m_Angle;   // Angle between arms (default 90 degrees)
};

/**
 * @brief Custom path tear pattern using bezier curves
 */
class CustomPathTearPattern : public ClothTearPattern {
public:
    CustomPathTearPattern();

    TearPatternType GetType() const override { return TearPatternType::CustomPath; }

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::vector<Vec3> GetVisualizationPoints(
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::map<std::string, std::string> Serialize() const override;
    void Deserialize(const std::map<std::string, std::string>& data) override;
    std::shared_ptr<ClothTearPattern> Clone() const override;

    // Path control
    void AddControlPoint(const Vec3& point);
    void ClearControlPoints();
    const std::vector<Vec3>& GetControlPoints() const { return m_ControlPoints; }

    float GetWidth() const { return m_Width; }
    void SetWidth(float width) { m_Width = width; }

    int GetSampleCount() const { return m_SampleCount; }
    void SetSampleCount(int count) { m_SampleCount = count; }

private:
    std::vector<Vec3> m_ControlPoints;  // Bezier control points (local space)
    float m_Width;                       // Width of the path
    int m_SampleCount;                   // Number of samples along curve

    // Evaluate bezier curve at t [0,1]
    Vec3 EvaluateBezier(float t) const;
};

/**
 * @brief Stress-based tear pattern (automatic based on physics)
 */
class StressBasedTearPattern : public ClothTearPattern {
public:
    StressBasedTearPattern();

    TearPatternType GetType() const override { return TearPatternType::StressBased; }

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::vector<Vec3> GetVisualizationPoints(
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::map<std::string, std::string> Serialize() const override;
    void Deserialize(const std::map<std::string, std::string>& data) override;
    std::shared_ptr<ClothTearPattern> Clone() const override;

    // Stress analysis parameters
    float GetStressThreshold() const { return m_StressThreshold; }
    void SetStressThreshold(float threshold) { m_StressThreshold = threshold; }

    float GetPropagationRadius() const { return m_PropagationRadius; }
    void SetPropagationRadius(float radius) { m_PropagationRadius = radius; }

    // Set edge data for stress analysis
    void SetEdgeData(const std::vector<int>& indices, const std::vector<float>& restLengths);

private:
    float m_StressThreshold;      // Stress ratio threshold for tearing
    float m_PropagationRadius;    // How far tears propagate from stress point
    
    // Edge data for stress calculation
    std::vector<int> m_EdgeIndices;
    std::vector<float> m_EdgeRestLengths;
};
