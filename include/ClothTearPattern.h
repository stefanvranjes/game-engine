#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <string>
#include <memory>
#include <map>

// Forward declaration
class SpatialHashGrid;

/**
 * @brief Tear pattern types
 */
enum class TearPatternType {
    Linear,        // Straight line tears (knife cuts, slashes)
    Radial,        // Circular/star-shaped tears (bullet impacts, explosions)
    Cross,         // X-shaped tears (dual-direction impacts)
    CustomPath,    // User-defined bezier curve paths
    StressBased,   // Automatic tears based on physics stress analysis
    Spiral,        // Spiral/vortex tears (tornado effects, rotational damage)
    Grid,          // Regular grid-based tears (fabric weave failures)
    Procedural     // Noise-based organic tears (natural damage patterns)
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
     * @brief Get particles affected by this pattern (optimized with spatial grid)
     * @param particlePositions All particle positions in the cloth
     * @param position Pattern application position (world space)
     * @param direction Pattern direction/orientation (normalized)
     * @param scale Pattern scale multiplier
     * @param spatialGrid Optional spatial hash grid for acceleration (nullptr = brute force)
     * @return Indices of particles to tear
     */
    virtual std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid
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

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid
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

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid
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

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid
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

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid
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

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid
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

/**
 * @brief Spiral tear pattern (vortex/tornado effects)
 */
class SpiralTearPattern : public ClothTearPattern {
public:
    SpiralTearPattern();
    SpiralTearPattern(float radius, float turns, float width);

    TearPatternType GetType() const override { return TearPatternType::Spiral; }

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid
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

    float GetTurns() const { return m_Turns; }
    void SetTurns(float turns) { m_Turns = turns; }

    float GetWidth() const { return m_Width; }
    void SetWidth(float width) { m_Width = width; }

    float GetTightnessFactor() const { return m_TightnessFactor; }
    void SetTightnessFactor(float factor) { m_TightnessFactor = factor; }

private:
    float m_Radius;           // Maximum radius of the spiral
    float m_Turns;            // Number of complete rotations
    float m_Width;            // Width of the spiral path
    float m_TightnessFactor;  // How tightly wound (0-1)
};

/**
 * @brief Grid tear pattern (regular grid-based tears)
 */
class GridTearPattern : public ClothTearPattern {
public:
    GridTearPattern();
    GridTearPattern(float gridSize, float cellSize, float lineWidth);

    TearPatternType GetType() const override { return TearPatternType::Grid; }

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid
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
    float GetGridSize() const { return m_GridSize; }
    void SetGridSize(float size) { m_GridSize = size; }

    float GetCellSize() const { return m_CellSize; }
    void SetCellSize(float size) { m_CellSize = size; }

    float GetLineWidth() const { return m_LineWidth; }
    void SetLineWidth(float width) { m_LineWidth = width; }

    float GetOrientation() const { return m_Orientation; }
    void SetOrientation(float angle) { m_Orientation = angle; }

private:
    float m_GridSize;      // Size of the grid area
    float m_CellSize;      // Spacing between grid lines
    float m_LineWidth;     // Width of each grid line
    float m_Orientation;   // Grid rotation angle (degrees)
};

/**
 * @brief Procedural tear pattern (noise-based organic tears)
 */
class ProceduralTearPattern : public ClothTearPattern {
public:
    ProceduralTearPattern();
    ProceduralTearPattern(float radius, float noiseScale, float threshold);

    TearPatternType GetType() const override { return TearPatternType::Procedural; }

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale = 1.0f
    ) const override;

    std::vector<int> GetAffectedParticles(
        const std::vector<Vec3>& particlePositions,
        const Vec3& position,
        const Vec3& direction,
        float scale,
        const SpatialHashGrid* spatialGrid
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

    float GetNoiseScale() const { return m_NoiseScale; }
    void SetNoiseScale(float scale) { m_NoiseScale = scale; }

    float GetThreshold() const { return m_Threshold; }
    void SetThreshold(float threshold) { m_Threshold = threshold; }

    int GetSeed() const { return m_Seed; }
    void SetSeed(int seed) { m_Seed = seed; }

    int GetOctaves() const { return m_Octaves; }
    void SetOctaves(int octaves) { m_Octaves = octaves; }

private:
    float m_Radius;       // Area of effect
    float m_NoiseScale;   // Scale of the noise pattern
    float m_Threshold;    // Density threshold for tear inclusion
    int m_Seed;           // Random seed for reproducibility
    int m_Octaves;        // Number of noise octaves for detail

    // Simple Perlin noise implementation
    float PerlinNoise(float x, float y, float z) const;
    float Fade(float t) const;
    float Lerp(float t, float a, float b) const;
    float Grad(int hash, float x, float y, float z) const;
};
