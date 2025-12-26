#pragma once
#include "Math/Vec3.h"
#include <memory>
#include <unordered_map>
class GameObject;
class AudioSource;
/**
 * @class AudioOcclusion
 * @brief Physics-based audio occlusion system using raycasting and material properties.
 * 
 * Computes how much audio is blocked or attenuated between a sound source and the listener
 * based on intervening geometry, materials, and distance through obstacles.
 */
// Stub header - not implemented
class AudioOcclusion {
public:
    /**
     * @enum MaterialType
     * @brief Audio material properties (absorption, reflection, etc.).
     */
    enum class MaterialType {
        Air,            // 0% occlusion
        Glass,          // 10% occlusion (transparent to sound)
        Drywall,        // 40% occlusion
        Brick,          // 60% occlusion
        Wood,           // 50% occlusion
        Metal,          // 70% occlusion
        Concrete,       // 75% occlusion
        Stone,          // 80% occlusion
        Water,          // 85% occlusion (heavy dampening)
        Custom          // User-defined
    };
    /**
     * @struct MaterialProperties
     * @brief Audio properties of a material.
     */
    struct MaterialProperties {
        float occlusionFactor = 0.0f;  // 0.0 = transparent, 1.0 = fully opaque
        float dampingFactor = 0.0f;    // How much high frequencies are absorbed (0.0 to 1.0)
        float reflectionFactor = 0.5f; // How much sound reflects (0.0 to 1.0)
        
        // Thickness modeling
        float defaultThickness = 0.1f; // Default thickness for raycast hits (meters)
    };
    /**
     * @struct OcclusionResult
     * @brief Result of occlusion computation.
     */
    struct OcclusionResult {
        bool isOccluded = false;
        float occlusionStrength = 0.0f;      // 0.0 = clear, 1.0 = fully blocked
        float dampingStrength = 0.0f;        // High-frequency dampening (0.0 to 1.0)
        
        float lpfCutoff = 20000.0f;          // Low-pass filter cutoff (Hz)
        float hpfCutoff = 20.0f;             // High-pass filter cutoff (Hz)
        
        int occludingObstaclesCount = 0;     // Number of obstacles blocking path
        float distanceThroughMaterial = 0.0f; // Total distance through occlusive material (m)
    };
public:
    static AudioOcclusion& Get();
    /**
     * @brief Initialize the occlusion system.
     * @return true if successful
     */
    bool Initialize();
    /**
     * @brief Shutdown the occlusion system.
     */
    void Shutdown();
    /**
     * @brief Register a GameObject as an occlusive obstacle.
     * @param gameObject The object to register
     * @param material Material type for audio properties
     */
    void RegisterObstacle(GameObject* gameObject, MaterialType material = MaterialType::Brick);
    /**
     * @brief Unregister a GameObject as an occlusive obstacle.
     */
    void UnregisterObstacle(GameObject* gameObject);
    /**
     * @brief Update material for a registered obstacle.
     */
    void UpdateObstacleMaterial(GameObject* gameObject, MaterialType material);
    /**
     * @brief Compute occlusion between listener and audio source.
     * @param listenerPos Listener position
     * @param sourcePos Audio source position
     * @param sourceObject Optional: The AudioSource's parent GameObject (to exclude self-occlusion)
     * @return Occlusion result
     */
    OcclusionResult ComputeOcclusion(const Vec3& listenerPos, const Vec3& sourcePos,
                                     GameObject* sourceObject = nullptr);
    /**
     * @brief Compute occlusion and apply to an AudioSource.
     * @param audioSource The audio source to apply occlusion to
     * @param listenerPos Listener position
     * @param sourcePos Source position
     */
    void ApplyOcclusionToSource(AudioSource* audioSource, const Vec3& listenerPos, const Vec3& sourcePos);
    // ============== Material Properties ==============
    /**
     * @brief Get material properties for a type.
     */
    const MaterialProperties& GetMaterialProperties(MaterialType material) const;
    /**
     * @brief Set custom material properties.
     */
    void SetMaterialProperties(MaterialType material, const MaterialProperties& props);
    /**
     * @brief Set a custom material by name.
     */
    void SetCustomMaterial(const std::string& name, const MaterialProperties& props);
    // ============== Configuration ==============
    /**
     * @brief Enable/disable occlusion computation.
     */
    void SetEnabled(bool enabled) { m_enabled = enabled; }
    /**
     * @brief Set number of raycast samples between source and listener.
     * @param samples Number of rays to cast (higher = more accurate but slower)
     */
    void SetRaycastSampleCount(int samples);
    /**
     * @brief Set maximum distance for occlusion computation.
     * Occlusion ignored beyond this distance (optimization).
     */
    void SetMaxOcclusionDistance(float distance);
    /**
     * @brief Enable/disable per-frequency occlusion filtering.
     * When true, occlusion applies different attenuation per frequency band.
     */
    void SetAdvancedFiltering(bool enabled) { m_advancedFiltering = enabled; }
    /**
     * @brief Set low-pass filter curve parameters.
     * @param minFrequency Minimum cutoff frequency (Hz)
     * @param maxFrequency Maximum cutoff frequency (Hz)
     * @param occlusionScale How strongly occlusion affects LPF
     */
    void SetLPFParameters(float minFrequency, float maxFrequency, float occlusionScale);
    // ============== Debugging ==============
    /**
     * @brief Get the last computed occlusion result (for debugging).
     */
    const OcclusionResult& GetLastOcclusionResult() const { return m_lastResult; }
    /**
     * @brief Get count of registered obstacles.
     */
    int GetObstacleCount() const { return static_cast<int>(m_obstacles.size()); }
private:
    AudioOcclusion();
    ~AudioOcclusion();
    // Raycast helpers
    bool RaycastToSource(const Vec3& from, const Vec3& to, GameObject* excludeObject,
                        OcclusionResult& outResult);
    
    // Filter computation
    void ComputeFilters(float occlusionStrength, float dampingStrength,
                       float& outLPF, float& outHPF) const;
    bool m_initialized = false;
    bool m_enabled = true;
    // Material properties for each type
    std::unordered_map<int, MaterialProperties> m_materialProperties;
    std::unordered_map<std::string, MaterialProperties> m_customMaterials;
    // Registered obstacles: GameObject* -> MaterialType
    std::unordered_map<uintptr_t, std::pair<GameObject*, MaterialType>> m_obstacles;
    // Configuration
    int m_raycastSampleCount = 1;       // Single raycast by default
    float m_maxOcclusionDistance = 200.0f;
    bool m_advancedFiltering = false;
    // LPF Parameters
    float m_lpfMinFrequency = 500.0f;
    float m_lpfMaxFrequency = 20000.0f;
    float m_lpfOcclusionScale = 1.0f;
    // Cache for last computation
    OcclusionResult m_lastResult;
    AudioOcclusion() {}
    ~AudioOcclusion() {}
};
