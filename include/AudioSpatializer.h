#pragma once
#include "Math/Vec3.h"
#include "Math/Quaternion.h"
#include <vector>
#include <memory>
class GameObject;
/**
 * @class AudioSpatializer
 * @brief Advanced 3D audio spatialization with HRTF simulation and distance modeling.
 * 
 * Provides sophisticated spatial audio processing including:
 * - Head-Related Transfer Function (HRTF) simulation for elevation cues
 * - Advanced distance attenuation curves
 * - Directional cone effects with smooth falloff
 * - Doppler effect with velocity prediction
 * - Audio occlusion with frequency filtering
 */
// Stub header to allow AudioSystem to compile without full spatial audio implementation
class AudioSpatializer {
public:
    /**
     * @enum HRTFProfile
     * @brief HRTF profile selection (different head sizes/shapes).
     */
    enum class HRTFProfile {
        Generic,    // Generic HRTF
        Large,      // Larger head (deeper ears)
        Small,      // Smaller head (shallower ears)
        Custom      // Custom profile parameters
    };
    /**
     * @enum DistanceModel
     * @brief Distance attenuation curve model.
     */
    enum class DistanceModel {
        None,           // No distance attenuation
        Inverse,        // 1/distance (realistic outdoor sound)
        InverseClamped, // 1/distance with clamping between min/max
        Linear,         // Linear fade from min to max distance
        Exponential,    // Exponential decay (sharper falloff)
        Custom          // Custom curve function
    };
    /**
     * @struct SpatializationParams
     * @brief Parameters for spatial audio rendering.
     */
    struct SpatializationParams {
        Vec3 listenerPos;
        Vec3 listenerForward;
        Vec3 listenerUp;
        Vec3 sourcePos;
        Vec3 sourceVelocity;
        Vec3 sourceDirection;
        
        float minDistance = 1.0f;
        float maxDistance = 100.0f;
        float rolloff = 1.0f;
        
        float coneInnerAngle = 6.28f;  // Full sphere (radians)
        float coneOuterAngle = 6.28f;
        float coneOuterGain = 0.0f;
        
        float dopplerFactor = 1.0f;
        float speedOfSound = 343.0f;   // m/s (20°C air)
        
        DistanceModel distanceModel = DistanceModel::InverseClamped;
        float occlusionStrength = 0.0f; // 0.0 = clear, 1.0 = fully muffled
    };
    /**
     * @struct SpatializationOutput
     * @brief Computed spatial audio parameters.
     */
    struct SpatializationOutput {
        float distance = 0.0f;
        float azimuth = 0.0f;           // Horizontal angle (-π to π)
        float elevation = 0.0f;         // Vertical angle (-π/2 to π/2)
        
        float volumeAttenuation = 1.0f; // Distance-based volume reduction
        float coneFalloff = 1.0f;       // Cone-based volume reduction
        float effectiveVolume = 1.0f;   // volumeAttenuation * coneFalloff
        
        float leftPan = 0.5f;           // 0.0 = left, 1.0 = right
        float centerGain = 1.0f;        // Center channel level
        float surroundGain = 0.0f;      // Surround/ambience level
        
        float dopplerPitch = 1.0f;      // Pitch shift from Doppler effect
        
        float lpfCutoff = 20000.0f;     // Low-pass filter (occlusion)
        float hpfCutoff = 20.0f;        // High-pass filter (distance haze)
    };
public:
    static AudioSpatializer& Get();
    /**
     * @brief Initialize the spatializer.
     */
    bool Initialize();
    /**
     * @brief Shutdown the spatializer.
     */
    void Shutdown();
    /**
     * @brief Compute spatial audio parameters for a source.
     * @param params Input spatialization parameters
     * @return Computed output parameters
     */
    SpatializationOutput ComputeSpatialization(const SpatializationParams& params);
    // ============== HRTF Configuration ==============
    /**
     * @brief Set HRTF profile for elevation cue processing.
     * @param profile The HRTF profile to use
     */
    void SetHRTFProfile(HRTFProfile profile);
    /**
     * @brief Enable/disable HRTF processing.
     */
    void SetHRTFEnabled(bool enabled) { m_hrtfEnabled = enabled; }
    /**
     * @brief Get current HRTF profile.
     */
    HRTFProfile GetHRTFProfile() const { return m_hrtfProfile; }
    // ============== Distance Attenuation ==============
    /**
     * @brief Set distance attenuation model.
     * @param model The attenuation model to use
     */
    void SetDistanceModel(DistanceModel model);
    /**
     * @brief Get the distance attenuation for a given distance.
     * @param distance Distance from source to listener
     * @param minDist Minimum distance (full volume)
     * @param maxDist Maximum distance (min volume)
     * @param rolloff Rolloff factor
     * @return Volume attenuation (0.0 to 1.0)
     */
    float GetDistanceAttenuation(float distance, float minDist, float maxDist, float rolloff) const;
    // ============== Directional Audio ==============
    /**
     * @brief Compute cone attenuation based on directional falloff.
     * @param forwardDir Source forward direction
     * @param toListenerDir Normalized direction from source to listener
     * @param innerAngle Inner cone angle (full volume)
     * @param outerAngle Outer cone angle (reduced volume)
     * @param outerGain Volume at outer angle
     * @return Attenuation factor (0.0 to 1.0)
     */
    float ComputeConeAttenuation(const Vec3& forwardDir, const Vec3& toListenerDir,
                                float innerAngle, float outerAngle, float outerGain) const;
    // ============== Doppler Effect ==============
    /**
     * @brief Compute Doppler pitch shift.
     * @param sourceVelocity Source velocity
     * @param listenerVelocity Listener velocity
     * @param toSourceDir Normalized direction from listener to source
     * @param dopplerFactor Doppler effect intensity (0.0 = disabled)
     * @param speedOfSound Speed of sound in current medium (m/s)
     * @return Pitch shift factor (< 1.0 = lower, > 1.0 = higher)
     */
    float ComputeDopplerPitch(const Vec3& sourceVelocity, const Vec3& listenerVelocity,
                             const Vec3& toSourceDir, float dopplerFactor,
                             float speedOfSound = 343.0f) const;
    // ============== Occlusion & Filtering ==============
    /**
     * @brief Set audio occlusion for a source.
     * @param occlusionStrength 0.0 = clear, 1.0 = fully muffled
     */
    void SetOcclusion(float occlusionStrength);
    /**
     * @brief Get filter parameters for a given occlusion level.
     * @param occlusionStrength Occlusion amount (0.0 to 1.0)
     * @param lpfCutout Output low-pass filter cutoff (Hz)
     * @param hpfCutoff Output high-pass filter cutoff (Hz)
     */
    void GetOcclusionFilters(float occlusionStrength, float& lpfCutout, float& hpfCutoff) const;
    // ============== Spatial Panning ==============
    /**
     * @brief Compute stereo/surround panning from source position.
     * @param azimuth Horizontal angle in radians (-π = left, π = right)
     * @param elevation Vertical angle in radians (-π/2 = bottom, π/2 = top)
     * @param leftPan Output left channel pan (0.0 to 1.0)
     * @param centerGain Output center channel gain
     * @param surroundGain Output surround/height channel gain
     */
    void ComputePanning(float azimuth, float elevation, float& leftPan,
                       float& centerGain, float& surroundGain) const;
    // ============== Utilities ==============
    /**
     * @brief Convert Cartesian coordinates to spherical.
     * @param direction Direction vector
     * @param outAzimuth Output azimuth angle (-π to π)
     * @param outElevation Output elevation angle (-π/2 to π/2)
     */
    static void CartesianToSpherical(const Vec3& direction, float& outAzimuth, float& outElevation);
    /**
     * @brief Convert spherical coordinates to Cartesian.
     * @param azimuth Azimuth angle
     * @param elevation Elevation angle
     * @return Direction vector
     */
    static Vec3 SphericalToCartesian(float azimuth, float elevation);
    /**
     * @brief Normalize angle to range [-π, π].
     */
    static float NormalizeAngle(float angle);
private:
    AudioSpatializer();
    ~AudioSpatializer();
    // HRTF Helper: Compute elevation-dependent filtering
    void ApplyHRTFFiltering(const SpatializationOutput& output, float elevation);
    bool m_initialized = false;
    HRTFProfile m_hrtfProfile = HRTFProfile::Generic;
    bool m_hrtfEnabled = true;
    DistanceModel m_distanceModel = DistanceModel::InverseClamped;
    
    float m_currentOcclusion = 0.0f;
    // HRTF Parameters by profile (ITD, ILD, etc.)
    struct HRTFData {
        float itdScale = 1.0f;  // Inter-temporal delay scale
        float ildScale = 1.0f;  // Inter-level difference scale
    };
    HRTFData m_hrtfData;
    AudioSpatializer() {}
    ~AudioSpatializer() {}
};