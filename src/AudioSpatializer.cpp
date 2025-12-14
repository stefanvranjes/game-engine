#include "AudioSpatializer.h"
#include <cmath>
#include <algorithm>
#include <iostream>

AudioSpatializer& AudioSpatializer::Get() {
    static AudioSpatializer instance;
    return instance;
}

AudioSpatializer::AudioSpatializer() {}

AudioSpatializer::~AudioSpatializer() {
    Shutdown();
}

bool AudioSpatializer::Initialize() {
    if (m_initialized) return true;

    m_initialized = true;
    return true;
}

void AudioSpatializer::Shutdown() {
    m_initialized = false;
}

SpatializationOutput AudioSpatializer::ComputeSpatialization(const SpatializationParams& params) {
    SpatializationOutput output;

    // Compute distance
    Vec3 toSource = params.sourcePos - params.listenerPos;
    output.distance = toSource.Length();

    if (output.distance < 0.001f) {
        // Source at listener position
        output.effectiveVolume = 1.0f;
        output.leftPan = 0.5f;
        output.dopplerPitch = 1.0f;
        return output;
    }

    Vec3 toSourceNorm = toSource.Normalized();

    // Compute azimuth and elevation
    CartesianToSpherical(toSourceNorm, output.azimuth, output.elevation);

    // Distance attenuation
    output.volumeAttenuation = GetDistanceAttenuation(output.distance, params.minDistance,
                                                       params.maxDistance, params.rolloff);

    // Cone attenuation
    output.coneFalloff = ComputeConeAttenuation(params.sourceDirection, toSourceNorm,
                                                params.coneInnerAngle, params.coneOuterAngle,
                                                params.coneOuterGain);

    // Combined volume
    output.effectiveVolume = output.volumeAttenuation * output.coneFalloff;

    // Doppler effect
    output.dopplerPitch = ComputeDopplerPitch(params.sourceVelocity, Vec3(0, 0, 0), // Listener at origin
                                             toSourceNorm, params.dopplerFactor, params.speedOfSound);

    // Panning
    ComputePanning(output.azimuth, output.elevation, output.leftPan, output.centerGain, output.surroundGain);

    // Occlusion filtering
    GetOcclusionFilters(params.occlusionStrength, output.lpfCutoff, output.hpfCutoff);

    // HRTF processing
    if (m_hrtfEnabled) {
        ApplyHRTFFiltering(output, output.elevation);
    }

    return output;
}

void AudioSpatializer::SetHRTFProfile(HRTFProfile profile) {
    m_hrtfProfile = profile;

    // Configure HRTF parameters based on profile
    switch (profile) {
    case HRTFProfile::Generic:
        m_hrtfData.itdScale = 1.0f;
        m_hrtfData.ildScale = 1.0f;
        break;
    case HRTFProfile::Large:
        m_hrtfData.itdScale = 1.2f;  // Larger head = longer ITD
        m_hrtfData.ildScale = 1.1f;
        break;
    case HRTFProfile::Small:
        m_hrtfData.itdScale = 0.8f;
        m_hrtfData.ildScale = 0.9f;
        break;
    case HRTFProfile::Custom:
        // Configured externally
        break;
    }
}

void AudioSpatializer::SetDistanceModel(DistanceModel model) {
    m_distanceModel = model;
}

float AudioSpatializer::GetDistanceAttenuation(float distance, float minDist, float maxDist, float rolloff) const {
    if (distance <= minDist) {
        return 1.0f;
    }

    if (distance >= maxDist) {
        return 0.0f;
    }

    switch (m_distanceModel) {
    case DistanceModel::None:
        return 1.0f;

    case DistanceModel::Inverse:
        // 1/distance model
        if (distance > 0.0f) {
            return minDist / (minDist + rolloff * (distance - minDist));
        }
        return 1.0f;

    case DistanceModel::InverseClamped:
        // Inverse with clamping
        if (distance > 0.0f) {
            float att = minDist / (minDist + rolloff * (distance - minDist));
            return std::max(0.0f, std::min(1.0f, att));
        }
        return 1.0f;

    case DistanceModel::Linear:
        // Linear fade
        {
            float range = maxDist - minDist;
            if (range > 0.0f) {
                return 1.0f - ((distance - minDist) / range);
            }
        }
        return 0.0f;

    case DistanceModel::Exponential:
        // Exponential decay
        {
            float range = maxDist - minDist;
            if (range > 0.0f) {
                float normalized = (distance - minDist) / range;
                return std::exp(-rolloff * normalized * normalized);
            }
        }
        return 0.0f;

    case DistanceModel::Custom:
        // Would be implemented by user
        return 1.0f;
    }

    return 1.0f;
}

float AudioSpatializer::ComputeConeAttenuation(const Vec3& forwardDir, const Vec3& toListenerDir,
                                              float innerAngle, float outerAngle, float outerGain) const {
    Vec3 normalizedForward = forwardDir.Normalized();

    // Angle between source direction and listener direction
    float cosAngle = normalizedForward.Dot(toListenerDir);
    float angle = std::acos(std::max(-1.0f, std::min(1.0f, cosAngle)));

    // Full sphere (no cone)
    if (innerAngle >= 6.28f && outerAngle >= 6.28f) {
        return 1.0f;
    }

    if (angle <= innerAngle) {
        return 1.0f;
    }

    if (angle >= outerAngle) {
        return outerGain;
    }

    // Smooth interpolation between inner and outer
    float range = outerAngle - innerAngle;
    if (range > 0.0f) {
        float factor = (angle - innerAngle) / range;
        return 1.0f + (outerGain - 1.0f) * factor;
    }

    return 1.0f;
}

float AudioSpatializer::ComputeDopplerPitch(const Vec3& sourceVelocity, const Vec3& listenerVelocity,
                                           const Vec3& toSourceDir, float dopplerFactor,
                                           float speedOfSound) const {
    if (dopplerFactor < 0.001f) {
        return 1.0f;
    }

    // Doppler formula: f' = f * (v_sound + v_listener) / (v_sound + v_source)
    // Simplified for 1D (radial velocity only)
    float sourceRadialVel = -sourceVelocity.Dot(toSourceDir);
    float listenerRadialVel = listenerVelocity.Dot(toSourceDir);

    float dopplerShift = (speedOfSound + dopplerFactor * listenerRadialVel) /
                        (speedOfSound + dopplerFactor * sourceRadialVel);

    return std::max(0.5f, std::min(2.0f, dopplerShift)); // Clamp to reasonable range
}

void AudioSpatializer::GetOcclusionFilters(float occlusionStrength, float& lpfCutout, float& hpfCutoff) const {
    // Occlusion: reduce high frequencies (LPF) and boost lows slightly (HPS reduction)
    occlusionStrength = std::max(0.0f, std::min(1.0f, occlusionStrength));

    // LPF: Start at 20kHz (no filter) down to ~500Hz at full occlusion
    const float minLPF = 500.0f;
    const float maxLPF = 20000.0f;
    lpfCutout = maxLPF - (maxLPF - minLPF) * occlusionStrength;

    // HPF: Slight boost (less cutting) of low end
    const float minHPF = 20.0f;
    const float maxHPF = 200.0f;
    hpfCutoff = minHPF + (maxHPF - minHPF) * occlusionStrength;
}

void AudioSpatializer::ComputePanning(float azimuth, float elevation, float& leftPan,
                                     float& centerGain, float& surroundGain) const {
    // Normalize azimuth to [-π, π]
    azimuth = NormalizeAngle(azimuth);

    // Stereo panning based on azimuth
    // azimuth = -π (left) to π (right)
    // leftPan = 0.0 (right) to 1.0 (left) to 0.5 (center)
    leftPan = 0.5f + (azimuth / (3.14159f * 2.0f));
    leftPan = std::max(0.0f, std::min(1.0f, leftPan));

    // Center channel: boost when source is ahead
    // elevation = -π/2 (bottom) to π/2 (top)
    float frontback = std::cos(azimuth); // 1.0 = front, -1.0 = back, 0 = side
    centerGain = 0.5f + 0.5f * frontback;

    // Surround/height channel: boost for elevated sources
    surroundGain = std::abs(elevation) / (3.14159f * 0.5f);
    surroundGain = std::min(1.0f, surroundGain * 0.5f); // Moderate surround level
}

void AudioSpatializer::CartesianToSpherical(const Vec3& direction, float& outAzimuth, float& outElevation) {
    outAzimuth = std::atan2(direction.x, direction.z); // Azimuth: angle in XZ plane
    
    float horizontalLength = std::sqrt(direction.x * direction.x + direction.z * direction.z);
    outElevation = std::atan2(direction.y, horizontalLength);
}

Vec3 AudioSpatializer::SphericalToCartesian(float azimuth, float elevation) {
    float cosElevation = std::cos(elevation);
    return Vec3(
        std::sin(azimuth) * cosElevation,
        std::sin(elevation),
        std::cos(azimuth) * cosElevation
    );
}

float AudioSpatializer::NormalizeAngle(float angle) {
    const float PI = 3.14159265359f;
    const float TWO_PI = 2.0f * PI;

    while (angle > PI) {
        angle -= TWO_PI;
    }
    while (angle < -PI) {
        angle += TWO_PI;
    }
    return angle;
}

void AudioSpatializer::SetOcclusion(float occlusionStrength) {
    m_currentOcclusion = std::max(0.0f, std::min(1.0f, occlusionStrength));
}

void AudioSpatializer::ApplyHRTFFiltering(const SpatializationOutput& output, float elevation) {
    // HRTF: Apply subtle filtering based on elevation
    // Higher elevations -> slightly boost upper mids
    // This simulates the spectral cues for elevation perception

    // In a full implementation, we'd apply complex IIR filters
    // For now, we compute adjustment factors that can be applied to EQ
    
    float elevationNorm = elevation / (3.14159f * 0.5f); // [-1, 1]
    
    // ITD (Inter-temporal delay): Not directly applicable here
    // ILD (Inter-level difference): Already handled by panning
    
    // Spectral modification: Boost around 2-4kHz for overhead sources
    if (elevation > 0.5f) {
        // Upper elevation - subtle boost
    }
}
