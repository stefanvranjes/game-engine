#include "AudioSpatializer.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265359f
#endif

AudioSpatializer& AudioSpatializer::Get() {
    static AudioSpatializer instance;
    return instance;
}

bool AudioSpatializer::Initialize() {
    m_initialized = true;
    return true;
}

void AudioSpatializer::Shutdown() {
    m_initialized = false;
}

AudioSpatializer::SpatializationOutput AudioSpatializer::ComputeSpatialization(const SpatializationParams& params) {
    SpatializationOutput out;
    
    // 1. Distance & Attenuation
    Vec3 toSource = params.sourcePos - params.listenerPos;
    float dist = toSource.Length();
    out.distance = dist;
    
    if (dist < 0.001f) {
        toSource = params.listenerForward; // Fallback
        dist = 0.001f;
    }
    
    Vec3 dirToSource = toSource / dist;
    
    out.volumeAttenuation = GetDistanceAttenuation(dist, params.minDistance, params.maxDistance, params.rolloff);
    
    // 2. Directional Cone (Source Directivity)
    // If source has direction
    if (params.sourceDirection.LengthSquared() > 0.5f) { // Assuming normalized
        // dirToListener is -dirToSource
        out.coneFalloff = ComputeConeAttenuation(params.sourceDirection, -dirToSource, 
                                                params.coneInnerAngle, params.coneOuterAngle, params.coneOuterGain);
    } else {
        out.coneFalloff = 1.0f;
    }
    
    out.effectiveVolume = out.volumeAttenuation * out.coneFalloff;
    
    // 3. Azimuth & Elevation (Listener Space)
    // Transform direction to listener local space
    Vec3 right = params.listenerForward.Cross(params.listenerUp).Normalized();
    Vec3 up = right.Cross(params.listenerForward).Normalized();
    
    // Project dirToSource onto Forward, Right, Up
    float fwdDot = dirToSource.Dot(params.listenerForward);
    float rightDot = dirToSource.Dot(right);
    float upDot = dirToSource.Dot(up);
    
    // Cartesian to Spherical in local space
    // Azimuth: Angle in horizontal plane (Right/Forward)
    // atan2(y, x) -> atan2(Right, Forward) -> 0 is Forward, PI/2 is Right
    out.azimuth = std::atan2(rightDot, fwdDot);
    
    // Elevation: Angle from horizontal plane
    out.elevation = std::asin(std::max(-1.0f, std::min(1.0f, upDot)));
    
    // 4. Panning
    ComputePanning(out.azimuth, out.elevation, out.leftPan, out.centerGain, out.surroundGain);
    
    // 5. Doppler
    out.dopplerPitch = ComputeDopplerPitch(params.sourceVelocity, Vec3(0,0,0) /*Listener Vel TODO*/, dirToSource, params.dopplerFactor, params.speedOfSound);
    
    // 6. Filtering (Occlusion passed in params + specific processing)
    // Map occlusion (0-1) to filters
    // Simple LPF mapping
    if (params.occlusionStrength > 0.0f) {
        float occ = std::fmin(1.0f, params.occlusionStrength);
        float minFreq = 500.0f;
        float maxFreq = 22000.0f;
        // Logarithmic scale
        out.lpfCutoff = std::exp(std::log(maxFreq) + occ * (std::log(minFreq) - std::log(maxFreq)));
    }
    
    // HRTF filtering if enabled
    if (m_hrtfEnabled) {
        ApplyHRTFFiltering(out, out.elevation);
    }
    
    return out;
}

float AudioSpatializer::GetDistanceAttenuation(float distance, float minDist, float maxDist, float rolloff) const {
    if (m_distanceModel == DistanceModel::None) return 1.0f;
    
    // Min distance clamp
    float d = std::max(distance, minDist);
    
    if (m_distanceModel == DistanceModel::Linear) {
        if (d >= maxDist) return 0.0f;
        return 1.0f - ((d - minDist) / (maxDist - minDist));
    }
    
    if (m_distanceModel == DistanceModel::Inverse) {
        // Gain = Min / (Min + Rolloff * (Dist - Min))
        return minDist / (minDist + rolloff * (d - minDist));
    }
    
    if (m_distanceModel == DistanceModel::InverseClamped) {
        if (d >= maxDist) return 0.0f;
        return minDist / (minDist + rolloff * (d - minDist));
    }
    
    return 1.0f;
}

float AudioSpatializer::ComputeConeAttenuation(const Vec3& forwardDir, const Vec3& toListenerDir,
                            float innerAngle, float outerAngle, float outerGain) const {
    float dot = forwardDir.Dot(toListenerDir);
    float angle = std::acos(std::max(-1.0f, std::min(1.0f, dot))); // 0 to PI
    
    // Angles in headers are usually total width or half width? 
    // Usually total angle is passed (e.g. 90 deg cone).
    // So we compare angle to half-angles.
    float halfInner = innerAngle * 0.5f;
    float halfOuter = outerAngle * 0.5f;
    
    if (angle <= halfInner) return 1.0f;
    if (angle >= halfOuter) return outerGain;
    
    // Linear interp between inner and outer
    float t = (angle - halfInner) / (halfOuter - halfInner);
    return 1.0f + t * (outerGain - 1.0f);
}

float AudioSpatializer::ComputeDopplerPitch(const Vec3& sourceVelocity, const Vec3& listenerVelocity,
                         const Vec3& toSourceDir, float dopplerFactor, float speedOfSound) const {
    if (dopplerFactor <= 0.001f) return 1.0f;
    
    // Relative speed along the line of sight
    // v_listener projected onto direction to source
    float vls = listenerVelocity.Dot(toSourceDir);
    // v_source projected onto direction to source
    float vss = sourceVelocity.Dot(toSourceDir);
    
    // f_observed = f_source * (c + v_l) / (c + v_s)
    // Note signs depending on definition of dir (Listener->Source)
    // If approaching, pitch increases.
    
    // Simplified: pitch = 1.0 + (RelSpeed / SpeedOfSound) * Factor
    // Exact:
    // Relative velocity approaching = negative?
    // Let's use robust formula
    
    float relSpeed = vls - vss; // Positive if listener moving towards source faster than source moving away
    
    return std::max(0.1f, 1.0f + (relSpeed / speedOfSound) * dopplerFactor);
}

void AudioSpatializer::ComputePanning(float azimuth, float elevation, float& leftPan,
                   float& centerGain, float& surroundGain) const {
    // Simple Equal Power Panning Law
    // Azimuth: 0 = center (front), PI/2 = right, -PI/2 = left
    
    // Map azimuth to 0..1 (Left -> Right)
    // -PI/2 -> 0, PI/2 -> 1, 0 -> 0.5
    
    // Clamp azimuth for stereo field -PI/2 to PI/2
    // Anything behind is also panned but maybe muffled/surround
    
    // Basic stereo pan
    float pan = 0.5f * (std::sin(azimuth) + 1.0f);
    pan = std::max(0.0f, std::min(1.0f, pan));
    
    leftPan = 1.0f - pan;
    
    // Center gain drops as we move to sides
    centerGain = std::cos(azimuth);
    if (centerGain < 0) centerGain = 0; // Only front
    
    // Surround gain increases as we move behind
    // Behind is when abs(azimuth) > PI/2
    if (std::abs(azimuth) > M_PI/2.0f) {
         surroundGain = (std::abs(azimuth) - M_PI/2.0f) / (M_PI/2.0f);
    } else {
        surroundGain = 0.0f;
    }
}

void AudioSpatializer::ApplyHRTFFiltering(const SpatializationOutput& output, float elevation) {
    // Stub: Modify filters based on elevation
    // e.g. High elevation -> notches at 8kHz
}
