#pragma once

#include <string>
#include "Math/Vec3.h"
#include <miniaudio.h>
#include <memory> 

class AudioSource {
public:
    AudioSource();
    ~AudioSource();

    void Load(const std::string& filepath);
    void Play();
    void Pause();
    void Stop();
    
    void SetLooping(bool loop);
    bool IsLooping() const;

    void SetVolume(float volume);
    float GetVolume() const;

    void SetPitch(float pitch);
    float GetPitch() const;

    // 3D Spatialization
    void SetPosition(const Vec3& position);
    void SetDirection(const Vec3& forward);
    void SetVelocity(const Vec3& velocity);
    
    // Attenuation
    enum class AttenuationModel {
        None,
        Inverse,
        Linear,
        Exponential
    };
    void SetAttenuationModel(AttenuationModel model);
    void SetRolloff(float rolloff);
    
    // MinDistance: Distance at which sound is at full volume
    // MaxDistance: Distance at which sound fades
    void SetMinDistance(float distance);
    void SetMaxDistance(float distance);
    
    // Directional Sound (Cones)
    // innerAngle: Inside this angle, volume is 1.0 (radians)
    // outerAngle: Outside this angle, volume is outerGain (radians)
    // outerGain: Volume multiplier outside the outer angle
    void SetCone(float innerAngleRad, float outerAngleRad, float outerGain);

    // Occlusion (0.0 = clear, 1.0 = fully occluded)
    void SetOcclusion(float strength);
    
    // Doppler
    void SetDopplerFactor(float factor);
    
    // Getters for UI
    Vec3 GetPosition() const { return m_Position; }
    float GetVolume() const { return m_Volume; }
    float GetPitch() const { return m_Pitch; }
    bool IsLooping() const { return m_Looping; }
    float GetMinDistance() const { return m_MinDistance; }
    float GetMaxDistance() const { return m_MaxDistance; }
    float GetRolloff() const { return m_Rolloff; }
    float GetDopplerFactor() const { return m_DopplerFactor; }

private:
    ma_sound m_sound;
    bool m_loaded = false;
    
    // Local cache for getters
    Vec3 m_Position;
    float m_Volume = 1.0f;
    float m_Pitch = 1.0f;
    bool m_Looping = false;
    float m_MinDistance = 1.0f;
    float m_MaxDistance = 100.0f;
    float m_Rolloff = 1.0f;
    float m_DopplerFactor = 1.0f;
    float m_Occlusion = 0.0f;
};
