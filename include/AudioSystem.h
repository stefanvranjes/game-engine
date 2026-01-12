#pragma once

#include <miniaudio.h>
#include "extras/nodes/ma_reverb_node/ma_reverb_node.h"
#include "Math/Vec3.h"
#include <vector>
#include <string>
#include <memory>

class AudioSystem {
public:
    static AudioSystem& Get();

    bool Initialize();
    void Shutdown();

    // Returns the raw miniaudio engine.
    ma_engine* GetEngine();

    // Set master volume (0.0 to 1.0+)
    void SetMasterVolume(float volume);
    
    // Set listener properties for 3D audio
    void SetListenerPosition(const Vec3& position);
    void SetListenerDirection(const Vec3& forward);
    void SetListenerVelocity(const Vec3& velocity);
    void UpdateListener(const Vec3& pos, const Vec3& forward, const Vec3& up, const Vec3& velocity);
    
    Vec3 GetListenerPosition() const;
    Vec3 GetListenerForward() const;
    Vec3 GetListenerUp() const;
    Vec3 GetListenerVelocity() const;

    // Active Listener Management
    void SetActiveListener(class AudioListener* listener);
    class AudioListener* GetActiveListener() const { return m_ActiveListener; }

    // Reverb Management
    struct ReverbProperties {
        float roomSize;   // 0.0 to 1.0 (Small to Large)
        float damping;    // 0.0 to 1.0 (Bright to Dull)
        float wetVolume;  // 0.0 to 1.0 (Reverb volume)
        float dryVolume;  // 0.0 to 1.0 (Original volume)
    };
    
    void SetReverbProperties(const ReverbProperties& props);
    // Accessor for AudioSource to attach to
    ma_sound_group* GetWorldGroup() { return &m_worldGroup; }
    ma_sound_group* GetMusicGroup() { return &m_musicGroup; }

    void SetSFXVolume(float volume);
    void SetMusicVolume(float volume);

    // Update audio system (called from main loop)
    void Update(float deltaTime);

private:
    AudioSystem();
    ~AudioSystem();

    ma_engine m_engine;
    ma_sound_group m_worldGroup; // All 3D sounds go here
    ma_sound_group m_musicGroup; // 2D Music goes here (no reverb usually)
    ma_reverb_node m_reverbNode; // The effect node
    
    bool m_initialized = false;
    class AudioListener* m_ActiveListener = nullptr;
    
    // Cache
    Vec3 m_ListenerPos;
    Vec3 m_ListenerForward;
    Vec3 m_ListenerUp;
    Vec3 m_ListenerVelocity;
};
