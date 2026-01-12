#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

class PhysXCloth;
class AudioSource;

// Manages rustling sounds for cloth objects
class ClothAudioSystem {
public:
    static ClothAudioSystem& Get();

    bool Initialize();
    void Shutdown();
    void Update(float deltaTime);

    // Register a cloth for audio
    void RegisterCloth(PhysXCloth* cloth, const std::string& soundPath, float volumeScale = 1.0f);
    void UnregisterCloth(PhysXCloth* cloth);

    // Global settings
    void SetMinVelocity(float v) { m_MinVelocity = v; }
    void SetVelocityScale(float s) { m_VelocityScale = s; }

private:
    ClothAudioSystem() = default;
    ~ClothAudioSystem() = default;

    struct ClothAudioData {
        PhysXCloth* cloth;
        std::shared_ptr<AudioSource> source;
        float volumeScale;
        float smoothVelocity; // Smoothed velocity for audio stability
    };

    std::unordered_map<PhysXCloth*, ClothAudioData> m_ClothMap;
    
    float m_MinVelocity = 0.1f;
    float m_VelocityScale = 0.5f; // Scale factor for converting velocity to volume
    bool m_Initialized = false;
};
