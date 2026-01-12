#pragma once

#include "AudioSurface.h"
#include "Math/Vec3.h"
#include <unordered_map>
#include <string>
#include <memory>

class GameObject;
class IPhysicsRigidBody;

class ImpactAudioSystem {
public:
    static ImpactAudioSystem& Get();

    bool Initialize();
    void Shutdown();

    // Configuration
    void RegisterObject(GameObject* obj, AudioSurfaceType type);
    void UnregisterObject(GameObject* obj);
    
    // Sound Mapping
    // Key is combined material types (min, max) to be order independent
    // Value is list of sound paths to randomize? For now single path.
    void SetImpactSound(AudioSurfaceType surfaceA, AudioSurfaceType surfaceB, const std::string& soundPath, float volumeScale = 1.0f);
    
    // Runtime
    void OnCollision(IPhysicsRigidBody* bodyA, IPhysicsRigidBody* bodyB, const Vec3& point, const Vec3& normal, float impulse);

    void SetMinImpulseThreshold(float threshold) { m_MinImpulse = threshold; }
    void SetMasterVolume(float volume) { m_MasterVolume = volume; }

private:
    ImpactAudioSystem() = default;
    ~ImpactAudioSystem() = default;

    std::unordered_map<uintptr_t, AudioSurfaceType> m_ObjectSurfaces;

    struct ImpactSoundData {
        std::string filepath;
        float volumeScale = 1.0f;
    };
    
    // Hash for pair of enums
    struct SurfacePairHash {
        std::size_t operator()(const std::pair<AudioSurfaceType, AudioSurfaceType>& p) const {
            return static_cast<size_t>(p.first) ^ (static_cast<size_t>(p.second) << 16);
        }
    };

    std::unordered_map<std::pair<AudioSurfaceType, AudioSurfaceType>, ImpactSoundData, SurfacePairHash> m_ImpactMap;

    void PlaySound(const std::string& path, const Vec3& position, float volume);

    float m_MinImpulse = 1.0f;
    float m_MasterVolume = 1.0f;
    bool m_Initialized = false;
};
