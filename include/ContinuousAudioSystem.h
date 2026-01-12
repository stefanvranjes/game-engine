#pragma once

#include "AudioSurface.h"
#include "Math/Vec3.h"
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>

class GameObject;
class IPhysicsRigidBody;
class AudioSource;

// Manages rolling and sliding sounds
class ContinuousAudioSystem {
public:
    static ContinuousAudioSystem& Get();

    bool Initialize();
    void Shutdown();

    // Configuration
    void SetRollSound(AudioSurfaceType surface, const std::string& soundPath, float volumeScale = 1.0f, float pitchScale = 1.0f);
    void SetSlideSound(AudioSurfaceType surface, const std::string& soundPath, float volumeScale = 1.0f, float pitchScale = 1.0f);
    
    // Runtime
    // Note: Depends on PhysXBackend event types
    void OnCollision(IPhysicsRigidBody* bodyA, IPhysicsRigidBody* bodyB, const Vec3& point, const Vec3& normal, float impulse, int eventType);

private:
    ContinuousAudioSystem() = default;
    ~ContinuousAudioSystem() = default;

    struct SoundDefinition {
        std::string filepath;
        float volumeScale = 1.0f;
        float pitchScale = 1.0f;
    };

    struct ActiveSound {
        std::string id; // Unique ID for debug
        std::shared_ptr<AudioSource> source;
        float currentVolume = 0.0f;
        float targetVolume = 0.0f;
        float targetPitch = 1.0f;
    };
    
    // Key: Pair of bodies (min ptr, max ptr)
    struct BodyPairHash {
        std::size_t operator()(const std::pair<IPhysicsRigidBody*, IPhysicsRigidBody*>& p) const {
            return reinterpret_cast<size_t>(p.first) ^ reinterpret_cast<size_t>(p.second);
        }
    };

    std::unordered_map<std::pair<IPhysicsRigidBody*, IPhysicsRigidBody*>, ActiveSound, BodyPairHash> m_ActiveSounds;
    std::unordered_map<int, SoundDefinition> m_RollSounds; // Key: SurfaceType
    std::unordered_map<int, SoundDefinition> m_SlideSounds; // Key: SurfaceType

    void UpdateSound(ActiveSound& sound, const Vec3& pos, float speed, float maxSpeed = 10.0f);
    void StopSound(std::pair<IPhysicsRigidBody*, IPhysicsRigidBody*> key);

    bool m_Initialized = false;
};
