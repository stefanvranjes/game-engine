#include "ImpactAudioSystem.h"
#include "PhysicsSystem.h"
#include "PhysXBackend.h"
#include "IPhysicsRigidBody.h"
#include "AudioSource.h"
#include "AudioSystem.h"
#include "GameObject.h"
#include <iostream>
#include <algorithm>

ImpactAudioSystem& ImpactAudioSystem::Get() {
    static ImpactAudioSystem instance;
    return instance;
}

bool ImpactAudioSystem::Initialize() {
    if (m_Initialized) return true;

    // Hook into Physics Backend
    // IPhysicsBackend* backend = PhysicsSystem::Get().GetBackend(); // Error: PhysicsSystem has no GetBackend
    
    // Get backend from Application
    #ifdef USE_PHYSX
    #include "Application.h"
    extern Application* g_Application; // Or use static Get if available
    // Assuming Application::Get() exists as added in Application.h
    PhysXBackend* physxBackend = Application::Get().GetPhysXBackend();
    
    if (physxBackend) {
        physxBackend->SetGlobalCollisionCallback([this](IPhysicsRigidBody* b1, IPhysicsRigidBody* b2, const Vec3& p, const Vec3& n, float i) {
            this->OnCollision(b1, b2, p, n, i);
        });
    }
    #endif

    // Setup Default Sounds (Placeholders)
    // Map(Generic, Generic) -> "impact_generic"
    
    m_Initialized = true;
    return true;
}

void ImpactAudioSystem::Shutdown() {
    m_ObjectSurfaces.clear();
    m_ImpactMap.clear();
    m_Initialized = false;
}

void ImpactAudioSystem::RegisterObject(GameObject* obj, AudioSurfaceType type) {
    if (!obj) return;
    m_ObjectSurfaces[reinterpret_cast<uintptr_t>(obj)] = type;
}

void ImpactAudioSystem::UnregisterObject(GameObject* obj) {
    if (!obj) return;
    m_ObjectSurfaces.erase(reinterpret_cast<uintptr_t>(obj));
}

void ImpactAudioSystem::SetImpactSound(AudioSurfaceType surfaceA, AudioSurfaceType surfaceB, const std::string& soundPath, float volumeScale) {
    // Order independent: Store min first
    if (surfaceA > surfaceB) std::swap(surfaceA, surfaceB);
    m_ImpactMap[{surfaceA, surfaceB}] = {soundPath, volumeScale};
}

void ImpactAudioSystem::OnCollision(IPhysicsRigidBody* bodyA, IPhysicsRigidBody* bodyB, const Vec3& point, const Vec3& normal, float impulse) {
    if (impulse < m_MinImpulse) return;

    // Get GameObjects
    GameObject* objA = static_cast<GameObject*>(bodyA->GetUserData());
    GameObject* objB = static_cast<GameObject*>(bodyB->GetUserData());
    
    if (!objA && !objB) return;

    // Determine Surface Types
    AudioSurfaceType typeA = AudioSurfaceType::Generic;
    AudioSurfaceType typeB = AudioSurfaceType::Generic;

    if (objA) {
        auto it = m_ObjectSurfaces.find(reinterpret_cast<uintptr_t>(objA));
        if (it != m_ObjectSurfaces.end()) typeA = it->second;
    }
    
    if (objB) {
        auto it = m_ObjectSurfaces.find(reinterpret_cast<uintptr_t>(objB));
        if (it != m_ObjectSurfaces.end()) typeB = it->second;
    }

    // Lookup Sound
    if (typeA > typeB) std::swap(typeA, typeB);
    
    auto it = m_ImpactMap.find({typeA, typeB});
    if (it == m_ImpactMap.end()) {
        // Fallback: Try Generic + Specific
        it = m_ImpactMap.find({AudioSurfaceType::Generic, typeB}); // typeB is the larger one, so generic+specific
        if (it == m_ImpactMap.end()) {
             it = m_ImpactMap.find({AudioSurfaceType::Generic, AudioSurfaceType::Generic});
        }
    }

    if (it != m_ImpactMap.end()) {
        const auto& data = it->second;
        
        // Calculate volume based on impulse
        // Impulse range? 1.0 to 100.0?
        // Map linearly or log?
        // simple map: volume = clamp(impulse / 10.0f, 0.1f, 1.0f) * scale
        
        float intensity = std::min(1.0f, std::max(0.1f, impulse / 20.0f));
        float volume = intensity * data.volumeScale * m_MasterVolume;
        
        PlaySound(data.filepath, point, volume);
    }
}

void ImpactAudioSystem::PlaySound(const std::string& path, const Vec3& position, float volume) {
    // Fire and forget sound
    // Needs AudioSystem to support one-shot 3D sounds.
    // Since AudioSource is an object, we usually need to spawn one or use a pool.
    
    // For now, create a temporary AudioSource (managed pointer issue?)
    // Or AudioSystem might have PlayOneShot.
    
    // Let's create a "OneShotAudioObject" (GameObject) with AudioSource?
    // That's heavy.
    // Better: AudioSystem::PlayOneShot(path, pos, vol)
    
    // Check AudioSystem capability. If not exists, use raw hack?
    // Implementing a pool of AudioSources is best.
    
    // For this prototype, let's just log it or hack a source.
    // auto source = std::make_shared<AudioSource>();
    // source->Load(path);
    // source->SetPosition(position);
    // source->SetVolume(volume);
    // source->Play();
    // But who keeps it alive? It goes out of scope.
    
    // TODO: Implement AudioSystem::PlayClipAtPoint
    // For now, I'll print to console to prove it logic works.
    std::cout << "[ImpactAudio] Playing " << path << " Vol: " << volume << " at " << position.x << "," << position.y << std::endl;
    
    // If I really want sound:
    // Create a managed fire-and-forget object.
    /*
    auto go = std::make_shared<GameObject>("OneShotAudio");
    auto source = std::make_shared<AudioSource>();
    source->Load(path);
    source->SetPosition(position);
    source->SetVolume(volume);
    source->SetMinDistance(2.0f);
    source->SetMaxDistance(30.0f);
    source->SetRolloff(1.0f);
    go->AddAudioSource(source);
    // We need to add GO to Scene... Use Scene::Current->AddObject(go);
    // Then delete after clip length.
    */
}
