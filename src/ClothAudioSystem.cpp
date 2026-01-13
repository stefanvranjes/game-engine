#include "ClothAudioSystem.h"
#include "PhysXCloth.h"
#include "AudioSource.h"
#include <algorithm>

ClothAudioSystem& ClothAudioSystem::Get() {
    static ClothAudioSystem instance;
    return instance;
}

#ifdef USE_PHYSX
bool ClothAudioSystem::Initialize() {
    if (m_Initialized) return true;
    m_Initialized = true;
    return true;
}

void ClothAudioSystem::Shutdown() {
    for (auto& pair : m_ClothMap) {
        if (pair.second.source) pair.second.source->Stop();
    }
    m_ClothMap.clear();
    m_Initialized = false;
}

void ClothAudioSystem::RegisterCloth(PhysXCloth* cloth, const std::string& soundPath, float volumeScale) {
    if (!cloth) return;

    // Remove existing if any
    UnregisterCloth(cloth);

    ClothAudioData data;
    data.cloth = cloth;
    data.volumeScale = volumeScale;
    data.smoothVelocity = 0.0f;
    
    data.source = std::make_shared<AudioSource>();
    data.source->Load(soundPath);
    data.source->SetLooping(true);
    
    // Set initial position (center of cloth)
    Vec3 minB, maxB;
    cloth->GetWorldBounds(minB, maxB);
    data.source->SetPosition((minB + maxB) * 0.5f);
    
    // Start playing at volume 0
    data.source->SetVolume(0.0f);
    data.source->Play();

    m_ClothMap[cloth] = data;
}

void ClothAudioSystem::UnregisterCloth(PhysXCloth* cloth) {
    auto it = m_ClothMap.find(cloth);
    if (it != m_ClothMap.end()) {
        if (it->second.source) it->second.source->Stop();
        m_ClothMap.erase(it);
    }
}

void ClothAudioSystem::Update(float deltaTime) {
    for (auto& pair : m_ClothMap) {
        ClothAudioData& data = pair.second;
        if (!data.cloth || !data.source) continue;

        // Get velocity metric
        // We combine Average Velocity (movement) and Deformation Rate (rustling)
        float avgVel = data.cloth->GetAverageVelocity();
        float defRate = data.cloth->GetDeformationRate();
        
        // Combined energy metric
        float energy = avgVel + defRate * 0.5f;

        // Smooth it
        float alpha = 5.0f * deltaTime; // Smoothing factor
        // Clamp alpha 0..1
        if (alpha > 1.0f) alpha = 1.0f;
        
        data.smoothVelocity = data.smoothVelocity * (1.0f - alpha) + energy * alpha;
        
        // Calculate volume
        float volume = 0.0f;
        if (data.smoothVelocity > m_MinVelocity) {
            volume = (data.smoothVelocity - m_MinVelocity) * m_VelocityScale;
            if (volume > 1.0f) volume = 1.0f;
        }
        
        // Update Source
        Vec3 minB, maxB;
        data.cloth->GetWorldBounds(minB, maxB);
        data.source->SetPosition((minB + maxB) * 0.5f);
        
        data.source->SetVolume(volume * data.volumeScale);
        
        // Pitch modulation?
        // data.source->SetPitch(0.9f + volume * 0.2f);
    }
}
#else
// Dummy implementations when PhysX is disabled
bool ClothAudioSystem::Initialize() { return true; }
void ClothAudioSystem::Shutdown() {}
void ClothAudioSystem::RegisterCloth(PhysXCloth* cloth, const std::string& soundPath, float volumeScale) {}
void ClothAudioSystem::UnregisterCloth(PhysXCloth* cloth) {}
void ClothAudioSystem::Update(float deltaTime) {}
#endif
