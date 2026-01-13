#include "ContinuousAudioSystem.h"
#include "PhysicsSystem.h"
#include "PhysXBackend.h"
#include "ImpactAudioSystem.h" // For getting object surfaces? Or should we expose that?
#include "AudioSource.h"
#include "GameObject.h"
#include "IPhysicsRigidBody.h"
#include <iostream>
#include <algorithm>

// Need access to ImpactAudioSystem's surface registry or duplicate it?
// Best to expose GetSurfaceType from ImpactAudioSystem or move registry to shared place.
// For now, let's assume we can access it or we duplicate logic (which is bad).
// Let's declare helper to get surface.

// Helper to get surface type (Hack: duplicating map lookups or accessing singleton if exposed)
// We will add a helper in ImpactAudioSystem to query surface type.
extern AudioSurfaceType GetObjectSurfaceType(GameObject* obj); 
// Note: We need to modify ImpactAudioSystem to provide this, or move registry to AudioSurface.h/cpp or Component.

ContinuousAudioSystem& ContinuousAudioSystem::Get() {
    static ContinuousAudioSystem instance;
    return instance;
}

bool ContinuousAudioSystem::Initialize() {
    if (m_Initialized) return true;

    // Listen to Global Collision Callback
    // Note: ImpactAudioSystem also sets this. We need to CHAIN them.
    // Ideally PhysXBackend supports multiple listeners or we use a composite.
    // PhysXBackend::SetGlobalCollisionCallback replaces the callback.
    // So we must verify if one is already set.
    
    // BETTER: Modify PhysXBackend to support a list of callbacks? Or use an EventDispatcher.
    // For now, let's try to wrap the existing one if possible, or assume we control the main lambda.
    
    // We will update local logic to check for CollisionEventType
    
    m_Initialized = true;
    return true;
}

void ContinuousAudioSystem::Shutdown() {
    // Stop all sounds
    for (auto& pair : m_ActiveSounds) {
        if (pair.second.source) pair.second.source->Stop();
    }
    m_ActiveSounds.clear();
    m_Initialized = false;
}

void ContinuousAudioSystem::SetRollSound(AudioSurfaceType surface, const std::string& soundPath, float volumeScale, float pitchScale) {
    m_RollSounds[static_cast<int>(surface)] = {soundPath, volumeScale, pitchScale};
}

void ContinuousAudioSystem::SetSlideSound(AudioSurfaceType surface, const std::string& soundPath, float volumeScale, float pitchScale) {
    m_SlideSounds[static_cast<int>(surface)] = {soundPath, volumeScale, pitchScale};
}

void ContinuousAudioSystem::OnCollision(IPhysicsRigidBody* bodyA, IPhysicsRigidBody* bodyB, const Vec3& point, const Vec3& normal, float impulse, int eventType) {
    auto key = std::make_pair(std::min(bodyA, bodyB), std::max(bodyA, bodyB));
    
    // PhysXCollisionEventType: 0=Enter, 1=Stay, 2=Exit (defined in PhysXBackend)
    // We assume implicit cast or values match our expectation. 
    // Enter(0), Stay(1), Exit(2).
    
    if (eventType == 2) { // Exit
        StopSound(key);
        return;
    }

    // Determine if rolling or sliding
    // "impulse" passed here is actually relative Velocity magnitude in our modified Backend logic for Stay events?
    // In OnContact modification earlier, we computed relVel.Length().
    
    float speed = impulse; // Using impulse (velocity) proxy
    if (speed < 0.1f) {
        // Too slow, stop sound
        StopSound(key);
        return;
    }

    // Check if we already have a sound
    auto it = m_ActiveSounds.find(key);
    if (it == m_ActiveSounds.end()) {
        // Start new sound
        // Determine Surface (Assuming one is dynamic object rolling on static surface, or both dynamic)
        GameObject* objA = static_cast<GameObject*>(bodyA->GetUserData());
        GameObject* objB = static_cast<GameObject*>(bodyB->GetUserData());
        
        // Strategy: Play sound of the "softer" or specific material?
        // E.g. Rock rolling on Wood -> Play Rock Roll or Wood Roll? Usually the object *doing* the rolling.
        // Let's pick the first one that has a defined sound.
        
        // Need to get surface type. 
        // Hack: We need a shared registry. For now, assume ImpactAudioSystem methods are static or we can access.
        // Actually, let's just use defaults or require registration here too?
        // Let's defer "Getting Surface" to a shared component or just assume generic for now to prove loop.
        
        SoundDefinition* def = nullptr;
        // Search m_RollSounds for generic or specific
        if (m_RollSounds.count(0)) def = &m_RollSounds[0]; // Generic
        
        if (def) {
            ActiveSound sound;
            sound.source = std::make_shared<AudioSource>();
            sound.source->Load(def->filepath);
            sound.source->SetLooping(true);
            sound.source->SetPosition(point);
            sound.source->SetMinDistance(1.0f);
            sound.source->SetMaxDistance(50.0f); // Default
            sound.source->Play();
            
            m_ActiveSounds[key] = sound;
            it = m_ActiveSounds.find(key);
        }
    }
    
    if (it != m_ActiveSounds.end()) {
        // Update existing sound
        UpdateSound(it->second, point, speed);
    }
}

void ContinuousAudioSystem::UpdateSound(ActiveSound& sound, const Vec3& pos, float speed, float maxSpeed) {
    if (!sound.source) return;
    
    sound.source->SetPosition(pos);
    
    // Modulation
    float t = std::min(1.0f, speed / maxSpeed);
    float targetVol = t; // * define scale
    float targetPitch = 0.8f + 0.4f * t; 
    
    sound.source->SetVolume(targetVol);
    sound.source->SetPitch(targetPitch);
}

void ContinuousAudioSystem::StopSound(std::pair<IPhysicsRigidBody*, IPhysicsRigidBody*> key) {
    auto it = m_ActiveSounds.find(key);
    if (it != m_ActiveSounds.end()) {
        if (it->second.source) {
            it->second.source->Stop();
        }
        m_ActiveSounds.erase(it);
    }
}
