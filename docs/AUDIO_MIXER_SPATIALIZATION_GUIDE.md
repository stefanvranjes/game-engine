# Advanced Audio System: Mixer, Spatialization & Occlusion

## Overview

The Game Engine now includes a sophisticated, production-ready audio system with three major components:

1. **AudioMixer** - Higher-level mixing architecture with channel groups, fading, and effects
2. **AudioSpatializer** - Advanced 3D spatial audio with HRTF, Doppler, and distance attenuation
3. **AudioOcclusion** - Physics-based audio occlusion using raycasting and material properties

## System Architecture

```
AudioSystem (Core Manager)
    ├── AudioMixer (Channel Groups & Mixing)
    │   ├── Master Group (root)
    │   ├── Music Group
    │   ├── SFX Group
    │   ├── UI Group
    │   ├── Dialogue Group
    │   ├── Ambient Group
    │   └── Custom Groups
    │
    ├── AudioSpatializer (3D Spatial Processing)
    │   ├── HRTF Profiles (Generic, Large, Small, Custom)
    │   ├── Distance Attenuation Models (Inverse, Linear, Exponential)
    │   ├── Directional Cones & Falloff
    │   ├── Doppler Effect
    │   └── Panning (Stereo/Surround)
    │
    └── AudioOcclusion (Material-Based Blocking)
        ├── Obstacle Registration
        ├── Material Database (Glass, Brick, Metal, etc.)
        ├── Raycast Computation
        └── Frequency Filtering (LPF/HPF)
```

## Component Details

### AudioMixer

Higher-level abstraction over miniaudio's sound groups. Manages hierarchical mixing with independent volume/mute controls per channel.

**Features:**
- Predefined channel groups (Music, SFX, UI, Dialogue, Ambient)
- Custom group creation with parent-child hierarchy
- Per-group volume and mute state
- Fade-in/fade-out and cross-fades
- Dynamic range compression parameters
- Low-pass and high-pass filter configuration
- Master volume and global mute

**Usage:**

```cpp
// Get mixer instance
AudioMixer& mixer = AudioMixer::Get();

// Set group volume
mixer.SetGroupVolume(AudioMixer::ChannelGroupType::Music, 0.8f);
mixer.SetGroupVolume(AudioMixer::ChannelGroupType::SFX, 1.0f);

// Fade music over 2 seconds
mixer.FadeVolume(AudioMixer::ChannelGroupType::Music, 0.0f, 2.0f);

// Cross-fade: fade out music, fade in ambience
mixer.CrossFade(AudioMixer::ChannelGroupType::Music, 
                AudioMixer::ChannelGroupType::Ambient, 1.5f);

// Mute SFX temporarily
mixer.MuteGroup(AudioMixer::ChannelGroupType::SFX);
mixer.UnmuteGroup(AudioMixer::ChannelGroupType::SFX);

// Create custom group (e.g., Voice Chat)
AudioMixer::ChannelGroup* voiceChat = mixer.CreateCustomGroup(
    "VoiceChat", AudioMixer::ChannelGroupType::Master);

// Update mixer state in game loop
void Application::Update(float deltaTime) {
    AudioSystem::Get().Update(deltaTime);  // Handles mixer fades
    // ...
}
```

### AudioSpatializer

Advanced 3D spatialization engine. Computes spatial audio parameters including:
- Distance attenuation using various models (inverse, linear, exponential)
- Head-Related Transfer Functions (HRTF) for elevation cues
- Directional sound cones with smooth falloff
- Doppler pitch shifting from velocity
- Stereo/surround panning based on source position
- Frequency filtering for atmospheric effects

**HRTF Profiles:**
- **Generic**: Standard HRTF for average head size
- **Large**: Larger head with deeper ears (longer ITD)
- **Small**: Smaller head with shallower ears (shorter ITD)
- **Custom**: User-defined HRTF parameters

**Distance Models:**
- **Inverse**: Realistic outdoor sound (1/distance), commonly used
- **InverseClamped**: Inverse with safety bounds
- **Linear**: Linear fade from min to max distance
- **Exponential**: Sharp exponential decay for enclosed spaces
- **None**: No distance attenuation

**Usage:**

```cpp
AudioSpatializer& spatializer = AudioSpatializer::Get();

// Configure HRTF for better elevation perception
spatializer.SetHRTFProfile(AudioSpatializer::HRTFProfile::Generic);
spatializer.SetHRTFEnabled(true);

// Set distance attenuation model
spatializer.SetDistanceModel(AudioSpatializer::DistanceModel::InverseClamped);

// Compute spatial parameters for a sound source
AudioSpatializer::SpatializationParams params;
params.listenerPos = cameraPos;
params.listenerForward = cameraForward;
params.listenerUp = cameraUp;
params.sourcePos = soundSourcePos;
params.sourceVelocity = soundSourceVelocity;
params.sourceDirection = soundSourceForward;
params.minDistance = 1.0f;
params.maxDistance = 100.0f;
params.rolloff = 1.0f;  // How fast sound fades with distance
params.dopplerFactor = 1.0f;
params.coneInnerAngle = 1.57f;  // π/2 radians (90°)
params.coneOuterAngle = 3.14f;  // π radians (180°)
params.coneOuterGain = 0.5f;    // Volume at cone edge

auto output = spatializer.ComputeSpatialization(params);

// Apply computed parameters to audio source
audioSource->SetVolume(output.effectiveVolume);
audioSource->SetPitch(output.dopplerPitch);
// Apply panning: output.leftPan (0.0 = right, 1.0 = left)
// Apply filtering: LPF @ output.lpfCutoff Hz, HPF @ output.hpfCutoff Hz
```

### AudioOcclusion

Physics-based occlusion system. Determines how much audio is blocked by geometry between source and listener. Uses raycasting and material properties.

**Material Types:**
- **Air**: Transparent (0% occlusion)
- **Glass**: Transparent to sound (10% occlusion)
- **Drywall**: Moderate (40%)
- **Brick**: Strong (60%)
- **Wood**: Strong (50%)
- **Metal**: Heavy (70%)
- **Concrete**: Very Heavy (75%)
- **Stone**: Very Heavy (80%)
- **Water**: Extreme (85%)
- **Custom**: User-defined properties

**Material Properties Structure:**
```cpp
struct MaterialProperties {
    float occlusionFactor;      // 0.0 (transparent) to 1.0 (fully opaque)
    float dampingFactor;        // High-frequency absorption (0.0 to 1.0)
    float reflectionFactor;     // Sound reflection (0.0 to 1.0)
    float defaultThickness;     // Thickness for raycast hits (meters)
};
```

**Usage:**

```cpp
AudioOcclusion& occlusion = AudioOcclusion::Get();

// Register game objects as occlusive obstacles
occlusion.RegisterObstacle(wallGameObject, AudioOcclusion::MaterialType::Brick);
occlusion.RegisterObstacle(glassWindow, AudioOcclusion::MaterialType::Glass);
occlusion.RegisterObstacle(metalDoor, AudioOcclusion::MaterialType::Metal);

// Configure occlusion system
occlusion.SetEnabled(true);
occlusion.SetMaxOcclusionDistance(200.0f);  // Ignore occlusion beyond 200m
occlusion.SetAdvancedFiltering(true);        // Per-frequency occlusion

// Set LPF parameters for muffling
occlusion.SetLPFParameters(500.0f, 20000.0f, 1.5f);

// Compute occlusion in game loop
AudioOcclusion::OcclusionResult result = occlusion.ComputeOcclusion(
    listenerPos, soundSourcePos, soundSourceGameObject);

// Apply occlusion to audio source
audioSource->SetOcclusion(result.occlusionStrength);
// Also apply filters: LPF @ result.lpfCutoff Hz, HPF @ result.hpfCutoff Hz

// Or use convenience method
occlusion.ApplyOcclusionToSource(audioSource, listenerPos, soundSourcePos);

// Update material at runtime (e.g., breaking glass)
occlusion.UpdateObstacleMaterial(glassWindow, AudioOcclusion::MaterialType::Air);

// Unregister when object is destroyed
occlusion.UnregisterObstacle(wallGameObject);
```

## Integration with Existing Audio System

The new audio system integrates seamlessly with the existing `AudioSystem`, `AudioSource`, and `AudioListener` classes.

**AudioSource Enhancements:**
- `SetOcclusion(float strength)` - Apply occlusion from AudioOcclusion
- Supports volume, pitch, spatialization, directional cones, and Doppler

**AudioListener:**
- Single active listener that receives spatial updates from camera/player

**AudioSystem:**
- `Update(float deltaTime)` - Updates mixer fades and audio state
- Provides access to AudioMixer, AudioSpatializer, and AudioOcclusion singletons

## Integration Example: Complete 3D Audio Pipeline

```cpp
class GameAudioManager {
public:
    void Initialize() {
        AudioSystem::Get().Initialize();
        
        // Setup mixer defaults
        AudioMixer& mixer = AudioMixer::Get();
        mixer.SetGroupVolume(AudioMixer::ChannelGroupType::SFX, 0.8f);
        mixer.SetGroupVolume(AudioMixer::ChannelGroupType::Music, 0.7f);
        
        // Setup occlusion
        AudioOcclusion& occlusion = AudioOcclusion::Get();
        occlusion.SetAdvancedFiltering(true);
        occlusion.SetMaxOcclusionDistance(150.0f);
    }
    
    void Update(float deltaTime, const Vec3& listenerPos, const Vec3& listenerForward) {
        AudioSystem::Get().Update(deltaTime);
        
        AudioSpatializer& spatializer = AudioSpatializer::Get();
        AudioOcclusion& occlusion = AudioOcclusion::Get();
        
        // Update all active audio sources
        for (auto& source : activeSources) {
            // Spatialization
            AudioSpatializer::SpatializationParams params;
            params.listenerPos = listenerPos;
            params.listenerForward = listenerForward;
            params.sourcePos = source->GetPosition();
            params.sourceVelocity = source->GetVelocity();
            params.minDistance = source->GetMinDistance();
            params.maxDistance = source->GetMaxDistance();
            
            auto spatOutput = spatializer.ComputeSpatialization(params);
            source->SetVolume(spatOutput.effectiveVolume);
            source->SetPitch(spatOutput.dopplerPitch);
            // Apply panning via stereo matrix
            
            // Occlusion
            occlusion.ApplyOcclusionToSource(source, listenerPos, source->GetPosition());
        }
    }
};
```

## Performance Considerations

1. **Mixer**: Minimal overhead. Fades are computed once per frame. Consider caching group references.

2. **Spatializer**: O(1) per audio source. Reuse computed results if listener/source positions don't change frequently.

3. **Occlusion**: 
   - Raycast computation is the main bottleneck
   - Default: Single raycast per source per frame = O(n_obstacles)
   - Optimization: Cache results per source, update at lower frequency
   - Can increase `SetRaycastSampleCount()` for more accurate multi-ray tests

**Optimization Tips:**
```cpp
// Update occlusion less frequently (every 0.1 seconds for distant sources)
if (distanceToListener > 50.0f) {
    updateOcclusionInterval = 0.1f;
} else {
    updateOcclusionInterval = 0.016f;  // Every frame
}

// Cache frequently accessed groups
static auto sfxGroup = AudioMixer::Get().GetGroup(AudioMixer::ChannelGroupType::SFX);
mixer->SetGroupVolume(AudioMixer::ChannelGroupType::SFX, newVolume);  // Uses cached pointer
```

## Advanced Usage: Custom HRTF and Materials

```cpp
// Custom HRTF profile
AudioSpatializer::HRTFProfile customProfile = AudioSpatializer::HRTFProfile::Custom;
AudioSpatializer::Get().SetHRTFProfile(customProfile);

// Custom material with specific absorption
AudioOcclusion::MaterialProperties customMaterial;
customMaterial.occlusionFactor = 0.45f;
customMaterial.dampingFactor = 0.60f;
customMaterial.reflectionFactor = 0.3f;
customMaterial.defaultThickness = 0.15f;

AudioOcclusion::Get().SetMaterialProperties(
    AudioOcclusion::MaterialType::Custom, customMaterial);
AudioOcclusion::Get().RegisterObstacle(
    customWall, AudioOcclusion::MaterialType::Custom);
```

## Debugging

```cpp
// Check mixer state
auto groups = AudioMixer::Get().GetAllGroups();
for (auto* group : groups) {
    std::cout << group->name << ": " << group->volume << std::endl;
}

// Check occlusion computation
auto result = AudioOcclusion::Get().GetLastOcclusionResult();
std::cout << "Occlusion: " << result.occlusionStrength << std::endl;
std::cout << "Obstacles: " << result.occludingObstaclesCount << std::endl;
std::cout << "LPF: " << result.lpfCutoff << " Hz" << std::endl;

// Monitor obstacle count
int obstacleCount = AudioOcclusion::Get().GetObstacleCount();
```

## Future Enhancements

Potential improvements for future versions:

1. **DSP Effects**: Implement actual LPF/HPF filtering via miniaudio node graphs
2. **Room Acoustics**: Compute reflections and reverb from room geometry
3. **Spatial Audio Rendering**: Ambisonics or binaural rendering backend
4. **Frequency Bands**: Per-band occlusion (different attenuation for different frequencies)
5. **Sound Propagation**: Simulate sound traveling through multiple materials
6. **Spectral Analysis**: Real-time audio spectrum visualization
7. **Voice Chat Integration**: Dedicated voice group with automatic spatial mixing
8. **Dynamic Material Properties**: Temperature/humidity effects on sound propagation

## Files

- [include/AudioMixer.h](include/AudioMixer.h) - Channel group mixing API
- [src/AudioMixer.cpp](src/AudioMixer.cpp) - Mixer implementation
- [include/AudioSpatializer.h](include/AudioSpatializer.h) - 3D spatial audio API
- [src/AudioSpatializer.cpp](src/AudioSpatializer.cpp) - Spatializer implementation
- [include/AudioOcclusion.h](include/AudioOcclusion.h) - Occlusion API
- [src/AudioOcclusion.cpp](src/AudioOcclusion.cpp) - Occlusion implementation
- [include/AudioSystem.h](include/AudioSystem.h) - Core system manager
- [include/AudioSource.h](include/AudioSource.h) - Audio source component
- [include/AudioListener.h](include/AudioListener.h) - Audio listener component
