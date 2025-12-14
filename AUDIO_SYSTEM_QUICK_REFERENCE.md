# Audio System Quick Reference

## What Was Added

Three new advanced audio components for the Game Engine:

### 1. AudioMixer (`AudioMixer.h/cpp`)
Higher-level mixing system with predefined channel groups and dynamic controls.

**Key Features:**
- Standard groups: Master, Music, SFX, UI, Dialogue, Ambient
- Custom group creation
- Per-group volume, mute, fade controls
- Cross-fading between groups
- Filter and compression parameters

**Quick Example:**
```cpp
AudioMixer& mixer = AudioMixer::Get();
mixer.FadeVolume(AudioMixer::ChannelGroupType::Music, 0.0f, 2.0f);
mixer.MuteGroup(AudioMixer::ChannelGroupType::SFX);
```

---

### 2. AudioSpatializer (`AudioSpatializer.h/cpp`)
Advanced 3D spatial audio with HRTF, Doppler, and distance models.

**Key Features:**
- HRTF profiles (Generic, Large, Small, Custom)
- Distance attenuation (Inverse, Linear, Exponential)
- Directional sound cones with falloff
- Doppler pitch shifting
- Spherical coordinate conversion
- Stereo/surround panning

**Quick Example:**
```cpp
AudioSpatializer& spat = AudioSpatializer::Get();
spat.SetHRTFProfile(AudioSpatializer::HRTFProfile::Generic);
spat.SetDistanceModel(AudioSpatializer::DistanceModel::InverseClamped);

auto output = spat.ComputeSpatialization(params);
audioSource->SetVolume(output.effectiveVolume);
audioSource->SetPitch(output.dopplerPitch);
```

---

### 3. AudioOcclusion (`AudioOcclusion.h/cpp`)
Physics-based occlusion using raycasting and material properties.

**Key Features:**
- Material database (Air, Glass, Brick, Metal, Concrete, Stone, Water, Custom)
- Obstacle registration with materials
- Raycast-based occlusion computation
- Frequency filtering (LPF/HPF)
- Advanced filtering modes

**Quick Example:**
```cpp
AudioOcclusion& occlusion = AudioOcclusion::Get();
occlusion.RegisterObstacle(wallObject, AudioOcclusion::MaterialType::Brick);
occlusion.ApplyOcclusionToSource(audioSource, listenerPos, sourcePos);
```

---

## Integration Points

### AudioSystem
- New `Update(float deltaTime)` method for mixer updates
- Initializes all three subsystems
- Still provides legacy API compatibility

### AudioSource
- Existing `SetOcclusion()` enhanced for occlusion system
- Works with all existing properties

### Game Loop
```cpp
void Application::Update(float deltaTime) {
    AudioSystem::Get().Update(deltaTime);  // Update mixer fades
    
    // For each audio source:
    auto spat = AudioSpatializer::Get().ComputeSpatialization(params);
    audioSource->SetVolume(spat.effectiveVolume);
    
    AudioOcclusion::Get().ApplyOcclusionToSource(audioSource, listenerPos, sourcePos);
}
```

---

## Files Modified

### New Files Created
- `include/AudioMixer.h` - Mixer API
- `src/AudioMixer.cpp` - Mixer implementation
- `include/AudioSpatializer.h` - Spatializer API
- `src/AudioSpatializer.cpp` - Spatializer implementation
- `include/AudioOcclusion.h` - Occlusion API
- `src/AudioOcclusion.cpp` - Occlusion implementation
- `docs/AUDIO_MIXER_SPATIALIZATION_GUIDE.md` - Comprehensive guide

### Files Modified
- `include/AudioSystem.h` - Added `Update()` method, new includes
- `src/AudioSystem.cpp` - Initialize subsystems, add Update implementation
- `CMakeLists.txt` - Added three new source files
- `README.md` - Updated audio features list

### Files Not Modified (Backward Compatible)
- `include/AudioSource.h` - No changes needed
- `include/AudioListener.h` - No changes needed
- Existing audio source code unchanged

---

## Quick Start

1. **Build the project:**
   ```
   build.bat
   ```

2. **Call Update in game loop:**
   ```cpp
   AudioSystem::Get().Update(deltaTime);
   ```

3. **Use mixer for volume control:**
   ```cpp
   AudioMixer::Get().SetGroupVolume(AudioMixer::ChannelGroupType::Music, volume);
   ```

4. **Compute and apply spatialization:**
   ```cpp
   auto output = AudioSpatializer::Get().ComputeSpatialization(params);
   audioSource->SetVolume(output.effectiveVolume);
   ```

5. **Register obstacles for occlusion:**
   ```cpp
   AudioOcclusion::Get().RegisterObstacle(wall, AudioOcclusion::MaterialType::Brick);
   AudioOcclusion::Get().ApplyOcclusionToSource(source, listener, sourcePos);
   ```

---

## Performance Notes

- **Mixer**: O(1) overhead, minimal impact
- **Spatializer**: O(1) per source, CPU-based computation
- **Occlusion**: O(n_obstacles) per raycast, can be optimized with caching
  - Use lower update frequency for distant sources
  - Configure `SetMaxOcclusionDistance()` to skip far sources

---

## Documentation

See [AUDIO_MIXER_SPATIALIZATION_GUIDE.md](docs/AUDIO_MIXER_SPATIALIZATION_GUIDE.md) for:
- Detailed architecture overview
- Complete usage examples
- Advanced configuration options
- Integration patterns
- Optimization techniques
- Debugging tips

---

## Backward Compatibility

✓ **Fully backward compatible** - All changes are additive
- Existing AudioSource code works unchanged
- New systems are opt-in
- Legacy volume/attenuation still functional
- No breaking changes to existing API

---

## Next Steps / Future Work

Potential enhancements:
1. Actual DSP filtering via miniaudio node graphs
2. Room acoustics simulation
3. Ambisonics/binaural rendering backend
4. Spectral occlusion (different frequencies affected differently)
5. Voice chat integration with automatic spatial mixing
6. Dynamic material properties (temperature, humidity effects)
