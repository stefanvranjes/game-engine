#include "AudioSource.h"
#include "AudioSystem.h"
#include <iostream>

AudioSource::AudioSource() {
}

AudioSource::~AudioSource() {
    if (m_loaded) {
        ma_sound_uninit(&m_sound);
    }
}

void AudioSource::Load(const std::string& filepath, Type type) {
    if (m_loaded) {
        ma_sound_uninit(&m_sound);
        m_loaded = false;
    }

    ma_engine* engine = AudioSystem::Get().GetEngine();
    if (!engine) return;

    ma_uint32 flags = MA_SOUND_FLAG_ASYNC;
    ma_sound_group* targetGroup = nullptr;

    if (type == Type::Music) {
        flags |= MA_SOUND_FLAG_STREAM; // Stream music
        targetGroup = AudioSystem::Get().GetMusicGroup();
    } else {
        flags |= MA_SOUND_FLAG_DECODE; // Decode SFX to memory
        targetGroup = AudioSystem::Get().GetWorldGroup();
    }
    
    ma_result result = ma_sound_init_from_file(engine, filepath.c_str(), flags, targetGroup, NULL, &m_sound);
    if (result != MA_SUCCESS) {
        std::cerr << "Failed to load sound: " << filepath << " Error: " << result << std::endl;
        return;
    }

    if (type == Type::Music) {
        // Disable spatialization for music so it's always 2D
        ma_sound_set_spatialization_enabled(&m_sound, false);
    }

    m_loaded = true;
}

void AudioSource::Play() {
    if (!m_loaded) return;
    ma_sound_start(&m_sound);
}

void AudioSource::Pause() {
    if (!m_loaded) return;
    ma_sound_stop(&m_sound);
}

void AudioSource::Stop() {
    if (!m_loaded) return;
    ma_sound_stop(&m_sound);
    ma_sound_seek_to_pcm_frame(&m_sound, 0);
}

void AudioSource::SetLooping(bool loop) {
    if (!m_loaded) return;
    ma_sound_set_looping(&m_sound, loop);
}

bool AudioSource::IsLooping() const {
    if (!m_loaded) return false;
    return ma_sound_is_looping(&m_sound);
}

void AudioSource::SetVolume(float volume) {
    if (!m_loaded) return;
    ma_sound_set_volume(&m_sound, volume);
}

float AudioSource::GetVolume() const {
    if (!m_loaded) return 0.0f;
    return ma_sound_get_volume(&m_sound);
}

void AudioSource::SetPitch(float pitch) {
    if (!m_loaded) return;
    ma_sound_set_pitch(&m_sound, pitch);
}

float AudioSource::GetPitch() const {
    if (!m_loaded) return 1.0f;
    return ma_sound_get_pitch(&m_sound);
}

void AudioSource::SetPosition(const Vec3& position) {
    if (!m_loaded) return;
    ma_sound_set_position(&m_sound, position.x, position.y, position.z);
    m_Position = position;
}

void AudioSource::SetDirection(const Vec3& forward) {
    if (!m_loaded) return;
    ma_sound_set_direction(&m_sound, forward.x, forward.y, forward.z);
}

void AudioSource::SetVelocity(const Vec3& velocity) {
    if (!m_loaded) return;
    ma_sound_set_velocity(&m_sound, velocity.x, velocity.y, velocity.z);
}

void AudioSource::SetAttenuationModel(AttenuationModel model) {
    if (!m_loaded) return;
    
    ma_attenuation_model maModel = ma_attenuation_model_none;
    switch (model) {
        case AttenuationModel::None: maModel = ma_attenuation_model_none; break;
        case AttenuationModel::Inverse: maModel = ma_attenuation_model_inverse; break;
        case AttenuationModel::Linear: maModel = ma_attenuation_model_linear; break;
        case AttenuationModel::Exponential: maModel = ma_attenuation_model_exponential; break;
    }
    ma_sound_set_attenuation_model(&m_sound, maModel);
}

void AudioSource::SetRolloff(float rolloff) {
    if (!m_loaded) return;
    ma_sound_set_rolloff(&m_sound, rolloff);
}

void AudioSource::SetMinDistance(float distance) {
    if (!m_loaded) return;
    ma_sound_set_min_distance(&m_sound, distance);
}

void AudioSource::SetMaxDistance(float distance) {
    if (!m_loaded) return;
    ma_sound_set_max_distance(&m_sound, distance);
}

void AudioSource::SetCone(float innerAngleRad, float outerAngleRad, float outerGain) {
    if (!m_loaded) return;
    ma_sound_set_cone(&m_sound, innerAngleRad, outerAngleRad, outerGain);
}

void AudioSource::SetOcclusion(float strength) {
    if (!m_loaded) return;
    // Simple "Muffle" by reducing volume.
    // LPF would be better (e.g. ma_sound_set_lpf_cutoff), but standard engine doesn't expose it per sound easily
    // without custom node graph config.
    // Miniaudio High-Level API (ma_engine) sets up a basic graph.
    // Let's check if we can simulate it with volume.
    
    // Reduce volume based on occlusion strength (0 to 1)
    // But we need to play nice with Base Volume.
    // So we need to store base volume vs current effective volume.
    // For now, let's just multiply the base volume by (1 - strength).
    // Note: This overrides SetVolume calls if we aren't careful.
    // Better: Apply as a separate attenuation factor if possible? 
    // Miniaudio doesn't have "Occlusion" factor.
    // Let's modify SetVolume to take m_Occlusion into account.
    
    m_Occlusion = strength;
    float effectiveVolume = m_Volume * (1.0f - m_Occlusion);
    ma_sound_set_volume(&m_sound, effectiveVolume);
}

void AudioSource::SetDopplerFactor(float factor) {
    if (!m_loaded) return;
    ma_sound_set_doppler_factor(&m_sound, factor);
}
