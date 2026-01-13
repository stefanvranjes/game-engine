#include "AudioMixer.h"
#include <iostream>
#include <cmath>

AudioMixer& AudioMixer::Get() {
    static AudioMixer instance;
    return instance;
}

AudioMixer::AudioMixer() {}

AudioMixer::~AudioMixer() {
    Shutdown();
}

bool AudioMixer::Initialize(ma_engine* engine) {
    if (m_initialized) return true;
    if (!engine) return false;

    m_engine = engine;

    // Create master group
    m_masterGroup = new ChannelGroup();
    m_masterGroup->name = "Master";
    m_masterGroup->type = ChannelGroupType::Master;
    m_masterGroup->group = nullptr; // Master has no ma_sound_group

    // Create standard groups
    CreateStandardGroup("Music", ChannelGroupType::Music, m_masterGroup);
    CreateStandardGroup("SFX", ChannelGroupType::SFX, m_masterGroup);
    CreateStandardGroup("UI", ChannelGroupType::UI, m_masterGroup);
    CreateStandardGroup("Dialogue", ChannelGroupType::Dialogue, m_masterGroup);
    CreateStandardGroup("Ambient", ChannelGroupType::Ambient, m_masterGroup);

    m_initialized = true;
    return true;
}

void AudioMixer::Shutdown() {
    if (!m_initialized) return;

    // Clean up all groups (in reverse order for safety)
    for (auto* group : m_allGroups) {
        if (group->group) {
            ma_sound_group_uninit(group->group);
            delete group->group;
        }
        delete group;
    }

    if (m_masterGroup) {
        delete m_masterGroup;
        m_masterGroup = nullptr;
    }

    m_standardGroups.clear();
    m_customGroups.clear();
    m_allGroups.clear();
    m_initialized = false;
}

AudioMixer::ChannelGroup* AudioMixer::CreateStandardGroup(const std::string& name, AudioMixer::ChannelGroupType type, AudioMixer::ChannelGroup* parent) {
    if (!m_engine) return nullptr;

    auto* group = new ChannelGroup();
    group->name = name;
    group->type = type;
    group->parent = parent;

    // Create underlying ma_sound_group
    auto* maGroup = new ma_sound_group();
    ma_result result = ma_sound_group_init(m_engine, 0, NULL, maGroup);
    if (result != MA_SUCCESS) {
        std::cerr << "Failed to create sound group: " << name << std::endl;
        delete maGroup;
        delete group;
        return nullptr;
    }

    group->group = maGroup;

    if (parent) {
        parent->children.push_back(group);
    }

    m_standardGroups[static_cast<int>(type)] = group;
    m_allGroups.push_back(group);

    return group;
}

void AudioMixer::Update(float deltaTime) {
    if (!m_initialized) return;

    UpdateFades(deltaTime);
}

void AudioMixer::UpdateFades(float deltaTime) {
    for (auto* group : m_allGroups) {
        if (group->fadeDuration > 0.0f && group->fadeElapsed < group->fadeDuration) {
            group->fadeElapsed += deltaTime;
            float progress = std::min(1.0f, group->fadeElapsed / group->fadeDuration);

            // Linear interpolation
            float currentVolume = group->volume + (group->fadeTarget - group->volume) * progress;
            SetGroupVolume(group->type, currentVolume);

            // Fade complete
            if (group->fadeElapsed >= group->fadeDuration) {
                group->fadeDuration = 0.0f;
                SetGroupVolume(group->type, group->fadeTarget);
            }
        }
    }
}

AudioMixer::ChannelGroup* AudioMixer::GetGroup(ChannelGroupType type) {
    auto it = m_standardGroups.find(static_cast<int>(type));
    if (it != m_standardGroups.end()) {
        return it->second;
    }
    return nullptr;
}

AudioMixer::ChannelGroup* AudioMixer::GetGroupByName(const std::string& name) {
    for (auto* group : m_allGroups) {
        if (group->name == name) {
            return group;
        }
    }
    return nullptr;
}

AudioMixer::ChannelGroup* AudioMixer::CreateCustomGroup(const std::string& name, AudioMixer::ChannelGroupType parentType) {
    if (!m_engine) return nullptr;
    if (GetGroupByName(name)) return nullptr; // Already exists

    auto* group = new ChannelGroup();
    group->name = name;
    group->type = ChannelGroupType::Custom;

    auto* maGroup = new ma_sound_group();
    ma_result result = ma_sound_group_init(m_engine, 0, NULL, maGroup);
    if (result != MA_SUCCESS) {
        std::cerr << "Failed to create custom sound group: " << name << std::endl;
        delete maGroup;
        delete group;
        return nullptr;
    }

    group->group = maGroup;
    group->parent = GetGroup(parentType);

    if (group->parent) {
        group->parent->children.push_back(group);
    }

    m_customGroups[name] = group;
    m_allGroups.push_back(group);

    return group;
}

void AudioMixer::DestroyCustomGroup(const std::string& name) {
    auto it = m_customGroups.find(name);
    if (it == m_customGroups.end()) return;

    auto* group = it->second;

    // Remove from parent's children
    if (group->parent) {
        auto& children = group->parent->children;
        children.erase(std::remove(children.begin(), children.end(), group), children.end());
    }

    // Remove from all groups
    m_allGroups.erase(std::remove(m_allGroups.begin(), m_allGroups.end(), group), m_allGroups.end());

    // Cleanup
    if (group->group) {
        ma_sound_group_uninit(group->group);
        delete group->group;
    }
    delete group;
    m_customGroups.erase(it);
}

void AudioMixer::SetGroupVolume(AudioMixer::ChannelGroupType type, float volume) {
    auto* group = GetGroup(type);
    if (!group) return;

    group->volume = volume;
    ApplyGroupVolume(group);
}

void AudioMixer::SetGroupVolumeByName(const std::string& name, float volume) {
    auto* group = GetGroupByName(name);
    if (!group) return;

    group->volume = volume;
    ApplyGroupVolume(group);
}

void AudioMixer::ApplyGroupVolume(AudioMixer::ChannelGroup* group) {
    if (!group || !group->group) return;

    float effectiveVolume = group->volume;

    // Account for master volume
    effectiveVolume *= m_masterVolume;

    // Account for mute state
    if (group->muted || m_masterMuted) {
        effectiveVolume = 0.0f;
    }

    ma_sound_group_set_volume(group->group, effectiveVolume);
}

float AudioMixer::GetGroupVolume(AudioMixer::ChannelGroupType type) const {
    auto it = m_standardGroups.find(static_cast<int>(type));
    if (it != m_standardGroups.end()) {
        return it->second->volume;
    }
    return 0.0f;
}

void AudioMixer::MuteGroup(AudioMixer::ChannelGroupType type, bool mute) {
    auto* group = GetGroup(type);
    if (!group) return;

    group->muted = mute;
    if (mute) {
        group->muteVolume = group->volume;
        ApplyGroupVolume(group);
    } else {
        group->volume = group->muteVolume;
        ApplyGroupVolume(group);
    }
}

bool AudioMixer::IsGroupMuted(AudioMixer::ChannelGroupType type) const {
    auto* group = GetGroup(type);
    return group ? group->muted : false;
}

void AudioMixer::ToggleMute(AudioMixer::ChannelGroupType type) {
    auto* group = GetGroup(type);
    if (!group) return;
    MuteGroup(type, !group->muted);
}

void AudioMixer::FadeVolume(AudioMixer::ChannelGroupType type, float targetVolume, float duration) {
    auto* group = GetGroup(type);
    if (!group) return;

    group->fadeTarget = targetVolume;
    group->fadeDuration = duration;
    group->fadeElapsed = 0.0f;
}

void AudioMixer::CrossFade(AudioMixer::ChannelGroupType fromType, AudioMixer::ChannelGroupType toType, float duration) {
    FadeVolume(fromType, 0.0f, duration);
    FadeVolume(toType, 1.0f, duration);
}

void AudioMixer::SetLowPassFilter(AudioMixer::ChannelGroupType type, float cutoffHz) {
    auto* group = GetGroup(type);
    if (!group || !group->group) return;

    group->lpfCutoff = cutoffHz;
    // Miniaudio doesn't expose per-group LPF in the high-level API
    // This would require custom node graph setup
    // For now, we store the value for application via node graphs if implemented
}

void AudioMixer::SetHighPassFilter(AudioMixer::ChannelGroupType type, float cutoffHz) {
    auto* group = GetGroup(type);
    if (!group || !group->group) return;

    group->hpfCutoff = cutoffHz;
}

void AudioMixer::ResetFilters(AudioMixer::ChannelGroupType type) {
    auto* group = GetGroup(type);
    if (!group) return;

    group->lpfCutoff = 20000.0f;
    group->hpfCutoff = 20.0f;
}

float AudioMixer::GetLowPassFilterCutoff(AudioMixer::ChannelGroupType type) const {
    auto it = m_standardGroups.find(static_cast<int>(type));
    if (it != m_standardGroups.end()) {
        return it->second->lpfCutoff;
    }
    return 20000.0f;
}

float AudioMixer::GetHighPassFilterCutoff(AudioMixer::ChannelGroupType type) const {
    auto it = m_standardGroups.find(static_cast<int>(type));
    if (it != m_standardGroups.end()) {
        return it->second->hpfCutoff;
    }
    return 20.0f;
}

void AudioMixer::SetCompression(AudioMixer::ChannelGroupType type, float thresholdDb, float ratio,
                                 float attackMs, float releaseMs) {
    auto* group = GetGroup(type);
    if (!group) return;

    group->compressorThreshold = thresholdDb;
    // Store compression parameters for potential DSP implementation
}

void AudioMixer::DisableCompression(AudioMixer::ChannelGroupType type) {
    auto* group = GetGroup(type);
    if (!group) return;

    group->compressorThreshold = 0.0f;
}

void AudioMixer::SetMasterVolume(float volume) {
    m_masterVolume = volume;

    // Reapply to all groups
    for (auto* group : m_allGroups) {
        ApplyGroupVolume(group);
    }

    // Also apply to engine
    if (m_engine) {
        ma_engine_set_volume(m_engine, volume);
    }
}

float AudioMixer::GetMasterVolume() const {
    return m_masterVolume;
}

void AudioMixer::MuteAll(bool mute) {
    m_masterMuted = mute;

    // Reapply to all groups
    for (auto* group : m_allGroups) {
        ApplyGroupVolume(group);
    }
}

ma_sound_group* AudioMixer::GetMAAudioGroup(AudioMixer::ChannelGroupType type) {
    auto* group = GetGroup(type);
    return group ? group->group : nullptr;
}
