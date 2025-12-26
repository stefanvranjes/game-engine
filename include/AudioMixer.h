#pragma once
#include <miniaudio.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
/**
 * @class AudioMixer
 * @brief Higher-level audio mixing system with channel groups, effects, and dynamic controls.
 * 
 * Provides a hierarchical mixing architecture with multiple channel groups, each with
 * independent volume, mute, and effect controls. Enables complex audio management for
 * games with diverse audio requirements (SFX, music, UI, ambient, dialogue, etc.).
 */
// Stub header - not implemented  
class AudioMixer {
public:
    /**
     * @enum ChannelGroupType
     * @brief Predefined channel group categories.
     */
    enum class ChannelGroupType {
        Master,      // Master output (all audio routes here)
        Music,       // Background music/ambience
        SFX,         // Sound effects
        UI,          // UI/Menu sounds
        Dialogue,    // Voice/Dialogue
        Ambient,     // Environmental sounds
        Custom       // User-defined groups
    };
    /**
     * @struct ChannelGroup
     * @brief A mixing channel with its own volume, mute state, and effects chain.
     */
    struct ChannelGroup {
        std::string name;
        ChannelGroupType type;
        ma_sound_group* group = nullptr;
        
        float volume = 1.0f;           // Base volume (0.0 to 1.0+)
        bool muted = false;
        float muteVolume = 1.0f;       // Volume before muting (for unmute)
        
        // Effects
        float lpfCutoff = 20000.0f;    // Low-pass filter cutoff (Hz)
        float hpfCutoff = 20.0f;       // High-pass filter cutoff (Hz)
        float compressorThreshold = 0.0f; // Compression threshold (dB)
        
        // Hierarchy
        ChannelGroup* parent = nullptr;
        std::vector<ChannelGroup*> children;
        
        bool active = true;
        float fadeTarget = 1.0f;
        float fadeDuration = 0.0f;
        float fadeElapsed = 0.0f;
    };
public:
    static AudioMixer& Get();
    /**
     * @brief Initialize the mixer with standard channel groups.
     * @return true if initialization successful
     */
    bool Initialize(ma_engine* engine);
    /**
     * @brief Shutdown the mixer and clean up resources.
     */
    void Shutdown();
    /**
     * @brief Update mixer state (fades, time-based effects).
     * @param deltaTime Time elapsed since last update (seconds)
     */
    void Update(float deltaTime);
    // ============== Channel Group Management ==============
    /**
     * @brief Get a predefined channel group.
     * @param type The channel group type
     * @return Pointer to the ChannelGroup, or nullptr if not found
     */
    ChannelGroup* GetGroup(ChannelGroupType type);
    /**
     * @brief Get a channel group by name.
     * @param name The group's name
     * @return Pointer to the ChannelGroup, or nullptr if not found
     */
    ChannelGroup* GetGroupByName(const std::string& name);
    /**
     * @brief Create a custom channel group.
     * @param name Unique name for the group
     * @param parentType Parent group type (for hierarchy)
     * @return Pointer to the new ChannelGroup
     */
    ChannelGroup* CreateCustomGroup(const std::string& name, ChannelGroupType parentType = ChannelGroupType::Master);
    /**
     * @brief Destroy a custom channel group.
     * @param name Name of the group to destroy
     */
    void DestroyCustomGroup(const std::string& name);
    // ============== Volume & Mute Control ==============
    /**
     * @brief Set volume for a channel group.
     * @param type Channel group type
     * @param volume Volume level (0.0 to 1.0+)
     */
    void SetGroupVolume(ChannelGroupType type, float volume);
    /**
     * @brief Set volume by group name.
     * @param name Group name
     * @param volume Volume level
     */
    void SetGroupVolumeByName(const std::string& name, float volume);
    /**
     * @brief Get current volume of a channel group.
     */
    float GetGroupVolume(ChannelGroupType type) const;
    /**
     * @brief Mute a channel group.
     * @param type Channel group type
     */
    void MuteGroup(ChannelGroupType type, bool mute = true);
    /**
     * @brief Unmute a channel group.
     * @param type Channel group type
     */
    void UnmuteGroup(ChannelGroupType type) { MuteGroup(type, false); }
    /**
     * @brief Check if a channel group is muted.
     */
    bool IsGroupMuted(ChannelGroupType type) const;
    /**
     * @brief Toggle mute state.
     */
    void ToggleMute(ChannelGroupType type);
    // ============== Fading & Transitions ==============
    /**
     * @brief Fade a channel group's volume over time.
     * @param type Channel group type
     * @param targetVolume Target volume to fade to
     * @param duration Fade duration in seconds
     */
    void FadeVolume(ChannelGroupType type, float targetVolume, float duration);
    /**
     * @brief Cross-fade between two channel groups.
     * @param fromType Source group to fade out
     * @param toType Destination group to fade in
     * @param duration Fade duration in seconds
     */
    void CrossFade(ChannelGroupType fromType, ChannelGroupType toType, float duration);
    // ============== Effects & Filters ==============
    /**
     * @brief Apply low-pass filter to a channel group.
     * @param type Channel group type
     * @param cutoffHz Cutoff frequency in Hz
     */
    void SetLowPassFilter(ChannelGroupType type, float cutoffHz);
    /**
     * @brief Apply high-pass filter to a channel group.
     * @param type Channel group type
     * @param cutoffHz Cutoff frequency in Hz
     */
    void SetHighPassFilter(ChannelGroupType type, float cutoffHz);
    /**
     * @brief Reset filters to default (no filtering).
     */
    void ResetFilters(ChannelGroupType type);
    /**
     * @brief Get low-pass filter cutoff frequency.
     */
    float GetLowPassFilterCutoff(ChannelGroupType type) const;
    /**
     * @brief Get high-pass filter cutoff frequency.
     */
    float GetHighPassFilterCutoff(ChannelGroupType type) const;
    // ============== Compression & Dynamics ==============
    /**
     * @brief Apply dynamic range compression to a group.
     * @param type Channel group type
     * @param thresholdDb Threshold in dB (e.g., -20.0)
     * @param ratio Compression ratio (e.g., 4.0 for 4:1)
     * @param attackMs Attack time in milliseconds
     * @param releaseMs Release time in milliseconds
     */
    void SetCompression(ChannelGroupType type, float thresholdDb, float ratio,
                        float attackMs, float releaseMs);
    /**
     * @brief Disable compression on a group.
     */
    void DisableCompression(ChannelGroupType type);
    // ============== Master Controls ==============
    /**
     * @brief Set master volume (affects all groups).
     */
    void SetMasterVolume(float volume);
    /**
     * @brief Get master volume.
     */
    float GetMasterVolume() const;
    /**
     * @brief Mute all audio.
     */
    void MuteAll(bool mute = true);
    /**
     * @brief Unmute all audio.
     */
    void UnmuteAll() { MuteAll(false); }
    // ============== Utilities ==============
    /**
     * @brief Get all active channel groups.
     */
    const std::vector<ChannelGroup*>& GetAllGroups() const { return m_allGroups; }
    /**
     * @brief Get the underlying miniaudio sound group.
     */
    ma_sound_group* GetMAAudioGroup(ChannelGroupType type);
private:
    AudioMixer();
    ~AudioMixer();
    // Internal helpers
    void ApplyGroupVolume(ChannelGroup* group);
    void UpdateFades(float deltaTime);
    ChannelGroup* CreateStandardGroup(const std::string& name, ChannelGroupType type, ChannelGroup* parent);
    ma_engine* m_engine = nullptr;
    bool m_initialized = false;
    // Standard groups
    std::unordered_map<int, ChannelGroup*> m_standardGroups;
    std::unordered_map<std::string, ChannelGroup*> m_customGroups;
    std::vector<ChannelGroup*> m_allGroups;
    // Master control
    float m_masterVolume = 1.0f;
    bool m_masterMuted = false;
    // Hierarchy root
    ChannelGroup* m_masterGroup = nullptr;
    AudioMixer() {}
    ~AudioMixer() {}
};
