#pragma once

#include "SoftBodyDeformationRecorder.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

class PhysXSoftBody;

/**
 * @brief Manages synchronized recording of multiple soft bodies
 * 
 * Provides a unified system for recording and playing back multiple soft bodies
 * with perfect temporal synchronization using a master clock.
 */
class MultiBodyRecorder {
public:
    MultiBodyRecorder();
    ~MultiBodyRecorder();
    
    /**
     * @brief Add a soft body to the recording group
     * @param softBody Soft body to add
     * @param name Unique name for this body
     * @return True if added successfully
     */
    bool AddSoftBody(PhysXSoftBody* softBody, const std::string& name);
    
    /**
     * @brief Remove a soft body from the group
     * @param name Name of body to remove
     * @return True if removed successfully
     */
    bool RemoveSoftBody(const std::string& name);
    
    /**
     * @brief Clear all soft bodies from group
     */
    void Clear();
    
    /**
     * @brief Start synchronized recording for all bodies
     * @param sampleRate Sample rate for all bodies (default: 60 FPS)
     */
    void StartRecording(float sampleRate = 60.0f);
    
    /**
     * @brief Stop recording for all bodies
     */
    void StopRecording();
    
    /**
     * @brief Pause recording for all bodies
     */
    void PauseRecording();
    
    /**
     * @brief Resume recording for all bodies
     */
    void ResumeRecording();
    
    /**
     * @brief Check if currently recording
     */
    bool IsRecording() const { return m_IsRecording; }
    
    /**
     * @brief Start synchronized playback for all bodies
     */
    void StartPlayback();
    
    /**
     * @brief Stop playback for all bodies
     */
    void StopPlayback();
    
    /**
     * @brief Pause playback for all bodies
     */
    void PausePlayback();
    
    /**
     * @brief Check if currently playing back
     */
    bool IsPlayingBack() const { return m_IsPlaying; }
    
    /**
     * @brief Seek all bodies to specific time
     * @param time Time in seconds
     */
    void SeekPlayback(float time);
    
    /**
     * @brief Set playback speed for all bodies
     * @param speed Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
     */
    void SetPlaybackSpeed(float speed);
    
    /**
     * @brief Get current playback speed
     */
    float GetPlaybackSpeed() const { return m_PlaybackSpeed; }
    
    /**
     * @brief Set loop mode for all bodies
     */
    void SetLoopMode(LoopMode mode);
    
    /**
     * @brief Update all recorders (call per frame during recording/playback)
     * @param deltaTime Time since last frame
     */
    void Update(float deltaTime);
    
    /**
     * @brief Save entire group to file
     * @param filename Output file path
     * @return True if successful
     */
    bool SaveToFile(const std::string& filename) const;
    
    /**
     * @brief Load entire group from file
     * @param filename Input file path
     * @return True if successful
     */
    bool LoadFromFile(const std::string& filename);
    
    /**
     * @brief Get synchronized duration (longest recording)
     */
    float GetDuration() const;
    
    /**
     * @brief Get current playback time
     */
    float GetPlaybackTime() const { return m_MasterTime; }
    
    /**
     * @brief Get number of bodies in group
     */
    int GetBodyCount() const { return static_cast<int>(m_Bodies.size()); }
    
    /**
     * @brief Get body names
     */
    std::vector<std::string> GetBodyNames() const;
    
    /**
     * @brief Get recorder for specific body
     */
    SoftBodyDeformationRecorder* GetRecorder(const std::string& name);
    
    /**
     * @brief Apply compression settings to all recorders
     */
    void SetCompressionMode(CompressionMode mode);
    
    /**
     * @brief Apply interpolation mode to all recorders
     */
    void SetInterpolationMode(InterpolationMode mode);
    
private:
    struct BodyEntry {
        PhysXSoftBody* softBody;
        std::unique_ptr<SoftBodyDeformationRecorder> recorder;
        std::string name;
    };
    
    std::vector<BodyEntry> m_Bodies;
    std::unordered_map<std::string, int> m_NameToIndex;
    
    float m_MasterTime;
    float m_SampleRate;
    float m_PlaybackSpeed;
    
    bool m_IsRecording;
    bool m_IsPlaying;
    bool m_IsPaused;
    
    LoopMode m_LoopMode;
    
    // Helper methods
    int FindBodyIndex(const std::string& name) const;
};
