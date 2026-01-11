#pragma once

#include "SoftBodyDeformationRecorder.h"
#include <memory>
#include <string>
#include <functional>

class PhysXSoftBody;
class MultiBodyRecorder;

/**
 * @brief ImGui panel for deformation recording controls
 * 
 * Provides a complete UI for managing deformation recording and playback
 * in the soft body editor.
 */
class SoftBodyRecordingPanel {
public:
    SoftBodyRecordingPanel();
    ~SoftBodyRecordingPanel();
    
    /**
     * @brief Render the recording panel UI
     * @param softBody Active soft body (can be nullptr)
     */
    void Render(PhysXSoftBody* softBody);
    
private:
    // UI state - Recording
    float m_SampleRate;
    bool m_RecordVelocities;
    int m_MaxKeyframes;
    
    // UI state - Playback
    float m_PlaybackSpeed;
    int m_LoopModeIndex;
    
    // UI state - Compression
    int m_CompressionModeIndex;
    int m_ReferenceInterval;
    bool m_PartialRecording;
    float m_ChangeThreshold;
    int m_InterpolationModeIndex;
    
    // UI state - Streaming
    bool m_StreamingMode;
    int m_MemoryBudgetMB;
    
    // UI state - File operations
    char m_SaveFilename[256];
    char m_LoadFilename[256];
    int m_FileFormatIndex;
    float m_SaveProgress;
    bool m_ShowSaveDialog;
    bool m_ShowLoadDialog;
    
    // Multi-body recording
    std::unique_ptr<MultiBodyRecorder> m_MultiBodyRecorder;
    char m_NewBodyName[128];
    int m_SelectedBodyIndex;
    
    // UI sections
    void RenderRecordingControls(PhysXSoftBody* softBody);
    void RenderPlaybackControls(PhysXSoftBody* softBody);
    void RenderCompressionSettings(PhysXSoftBody* softBody);
    void RenderStreamingSettings(PhysXSoftBody* softBody);
    void RenderFileOperations(PhysXSoftBody* softBody);
    void RenderStatistics(PhysXSoftBody* softBody);
    void RenderMultiBodySection();
    
    // Helpers
    void ApplySettings(PhysXSoftBody* softBody);
    const char* GetCompressionModeName(int index) const;
    const char* GetLoopModeName(int index) const;
    const char* GetInterpolationModeName(int index) const;
    const char* GetFileFormatName(int index) const;
};
