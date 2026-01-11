#include "SoftBodyRecordingPanel.h"
#include "PhysXSoftBody.h"
#include "MultiBodyRecorder.h"
#include <imgui.h>
#include <cstring>

SoftBodyRecordingPanel::SoftBodyRecordingPanel()
    : m_SampleRate(60.0f)
    , m_RecordVelocities(false)
    , m_MaxKeyframes(1000)
    , m_PlaybackSpeed(1.0f)
    , m_LoopModeIndex(0)
    , m_CompressionModeIndex(0)
    , m_ReferenceInterval(30)
    , m_PartialRecording(false)
    , m_ChangeThreshold(0.001f)
    , m_InterpolationModeIndex(0)
    , m_StreamingMode(false)
    , m_MemoryBudgetMB(100)
    , m_FileFormatIndex(1)  // Binary default
    , m_SaveProgress(0.0f)
    , m_ShowSaveDialog(false)
    , m_ShowLoadDialog(false)
    , m_SelectedBodyIndex(-1)
{
    std::strcpy(m_SaveFilename, "recording.bin");
    std::strcpy(m_LoadFilename, "recording.bin");
    std::strcpy(m_NewBodyName, "");
    
    m_MultiBodyRecorder = std::make_unique<MultiBodyRecorder>();
}

SoftBodyRecordingPanel::~SoftBodyRecordingPanel()
{
}

void SoftBodyRecordingPanel::Render(PhysXSoftBody* softBody)
{
    if (!ImGui::CollapsingHeader("Deformation Recording", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    
    if (!softBody) {
        ImGui::TextDisabled("No soft body selected");
        return;
    }
    
    ImGui::Indent();
    
    RenderRecordingControls(softBody);
    ImGui::Spacing();
    
    RenderPlaybackControls(softBody);
    ImGui::Spacing();
    
    RenderCompressionSettings(softBody);
    ImGui::Spacing();
    
    RenderStreamingSettings(softBody);
    ImGui::Spacing();
    
    RenderFileOperations(softBody);
    ImGui::Spacing();
    
    RenderStatistics(softBody);
    ImGui::Spacing();
    
    RenderMultiBodySection();
    
    ImGui::Unindent();
}

void SoftBodyRecordingPanel::RenderRecordingControls(PhysXSoftBody* softBody)
{
    auto recorder = softBody->GetRecorder();
    bool isRecording = recorder->IsRecording();
    
    ImGui::SeparatorText("Recording");
    
    // Record/Pause/Stop buttons
    if (isRecording) {
        if (ImGui::Button("Pause Recording")) {
            softBody->PauseRecording();
        }
        ImGui::SameLine();
        if (ImGui::Button("Stop Recording")) {
            softBody->StopRecording();
        }
    } else {
        if (ImGui::Button("Start Recording")) {
            ApplySettings(softBody);
            softBody->StartRecording(m_SampleRate, m_RecordVelocities);
        }
    }
    
    // Settings (disabled during recording)
    ImGui::BeginDisabled(isRecording);
    ImGui::SliderFloat("Sample Rate", &m_SampleRate, 10.0f, 120.0f, "%.0f FPS");
    ImGui::Checkbox("Record Velocities", &m_RecordVelocities);
    ImGui::InputInt("Memory Limit", &m_MaxKeyframes);
    ImGui::EndDisabled();
    
    // Status
    int keyframeCount = recorder->GetKeyframeCount();
    if (isRecording) {
        ImGui::Text("Status: Recording");
        ImGui::ProgressBar((float)keyframeCount / m_MaxKeyframes, ImVec2(-1, 0), 
                          (std::to_string(keyframeCount) + "/" + std::to_string(m_MaxKeyframes) + " frames").c_str());
    } else {
        ImGui::Text("Status: Stopped (%d frames)", keyframeCount);
    }
}

void SoftBodyRecordingPanel::RenderPlaybackControls(PhysXSoftBody* softBody)
{
    auto recorder = softBody->GetRecorder();
    bool isPlaying = recorder->IsPlayingBack();
    int keyframeCount = recorder->GetKeyframeCount();
    
    ImGui::SeparatorText("Playback");
    
    if (keyframeCount == 0) {
        ImGui::TextDisabled("No recording available");
        return;
    }
    
    // Play/Pause/Stop buttons
    if (isPlaying) {
        if (ImGui::Button("Pause Playback")) {
            softBody->PausePlayback();
        }
    } else {
        if (ImGui::Button("Play")) {
            softBody->StartPlayback();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        softBody->StopPlayback();
    }
    
    // Timeline scrubber
    float currentTime = recorder->GetPlaybackTime();
    float duration = recorder->GetDuration();
    if (ImGui::SliderFloat("Timeline", &currentTime, 0.0f, duration, "%.2fs")) {
        softBody->SeekPlayback(currentTime);
    }
    
    // Speed control
    if (ImGui::SliderFloat("Speed", &m_PlaybackSpeed, 0.1f, 10.0f, "%.1fx")) {
        softBody->SetPlaybackSpeed(m_PlaybackSpeed);
    }
    
    // Loop mode
    const char* loopModes[] = {"None", "Loop", "PingPong"};
    if (ImGui::Combo("Loop Mode", &m_LoopModeIndex, loopModes, 3)) {
        softBody->SetPlaybackLoopMode((LoopMode)m_LoopModeIndex);
    }
    
    // Time display
    ImGui::Text("Time: %.2fs / %.2fs", currentTime, duration);
}

void SoftBodyRecordingPanel::RenderCompressionSettings(PhysXSoftBody* softBody)
{
    auto recorder = softBody->GetRecorder();
    bool isRecording = recorder->IsRecording();
    
    ImGui::SeparatorText("Compression");
    
    ImGui::BeginDisabled(isRecording);
    
    // Compression mode
    const char* modes[] = {"None", "Delta Encoding", "Quantized"};
    if (ImGui::Combo("Mode", &m_CompressionModeIndex, modes, 3)) {
        recorder->SetCompressionMode((CompressionMode)m_CompressionModeIndex);
    }
    
    // Delta encoding settings
    if (m_CompressionModeIndex == 1) {
        if (ImGui::SliderInt("Reference Interval", &m_ReferenceInterval, 10, 100, "%d frames")) {
            recorder->SetReferenceFrameInterval(m_ReferenceInterval);
        }
    }
    
    // Partial recording
    if (ImGui::Checkbox("Partial Recording", &m_PartialRecording)) {
        recorder->SetPartialRecording(m_PartialRecording);
    }
    if (m_PartialRecording) {
        ImGui::Indent();
        if (ImGui::InputFloat("Change Threshold", &m_ChangeThreshold, 0.0001f, 0.001f, "%.4f")) {
            recorder->SetChangeThreshold(m_ChangeThreshold);
        }
        ImGui::Unindent();
    }
    
    // Interpolation mode
    const char* interpModes[] = {"Linear", "Cubic (Catmull-Rom)"};
    if (ImGui::Combo("Interpolation", &m_InterpolationModeIndex, interpModes, 2)) {
        recorder->SetInterpolationMode((InterpolationMode)m_InterpolationModeIndex);
    }
    
    ImGui::EndDisabled();
}

void SoftBodyRecordingPanel::RenderStreamingSettings(PhysXSoftBody* softBody)
{
    auto recorder = softBody->GetRecorder();
    bool isRecording = recorder->IsRecording();
    
    ImGui::SeparatorText("Streaming");
    
    ImGui::BeginDisabled(isRecording);
    
    if (ImGui::Checkbox("Enable Streaming", &m_StreamingMode)) {
        recorder->SetStreamingMode(m_StreamingMode, m_MemoryBudgetMB);
    }
    
    if (m_StreamingMode) {
        ImGui::Indent();
        if (ImGui::SliderInt("Memory Budget", &m_MemoryBudgetMB, 50, 500, "%d MB")) {
            recorder->SetStreamingMode(true, m_MemoryBudgetMB);
        }
        ImGui::Unindent();
    }
    
    ImGui::EndDisabled();
}

void SoftBodyRecordingPanel::RenderFileOperations(PhysXSoftBody* softBody)
{
    auto recorder = softBody->GetRecorder();
    
    ImGui::SeparatorText("File Operations");
    
    // Format selection
    const char* formats[] = {"JSON", "Binary", "Streaming"};
    ImGui::Combo("Format", &m_FileFormatIndex, formats, 3);
    
    // Save
    ImGui::InputText("Save File", m_SaveFilename, sizeof(m_SaveFilename));
    if (ImGui::Button("Save", ImVec2(100, 0))) {
        if (m_FileFormatIndex == 2) {
            // Streaming save with progress
            recorder->SaveStreaming(m_SaveFilename, 
                [this](int current, int total) {
                    m_SaveProgress = (float)current / total;
                });
            m_SaveProgress = 0.0f;
        } else {
            recorder->SaveToFile(m_SaveFilename, m_FileFormatIndex == 1);
        }
    }
    
    // Load
    ImGui::InputText("Load File", m_LoadFilename, sizeof(m_LoadFilename));
    if (ImGui::Button("Load", ImVec2(100, 0))) {
        if (m_FileFormatIndex == 2) {
            recorder->LoadStreaming(m_LoadFilename);
        } else {
            recorder->LoadFromFile(m_LoadFilename);
        }
    }
    
    // Progress bar
    if (m_SaveProgress > 0.0f && m_SaveProgress < 1.0f) {
        ImGui::ProgressBar(m_SaveProgress, ImVec2(-1, 0), "Saving...");
    }
}

void SoftBodyRecordingPanel::RenderStatistics(PhysXSoftBody* softBody)
{
    auto recorder = softBody->GetRecorder();
    
    if (!ImGui::TreeNode("Statistics")) {
        return;
    }
    
    // Basic stats
    ImGui::Text("Keyframes: %d", recorder->GetKeyframeCount());
    ImGui::Text("Duration: %.2fs", recorder->GetDuration());
    
    // Compression stats
    if (m_CompressionModeIndex != 0 || m_PartialRecording) {
        auto compStats = recorder->GetCompressionStats();
        ImGui::Separator();
        ImGui::Text("Compression Stats:");
        ImGui::Text("  Ratio: %.1f%%", compStats.compressionRatio * 100.0f);
        ImGui::Text("  Uncompressed: %.2f MB", compStats.uncompressedSize / 1024.0f / 1024.0f);
        ImGui::Text("  Compressed: %.2f MB", compStats.compressedSize / 1024.0f / 1024.0f);
        
        if (m_CompressionModeIndex == 1) {
            ImGui::Text("  Reference Frames: %d", compStats.referenceFrameCount);
            ImGui::Text("  Delta Frames: %d", compStats.deltaFrameCount);
        }
    }
    
    // Partial recording stats
    if (m_PartialRecording) {
        auto partialStats = recorder->GetPartialRecordingStats();
        ImGui::Separator();
        ImGui::Text("Partial Recording Stats:");
        ImGui::Text("  Avg Changed: %.1f%%", partialStats.averageChangedRatio * 100.0f);
        ImGui::Text("  Sparse Frames: %d", partialStats.sparseFrameCount);
        ImGui::Text("  Dense Frames: %d", partialStats.denseFrameCount);
    }
    
    // Streaming stats
    if (m_StreamingMode) {
        auto streamStats = recorder->GetStreamingStats();
        ImGui::Separator();
        ImGui::Text("Streaming Stats:");
        ImGui::Text("  Total Chunks: %zu", streamStats.totalChunks);
        ImGui::Text("  Loaded Chunks: %zu", streamStats.loadedChunks);
        ImGui::Text("  Cache Hit Ratio: %.1f%%", streamStats.cacheHitRatio * 100.0f);
        ImGui::Text("  Memory Usage: %.2f MB", streamStats.memoryUsage / 1024.0f / 1024.0f);
        ImGui::Text("  Memory Budget: %.2f MB", streamStats.memoryBudget / 1024.0f / 1024.0f);
    }
    
    ImGui::TreePop();
}

void SoftBodyRecordingPanel::ApplySettings(PhysXSoftBody* softBody)
{
    auto recorder = softBody->GetRecorder();
    
    recorder->SetCompressionMode((CompressionMode)m_CompressionModeIndex);
    recorder->SetReferenceFrameInterval(m_ReferenceInterval);
    recorder->SetPartialRecording(m_PartialRecording);
    recorder->SetChangeThreshold(m_ChangeThreshold);
    recorder->SetInterpolationMode((InterpolationMode)m_InterpolationModeIndex);
    recorder->SetStreamingMode(m_StreamingMode, m_MemoryBudgetMB);
    recorder->SetMaxKeyframes(m_MaxKeyframes);
}

const char* SoftBodyRecordingPanel::GetCompressionModeName(int index) const
{
    const char* names[] = {"None", "Delta Encoding", "Quantized"};
    return (index >= 0 && index < 3) ? names[index] : "Unknown";
}

const char* SoftBodyRecordingPanel::GetLoopModeName(int index) const
{
    const char* names[] = {"None", "Loop", "PingPong"};
    return (index >= 0 && index < 3) ? names[index] : "Unknown";
}

const char* SoftBodyRecordingPanel::GetInterpolationModeName(int index) const
{
    const char* names[] = {"Linear", "Cubic"};
    return (index >= 0 && index < 2) ? names[index] : "Unknown";
}

const char* SoftBodyRecordingPanel::GetFileFormatName(int index) const
{
    const char* names[] = {"JSON", "Binary", "Streaming"};
    return (index >= 0 && index < 3) ? names[index] : "Unknown";
}

void SoftBodyRecordingPanel::RenderMultiBodySection()
{
    ImGui::SeparatorText("Multi-Body Recording");
    
    if (!ImGui::TreeNode("Multi-Body Controls")) {
        return;
    }
    
    // Body count and status
    int bodyCount = m_MultiBodyRecorder->GetBodyCount();
    ImGui::Text("Bodies in Group: %d", bodyCount);
    
    if (bodyCount > 0) {
        ImGui::SameLine();
        if (m_MultiBodyRecorder->IsRecording()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "(Recording)");
        } else if (m_MultiBodyRecorder->IsPlayingBack()) {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "(Playing)");
        }
    }
    
    ImGui::Spacing();
    
    // Add body section
    if (ImGui::CollapsingHeader("Add Body to Group", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputText("Body Name", m_NewBodyName, sizeof(m_NewBodyName));
        
        if (ImGui::Button("Add Current Body", ImVec2(150, 0))) {
            if (strlen(m_NewBodyName) > 0) {
                // Note: Would need reference to current soft body from parent
                // This is a simplified version
                ImGui::OpenPopup("Need Soft Body");
            }
        }
        
        if (ImGui::BeginPopup("Need Soft Body")) {
            ImGui::Text("Please select a soft body first");
            ImGui::EndPopup();
        }
    }
    
    ImGui::Spacing();
    
    // Body list
    if (bodyCount > 0 && ImGui::CollapsingHeader("Bodies in Group", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto names = m_MultiBodyRecorder->GetBodyNames();
        
        for (size_t i = 0; i < names.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            
            bool selected = (m_SelectedBodyIndex == static_cast<int>(i));
            if (ImGui::Selectable(names[i].c_str(), selected)) {
                m_SelectedBodyIndex = static_cast<int>(i);
            }
            
            ImGui::SameLine(200);
            if (ImGui::SmallButton("Remove")) {
                m_MultiBodyRecorder->RemoveSoftBody(names[i]);
                m_SelectedBodyIndex = -1;
            }
            
            ImGui::PopID();
        }
    }
    
    ImGui::Spacing();
    
    // Group recording controls
    if (bodyCount > 0) {
        ImGui::SeparatorText("Group Controls");
        
        bool isRecording = m_MultiBodyRecorder->IsRecording();
        bool isPlaying = m_MultiBodyRecorder->IsPlayingBack();
        
        // Recording
        if (!isRecording && !isPlaying) {
            if (ImGui::Button("Start Group Recording", ImVec2(180, 0))) {
                m_MultiBodyRecorder->StartRecording(m_SampleRate);
            }
        } else if (isRecording) {
            if (ImGui::Button("Stop Group Recording", ImVec2(180, 0))) {
                m_MultiBodyRecorder->StopRecording();
            }
        }
        
        ImGui::Spacing();
        
        // Playback
        if (!isRecording) {
            if (!isPlaying) {
                if (ImGui::Button("Start Group Playback", ImVec2(180, 0))) {
                    m_MultiBodyRecorder->StartPlayback();
                }
            } else {
                if (ImGui::Button("Stop Group Playback", ImVec2(180, 0))) {
                    m_MultiBodyRecorder->StopPlayback();
                }
            }
            
            // Timeline
            float time = m_MultiBodyRecorder->GetPlaybackTime();
            float duration = m_MultiBodyRecorder->GetDuration();
            if (duration > 0.0f) {
                if (ImGui::SliderFloat("Group Timeline", &time, 0.0f, duration, "%.2fs")) {
                    m_MultiBodyRecorder->SeekPlayback(time);
                }
            }
            
            // Speed
            float speed = m_MultiBodyRecorder->GetPlaybackSpeed();
            if (ImGui::SliderFloat("Group Speed", &speed, 0.1f, 10.0f, "%.1fx")) {
                m_MultiBodyRecorder->SetPlaybackSpeed(speed);
            }
        }
        
        ImGui::Spacing();
        
        // File operations
        ImGui::SeparatorText("Group File Operations");
        
        static char groupFilename[256] = "scene.multibody";
        ImGui::InputText("Group File", groupFilename, sizeof(groupFilename));
        
        if (ImGui::Button("Save Group", ImVec2(120, 0))) {
            m_MultiBodyRecorder->SaveToFile(groupFilename);
        }
        
        ImGui::SameLine();
        
        if (ImGui::Button("Load Group", ImVec2(120, 0))) {
            m_MultiBodyRecorder->LoadFromFile(groupFilename);
        }
        
        ImGui::Spacing();
        
        // Group settings
        if (ImGui::CollapsingHeader("Group Settings")) {
            const char* compModes[] = {"None", "Delta Encoding", "Quantized"};
            int compMode = 0;
            if (ImGui::Combo("Compression", &compMode, compModes, 3)) {
                m_MultiBodyRecorder->SetCompressionMode((CompressionMode)compMode);
            }
            
            const char* interpModes[] = {"Linear", "Cubic"};
            int interpMode = 0;
            if (ImGui::Combo("Interpolation", &interpMode, interpModes, 2)) {
                m_MultiBodyRecorder->SetInterpolationMode((InterpolationMode)interpMode);
            }
        }
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                          "Add soft bodies to enable group recording");
    }
    
    ImGui::TreePop();
}
