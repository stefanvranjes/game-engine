#include "MultiBodyRecorder.h"
#include "PhysXSoftBody.h"
#include <fstream>
#include <algorithm>

MultiBodyRecorder::MultiBodyRecorder()
    : m_MasterTime(0.0f)
    , m_SampleRate(60.0f)
    , m_PlaybackSpeed(1.0f)
    , m_IsRecording(false)
    , m_IsPlaying(false)
    , m_IsPaused(false)
    , m_LoopMode(LoopMode::None)
{
}

MultiBodyRecorder::~MultiBodyRecorder()
{
    Clear();
}

bool MultiBodyRecorder::AddSoftBody(PhysXSoftBody* softBody, const std::string& name)
{
    if (!softBody || name.empty()) {
        return false;
    }
    
    // Check if name already exists
    if (m_NameToIndex.find(name) != m_NameToIndex.end()) {
        return false;
    }
    
    BodyEntry entry;
    entry.softBody = softBody;
    entry.recorder = std::make_unique<SoftBodyDeformationRecorder>();
    entry.name = name;
    
    // Initialize recorder with soft body's vertex count
    entry.recorder->Initialize(softBody->GetVertexCount());
    
    int index = static_cast<int>(m_Bodies.size());
    m_Bodies.push_back(std::move(entry));
    m_NameToIndex[name] = index;
    
    return true;
}

bool MultiBodyRecorder::RemoveSoftBody(const std::string& name)
{
    int index = FindBodyIndex(name);
    if (index < 0) {
        return false;
    }
    
    m_Bodies.erase(m_Bodies.begin() + index);
    
    // Rebuild name-to-index map
    m_NameToIndex.clear();
    for (size_t i = 0; i < m_Bodies.size(); ++i) {
        m_NameToIndex[m_Bodies[i].name] = static_cast<int>(i);
    }
    
    return true;
}

void MultiBodyRecorder::Clear()
{
    m_Bodies.clear();
    m_NameToIndex.clear();
    m_MasterTime = 0.0f;
    m_IsRecording = false;
    m_IsPlaying = false;
}

void MultiBodyRecorder::StartRecording(float sampleRate)
{
    m_SampleRate = sampleRate;
    m_MasterTime = 0.0f;
    m_IsRecording = true;
    m_IsPaused = false;
    
    // Start all recorders with same settings
    for (auto& entry : m_Bodies) {
        entry.recorder->StartRecording(sampleRate, false);
    }
}

void MultiBodyRecorder::StopRecording()
{
    m_IsRecording = false;
    m_IsPaused = false;
    
    for (auto& entry : m_Bodies) {
        entry.recorder->StopRecording();
    }
}

void MultiBodyRecorder::PauseRecording()
{
    if (m_IsRecording) {
        m_IsPaused = true;
        for (auto& entry : m_Bodies) {
            entry.recorder->PauseRecording();
        }
    }
}

void MultiBodyRecorder::ResumeRecording()
{
    if (m_IsRecording && m_IsPaused) {
        m_IsPaused = false;
        for (auto& entry : m_Bodies) {
            entry.recorder->ResumeRecording();
        }
    }
}

void MultiBodyRecorder::StartPlayback()
{
    m_MasterTime = 0.0f;
    m_IsPlaying = true;
    m_IsPaused = false;
    
    for (auto& entry : m_Bodies) {
        entry.recorder->StartPlayback();
    }
}

void MultiBodyRecorder::StopPlayback()
{
    m_IsPlaying = false;
    m_IsPaused = false;
    
    for (auto& entry : m_Bodies) {
        entry.recorder->StopPlayback();
    }
}

void MultiBodyRecorder::PausePlayback()
{
    if (m_IsPlaying) {
        m_IsPaused = true;
        for (auto& entry : m_Bodies) {
            entry.recorder->PausePlayback();
        }
    }
}

void MultiBodyRecorder::SeekPlayback(float time)
{
    m_MasterTime = time;
    
    for (auto& entry : m_Bodies) {
        entry.recorder->SeekPlayback(time);
    }
}

void MultiBodyRecorder::SetPlaybackSpeed(float speed)
{
    m_PlaybackSpeed = speed;
    
    for (auto& entry : m_Bodies) {
        entry.recorder->SetPlaybackSpeed(speed);
    }
}

void MultiBodyRecorder::SetLoopMode(LoopMode mode)
{
    m_LoopMode = mode;
    
    for (auto& entry : m_Bodies) {
        entry.recorder->SetLoopMode(mode);
    }
}

void MultiBodyRecorder::Update(float deltaTime)
{
    if (m_IsPaused) return;
    
    if (m_IsRecording) {
        m_MasterTime += deltaTime;
        
        // Record all bodies with synchronized timestamp
        for (auto& entry : m_Bodies) {
            const Vec3* positions = entry.softBody->GetVertexPositions();
            const Vec3* velocities = entry.softBody->GetVertexVelocities();
            entry.recorder->RecordFrame(positions, velocities, m_MasterTime);
        }
    }
    
    if (m_IsPlaying) {
        m_MasterTime += deltaTime * m_PlaybackSpeed;
        
        // Handle looping
        float duration = GetDuration();
        if (m_MasterTime > duration) {
            if (m_LoopMode == LoopMode::Loop) {
                m_MasterTime = 0.0f;
                SeekPlayback(0.0f);
            } else if (m_LoopMode == LoopMode::PingPong) {
                m_PlaybackSpeed = -m_PlaybackSpeed;
                m_MasterTime = duration;
            } else {
                StopPlayback();
            }
        } else if (m_MasterTime < 0.0f && m_LoopMode == LoopMode::PingPong) {
            m_PlaybackSpeed = -m_PlaybackSpeed;
            m_MasterTime = 0.0f;
        }
        
        // Update all bodies to same time
        for (auto& entry : m_Bodies) {
            entry.recorder->UpdatePlayback(deltaTime * m_PlaybackSpeed);
        }
    }
}

bool MultiBodyRecorder::SaveToFile(const std::string& filename) const
{
    nlohmann::json j;
    j["version"] = 1;
    j["bodyCount"] = m_Bodies.size();
    j["sampleRate"] = m_SampleRate;
    
    nlohmann::json bodies = nlohmann::json::array();
    for (const auto& entry : m_Bodies) {
        nlohmann::json bodyData;
        bodyData["name"] = entry.name;
        bodyData["vertexCount"] = entry.softBody->GetVertexCount();
        bodyData["recording"] = entry.recorder->ToJson();
        bodies.push_back(bodyData);
    }
    j["bodies"] = bodies;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << j.dump(2);
    return true;
}

bool MultiBodyRecorder::LoadFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    nlohmann::json j;
    file >> j;
    
    if (j["version"] != 1) {
        return false;
    }
    
    Clear();
    
    m_SampleRate = j["sampleRate"];
    
    for (const auto& bodyData : j["bodies"]) {
        std::string name = bodyData["name"];
        int vertexCount = bodyData["vertexCount"];
        
        // Note: We can't restore PhysXSoftBody pointers from file
        // User must manually associate loaded recorders with soft bodies
        BodyEntry entry;
        entry.softBody = nullptr;
        entry.name = name;
        entry.recorder = std::make_unique<SoftBodyDeformationRecorder>();
        entry.recorder->Initialize(vertexCount);
        entry.recorder->FromJson(bodyData["recording"]);
        
        int index = static_cast<int>(m_Bodies.size());
        m_Bodies.push_back(std::move(entry));
        m_NameToIndex[name] = index;
    }
    
    return true;
}

float MultiBodyRecorder::GetDuration() const
{
    float maxDuration = 0.0f;
    
    for (const auto& entry : m_Bodies) {
        float duration = entry.recorder->GetDuration();
        if (duration > maxDuration) {
            maxDuration = duration;
        }
    }
    
    return maxDuration;
}

std::vector<std::string> MultiBodyRecorder::GetBodyNames() const
{
    std::vector<std::string> names;
    names.reserve(m_Bodies.size());
    
    for (const auto& entry : m_Bodies) {
        names.push_back(entry.name);
    }
    
    return names;
}

SoftBodyDeformationRecorder* MultiBodyRecorder::GetRecorder(const std::string& name)
{
    int index = FindBodyIndex(name);
    if (index < 0) {
        return nullptr;
    }
    
    return m_Bodies[index].recorder.get();
}

void MultiBodyRecorder::SetCompressionMode(CompressionMode mode)
{
    for (auto& entry : m_Bodies) {
        entry.recorder->SetCompressionMode(mode);
    }
}

void MultiBodyRecorder::SetInterpolationMode(InterpolationMode mode)
{
    for (auto& entry : m_Bodies) {
        entry.recorder->SetInterpolationMode(mode);
    }
}

int MultiBodyRecorder::FindBodyIndex(const std::string& name) const
{
    auto it = m_NameToIndex.find(name);
    if (it == m_NameToIndex.end()) {
        return -1;
    }
    return it->second;
}
