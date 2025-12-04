#include "Sprite.h"
#include <cmath>
#include <iostream>

Sprite::Sprite(const std::string& name)
    : GameObject(name)
    , m_AtlasRows(1)
    , m_AtlasCols(1)
    , m_AnimationSpeed(10.0f)
    , m_Loop(true)
    , m_CurrentTime(0.0f)
    , m_CurrentFrame(0)
    , m_IsPlaying(false)
    , m_CurrentSequence("")
{
}

Sprite::~Sprite() {
}

void Sprite::Update(const Mat4& parentMatrix) {
    // Call base class update
    GameObject::Update(parentMatrix);
    
    // Use default deltaTime for backward compatibility
    float deltaTime = 1.0f / 60.0f;
    UpdateAnimation(deltaTime);
}

void Sprite::Update(const Mat4& parentMatrix, float deltaTime) {
    // Call base class update
    GameObject::Update(parentMatrix);
    
    // Use provided deltaTime
    UpdateAnimation(deltaTime);
}

void Sprite::UpdateAnimation(float deltaTime) {
    if (!m_IsPlaying) return;
    
    // Store previous frame for change detection
    int previousFrame = m_CurrentFrame;
    
    // Determine frame range and settings
    int startFrame = 0;
    int endFrame = (m_AtlasRows * m_AtlasCols) - 1;
    float speed = m_AnimationSpeed;
    bool loop = m_Loop;
    
    // If we're playing a sequence, use its settings
    if (!m_CurrentSequence.empty()) {
        auto it = m_Sequences.find(m_CurrentSequence);
        if (it != m_Sequences.end()) {
            const AnimationSequence& seq = it->second;
            startFrame = seq.startFrame;
            endFrame = seq.endFrame;
            loop = seq.loop;
            
            // Use sequence speed if specified, otherwise use default
            if (seq.speed > 0.0f) {
                speed = seq.speed;
            }
        }
    }
    
    // Calculate frame count for this range
    int frameCount = endFrame - startFrame + 1;
    if (frameCount <= 0) return;
    
    // Update animation time with actual deltaTime
    m_CurrentTime += deltaTime * speed;
    
    // Calculate frame within the sequence
    int frameIndex = static_cast<int>(m_CurrentTime);
    
    bool didLoop = false;
    bool didComplete = false;
    
    if (loop) {
        // Check if we looped
        int prevFrameIndex = static_cast<int>(m_CurrentTime - deltaTime * speed);
        if (frameIndex >= frameCount && prevFrameIndex < frameCount) {
            didLoop = true;
        }
        
        // Loop within the sequence range
        frameIndex = frameIndex % frameCount;
        m_CurrentFrame = startFrame + frameIndex;
    } else {
        // Clamp to sequence range
        if (frameIndex >= frameCount) {
            m_CurrentFrame = endFrame;
            m_IsPlaying = false;
            didComplete = true;
        } else {
            m_CurrentFrame = startFrame + frameIndex;
        }
    }
    
    UpdateUVs();
    
    // Trigger callbacks
    if (m_CurrentFrame != previousFrame && m_OnFrameChange) {
        m_OnFrameChange(m_CurrentFrame);
    }
    
    if (didLoop && m_OnLoop) {
        m_OnLoop();
    }
    
    if (didComplete && m_OnComplete) {
        m_OnComplete();
    }
}

void Sprite::UpdateUVs() {
    if (m_AtlasRows <= 0 || m_AtlasCols <= 0) return;
    
    // Calculate row and column from current frame
    int row = m_CurrentFrame / m_AtlasCols;
    int col = m_CurrentFrame % m_AtlasCols;
    
    // Calculate UV scale (size of one cell)
    Vec2 scale(1.0f / m_AtlasCols, 1.0f / m_AtlasRows);
    
    // Calculate UV offset (bottom-left corner of the cell)
    // OpenGL texture coordinates: (0,0) is bottom-left, (1,1) is top-right
    // Atlas is typically laid out top-to-bottom, so we need to invert the row
    Vec2 offset(
        col / static_cast<float>(m_AtlasCols),
        1.0f - (row + 1) / static_cast<float>(m_AtlasRows)
    );
    
    SetUVScale(scale);
    SetUVOffset(offset);
}

void Sprite::Play() {
    m_IsPlaying = true;
    m_CurrentTime = 0.0f;
    
    // Determine starting frame
    if (!m_CurrentSequence.empty()) {
        auto it = m_Sequences.find(m_CurrentSequence);
        if (it != m_Sequences.end()) {
            m_CurrentFrame = it->second.startFrame;
        } else {
            m_CurrentFrame = 0;
        }
    } else {
        m_CurrentFrame = 0;
    }
    
    UpdateUVs();
}

void Sprite::Stop() {
    m_IsPlaying = false;
    m_CurrentTime = 0.0f;
    m_CurrentFrame = 0;
    m_CurrentSequence = "";  // Clear sequence
    UpdateUVs();
}

void Sprite::Pause() {
    m_IsPlaying = false;
}

void Sprite::SetAtlas(int rows, int cols) {
    m_AtlasRows = rows;
    m_AtlasCols = cols;
    m_CurrentFrame = 0;
    m_CurrentTime = 0.0f;
    UpdateUVs();
}

void Sprite::AddSequence(const std::string& name, int startFrame, int endFrame, float speed, bool loop) {
    AnimationSequence seq(name, startFrame, endFrame, speed, loop);
    m_Sequences[name] = seq;
}

void Sprite::PlaySequence(const std::string& name) {
    auto it = m_Sequences.find(name);
    if (it == m_Sequences.end()) {
        std::cerr << "Warning: Animation sequence '" << name << "' not found!" << std::endl;
        return;
    }
    
    m_CurrentSequence = name;
    m_CurrentTime = 0.0f;
    m_CurrentFrame = it->second.startFrame;
    m_IsPlaying = true;
    UpdateUVs();
}

bool Sprite::HasSequence(const std::string& name) const {
    return m_Sequences.find(name) != m_Sequences.end();
}
