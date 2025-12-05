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
    , m_LastProcessedFrame(-1)
    , m_DefaultTransitionDuration(0.3f)
    , m_DefaultEaseType(Easing::EaseType::EaseInOutQuad)
    , m_TransitionBlendMode(TransitionBlendMode::CrossBlend)
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
    
    // Handle transition blending
    if (m_TransitionState.isTransitioning) {
        m_TransitionState.elapsed += deltaTime;
        
        // Calculate blend factor (0.0 = old, 1.0 = new)
        float t = m_TransitionState.elapsed / m_TransitionState.duration;
        float blendFactor = Easing::Ease(t, m_TransitionState.easeType);
        
        // Trigger update callback
        if (m_OnTransitionUpdate) {
            m_OnTransitionUpdate(blendFactor);
        }
        
        if (t >= 1.0f) {
            // Transition complete
            m_TransitionState.isTransitioning = false;
            
            // Store sequences before cleanup
            std::string fromSeq = m_TransitionState.fromSequence;
            std::string toSeq = m_TransitionState.toSequence;
            
            // Ensure we're showing the new sequence frame
            UpdateUVs();
            
            // Restore full opacity
            auto material = GetMaterial();
            if (material) {
                material->SetMyAlpha(1.0f);
            }
            
            // Trigger complete callback
            if (m_OnTransitionComplete) {
                m_OnTransitionComplete(fromSeq, toSeq);
            }
        } else {
            // Cross-blend between sequences
            if (m_TransitionBlendMode == TransitionBlendMode::CrossBlend) {
                // Update old sequence animation
                UpdateOldSequenceFrame(deltaTime);
                
                // Calculate UV coordinates for both frames
                Vec2 fromUVOffset, fromUVScale;
                Vec2 toUVOffset, toUVScale;
                
                CalculateFrameUVs(m_TransitionState.fromFrame, fromUVOffset, fromUVScale);
                CalculateFrameUVs(m_CurrentFrame, toUVOffset, toUVScale);
                
                // Interpolate UVs based on blend factor
                Vec2 blendedOffset(
                    fromUVOffset.x * (1.0f - blendFactor) + toUVOffset.x * blendFactor,
                    fromUVOffset.y * (1.0f - blendFactor) + toUVOffset.y * blendFactor
                );
                
                // Set blended UV coordinates
                SetUVOffset(blendedOffset);
                SetUVScale(fromUVScale); // Assume same scale for both sequences
                
                // Keep full opacity during cross-blend
                auto material = GetMaterial();
                if (material) {
                    material->SetMyAlpha(1.0f);
                }
            } else {
                // FadeOnly mode: current behavior (fade opacity)
                UpdateUVs(); // Update to current frame
                auto material = GetMaterial();
                if (material) {
                    material->SetMyAlpha(blendFactor);
                }
            }
        }
    }
    
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
    
    // Trigger frame-based events if frame changed
    if (m_CurrentFrame != m_LastProcessedFrame) {
        m_TriggeredEventsThisFrame.clear();
        m_LastProcessedFrame = m_CurrentFrame;
        
        // Check for events in current sequence
        if (!m_CurrentSequence.empty()) {
            auto it = m_Sequences.find(m_CurrentSequence);
            if (it != m_Sequences.end()) {
                const AnimationSequence& seq = it->second;
                for (const AnimationEvent& event : seq.events) {
                    if (event.frameNumber == m_CurrentFrame) {
                        // Check if we haven't already triggered this event this frame
                        if (m_TriggeredEventsThisFrame.find(event.frameNumber) == m_TriggeredEventsThisFrame.end()) {
                            if (event.callback) {
                                event.callback(event.eventName);
                            }
                            m_TriggeredEventsThisFrame.insert(event.frameNumber);
                        }
                    }
                }
            }
        }
    }
    
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
    m_TriggeredEventsThisFrame.clear();
    m_LastProcessedFrame = -1;
    
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
    m_TriggeredEventsThisFrame.clear();
    m_LastProcessedFrame = -1;
    UpdateUVs();
}

bool Sprite::HasSequence(const std::string& name) const {
    return m_Sequences.find(name) != m_Sequences.end();
}

void Sprite::PlaySequenceWithTransition(const std::string& name, float transitionDuration) {
    PlaySequenceWithTransition(name, transitionDuration, m_DefaultEaseType);
}

void Sprite::PlaySequenceWithTransition(const std::string& name, float transitionDuration, Easing::EaseType easeType) {
    // If already playing this sequence, do nothing
    if (m_CurrentSequence == name && !m_TransitionState.isTransitioning) {
        return;
    }
    
    // Use default duration if not specified
    if (transitionDuration < 0.0f) {
        transitionDuration = m_DefaultTransitionDuration;
    }
    
    // If not transitioning and no current sequence, or instant transition, just play normally
    if (m_CurrentSequence.empty() || transitionDuration <= 0.0f) {
        PlaySequence(name);
        return;
    }
    
    // Validate new sequence exists
    auto it = m_Sequences.find(name);
    if (it == m_Sequences.end()) {
        std::cerr << "Warning: Animation sequence '" << name << "' not found!" << std::endl;
        return;
    }
    
    // Start transition
    m_TransitionState.isTransitioning = true;
    m_TransitionState.fromSequence = m_CurrentSequence;
    m_TransitionState.toSequence = name;
    m_TransitionState.duration = transitionDuration;
    m_TransitionState.elapsed = 0.0f;
    m_TransitionState.fromFrame = m_CurrentFrame;
    m_TransitionState.fromTime = m_CurrentTime;  // Capture current animation time
    m_TransitionState.easeType = easeType;
    
    m_CurrentSequence = name;
    m_CurrentTime = 0.0f;
    m_CurrentFrame = it->second.startFrame;
    m_TriggeredEventsThisFrame.clear();
    m_LastProcessedFrame = -1;
    
    // Start with low opacity, will fade in
    auto material = GetMaterial();
    if (material) {
        material->SetMyAlpha(0.0f);
    }
    
    // Trigger transition start callback
    if (m_OnTransitionStart) {
        m_OnTransitionStart(m_TransitionState.fromSequence, m_TransitionState.toSequence);
    }
}

void Sprite::SetBlendMode(Material::BlendMode mode) {
    auto material = GetMaterial();
    if (material) {
        material->SetBlendMode(mode);
    }
}

void Sprite::AddEventToSequence(const std::string& sequenceName, int frame, const std::string& eventName, EventCallback callback) {
    auto it = m_Sequences.find(sequenceName);
    if (it == m_Sequences.end()) {
        std::cerr << "Warning: Cannot add event to sequence '" << sequenceName << "' - sequence not found!" << std::endl;
        return;
    }
    
    AnimationSequence& seq = it->second;
    
    // Validate frame is within sequence range
    if (frame < seq.startFrame || frame > seq.endFrame) {
        std::cerr << "Warning: Event frame " << frame << " is outside sequence '" << sequenceName 
                  << "' range [" << seq.startFrame << ", " << seq.endFrame << "]" << std::endl;
        return;
    }
    
    // Add the event
    seq.events.emplace_back(frame, eventName, callback);
}

void Sprite::ClearEventsForSequence(const std::string& sequenceName) {
    auto it = m_Sequences.find(sequenceName);
    if (it == m_Sequences.end()) {
        std::cerr << "Warning: Cannot clear events for sequence '" << sequenceName << "' - sequence not found!" << std::endl;
        return;
    }
    
    it->second.events.clear();
}

void Sprite::UpdateOldSequenceFrame(float deltaTime) {
    if (m_TransitionState.fromSequence.empty()) return;
    
    auto it = m_Sequences.find(m_TransitionState.fromSequence);
    if (it == m_Sequences.end()) return;
    
    const AnimationSequence& seq = it->second;
    float speed = seq.speed > 0.0f ? seq.speed : m_AnimationSpeed;
    
    m_TransitionState.fromTime += deltaTime * speed;
    
    int frameCount = seq.endFrame - seq.startFrame + 1;
    if (frameCount <= 0) return;
    
    int frameIndex = static_cast<int>(m_TransitionState.fromTime);
    
    if (seq.loop) {
        frameIndex = frameIndex % frameCount;
    } else {
        frameIndex = std::min(frameIndex, frameCount - 1);
    }
    
    m_TransitionState.fromFrame = seq.startFrame + frameIndex;
}

void Sprite::CalculateFrameUVs(int frame, Vec2& offset, Vec2& scale) const {
    if (m_AtlasRows <= 0 || m_AtlasCols <= 0) return;
    
    int row = frame / m_AtlasCols;
    int col = frame % m_AtlasCols;
    
    scale = Vec2(1.0f / m_AtlasCols, 1.0f / m_AtlasRows);
    offset = Vec2(
        col / static_cast<float>(m_AtlasCols),
        1.0f - (row + 1) / static_cast<float>(m_AtlasRows)
    );
}

