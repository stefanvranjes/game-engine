#pragma once

#include "GameObject.h"
#include "Math/Vec2.h"
#include "Easing.h"
#include <map>
#include <string>
#include <vector>
#include <set>
#include <functional>

class Sprite : public GameObject {
public:
    // Callback types
    using AnimationCallback = std::function<void()>;
    using FrameCallback = std::function<void(int frame)>;
    using EventCallback = std::function<void(const std::string& eventName)>;
    using TransitionCallback = std::function<void(const std::string& fromSequence, const std::string& toSequence)>;
    using TransitionUpdateCallback = std::function<void(float progress)>;


    struct AnimationEvent {
        int frameNumber;
        std::string eventName;
        EventCallback callback;
        
        AnimationEvent()
            : frameNumber(0) {}
        
        AnimationEvent(int frame, const std::string& name, EventCallback cb)
            : frameNumber(frame), eventName(name), callback(cb) {}
    };

    struct AnimationSequence {
        std::string name;
        int startFrame;
        int endFrame;
        float speed;  // Frames per second (0 = use sprite's default speed)
        bool loop;
        std::vector<AnimationEvent> events;
        
        AnimationSequence()
            : startFrame(0), endFrame(0), speed(0.0f), loop(true) {}
        
        AnimationSequence(const std::string& n, int start, int end, float spd = 0.0f, bool lp = true)
            : name(n), startFrame(start), endFrame(end), speed(spd), loop(lp) {}
    };

    struct TransitionState {
        bool isTransitioning;
        std::string fromSequence;
        std::string toSequence;
        float duration;          // Total transition duration
        float elapsed;           // Time elapsed in transition
        int fromFrame;           // Frame to blend from
        float fromTime;          // Animation time in old sequence
        Easing::EaseType easeType; // Easing curve for transition
        
        TransitionState()
            : isTransitioning(false), duration(0.0f), elapsed(0.0f), fromFrame(0), 
              fromTime(0.0f), easeType(Easing::EaseType::Linear) {}
    };

    Sprite(const std::string& name = "Sprite");
    virtual ~Sprite();

    virtual void Update(const Mat4& parentMatrix, float deltaTime) override;

    // Animation control
    void Play();
    void Stop();
    void Pause();
    void SetAtlas(int rows, int cols);
    void SetAnimationSpeed(float speed) { m_AnimationSpeed = speed; }
    void SetLoop(bool loop) { m_Loop = loop; }
    
    // Animation sequences
    void AddSequence(const std::string& name, int startFrame, int endFrame, float speed = 0.0f, bool loop = true);
    void PlaySequence(const std::string& name);
    void PlaySequenceWithTransition(const std::string& name, float transitionDuration = -1.0f);
    void PlaySequenceWithTransition(const std::string& name, float transitionDuration, Easing::EaseType easeType);
    bool HasSequence(const std::string& name) const;
    std::string GetCurrentSequence() const { return m_CurrentSequence; }
    
    // Transition control
    void SetDefaultTransitionDuration(float duration) { m_DefaultTransitionDuration = duration; }
    float GetDefaultTransitionDuration() const { return m_DefaultTransitionDuration; }
    void SetDefaultEaseType(Easing::EaseType type) { m_DefaultEaseType = type; }
    Easing::EaseType GetDefaultEaseType() const { return m_DefaultEaseType; }
    bool IsTransitioning() const { return m_TransitionState.isTransitioning; }
    
    // Transition blend modes
    enum class TransitionBlendMode {
        FadeOnly,        // Fade opacity only (legacy behavior)
        CrossBlend       // Blend between frame UVs
    };
    
    void SetTransitionBlendMode(TransitionBlendMode mode) { m_TransitionBlendMode = mode; }
    TransitionBlendMode GetTransitionBlendMode() const { return m_TransitionBlendMode; }

    // Visual properties
    void SetBlendMode(Material::BlendMode mode);
    
    // Animation events/callbacks
    void SetOnComplete(AnimationCallback callback) { m_OnComplete = callback; }
    void SetOnLoop(AnimationCallback callback) { m_OnLoop = callback; }
    void SetOnFrameChange(FrameCallback callback) { m_OnFrameChange = callback; }
    
    // Transition callbacks
    void SetOnTransitionStart(TransitionCallback callback) { m_OnTransitionStart = callback; }
    void SetOnTransitionUpdate(TransitionUpdateCallback callback) { m_OnTransitionUpdate = callback; }
    void SetOnTransitionComplete(TransitionCallback callback) { m_OnTransitionComplete = callback; }
    
    // Frame-based animation events
    void AddEventToSequence(const std::string& sequenceName, int frame, const std::string& eventName, EventCallback callback);
    void ClearEventsForSequence(const std::string& sequenceName);
    
    bool IsPlaying() const { return m_IsPlaying; }
    int GetCurrentFrame() const { return m_CurrentFrame; }

private:
    void UpdateAnimation(float deltaTime);
    void UpdateUVs();
    void UpdateOldSequenceFrame(float deltaTime);
    void CalculateFrameUVs(int frame, Vec2& offset, Vec2& scale) const;

    int m_AtlasRows;
    int m_AtlasCols;
    float m_AnimationSpeed;  // Default frames per second
    bool m_Loop;             // Default loop setting
    float m_CurrentTime;
    int m_CurrentFrame;
    bool m_IsPlaying;
    
    // Animation sequences
    std::map<std::string, AnimationSequence> m_Sequences;
    std::string m_CurrentSequence;  // Empty string = no sequence (use full atlas)
    
    // Animation callbacks
    AnimationCallback m_OnComplete;
    AnimationCallback m_OnLoop;
    FrameCallback m_OnFrameChange;
    
    // Transition callbacks
    TransitionCallback m_OnTransitionStart;
    TransitionUpdateCallback m_OnTransitionUpdate;
    TransitionCallback m_OnTransitionComplete;
    
    // Event tracking
    std::set<int> m_TriggeredEventsThisFrame;
    int m_LastProcessedFrame;
    
    // Transition blending
    TransitionState m_TransitionState;
    float m_DefaultTransitionDuration;  // Default: 0.3 seconds
    Easing::EaseType m_DefaultEaseType; // Default easing curve
    TransitionBlendMode m_TransitionBlendMode;
};
