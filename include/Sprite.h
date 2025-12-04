#pragma once

#include "GameObject.h"
#include "Math/Vec2.h"
#include <map>
#include <string>
#include <functional>

class Sprite : public GameObject {
public:
    // Callback types
    using AnimationCallback = std::function<void()>;
    using FrameCallback = std::function<void(int frame)>;

    struct AnimationSequence {
        std::string name;
        int startFrame;
        int endFrame;
        float speed;  // Frames per second (0 = use sprite's default speed)
        bool loop;
        
        AnimationSequence()
            : startFrame(0), endFrame(0), speed(0.0f), loop(true) {}
        
        AnimationSequence(const std::string& n, int start, int end, float spd = 0.0f, bool lp = true)
            : name(n), startFrame(start), endFrame(end), speed(spd), loop(lp) {}
    };

    Sprite(const std::string& name = "Sprite");
    virtual ~Sprite();

    virtual void Update(const Mat4& parentMatrix) override;
    void Update(const Mat4& parentMatrix, float deltaTime);

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
    bool HasSequence(const std::string& name) const;
    std::string GetCurrentSequence() const { return m_CurrentSequence; }
    
    // Animation events/callbacks
    void SetOnComplete(AnimationCallback callback) { m_OnComplete = callback; }
    void SetOnLoop(AnimationCallback callback) { m_OnLoop = callback; }
    void SetOnFrameChange(FrameCallback callback) { m_OnFrameChange = callback; }
    
    bool IsPlaying() const { return m_IsPlaying; }
    int GetCurrentFrame() const { return m_CurrentFrame; }

private:
    void UpdateAnimation(float deltaTime);
    void UpdateUVs();

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
};
