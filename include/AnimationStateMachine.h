#pragma once

#include "BlendCurve.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>

class Animator;

// ============================================================================
// Data Structures
// ============================================================================

enum class AnimationConditionMode {
    If,
    IfNot,
    Greater,
    Less,
    Equals,
    NotEqual
};

struct AnimationCondition {
    std::string parameter;
    AnimationConditionMode mode;
    float threshold; // For float comparison
    
    AnimationCondition(std::string param, AnimationConditionMode m, float th = 0.0f)
        : parameter(param), mode(m), threshold(th) {}
};

struct AnimationTransition {
    std::string toState;
    std::vector<AnimationCondition> conditions;
    
    float duration;
    BlendCurve curve;
    bool hasExitTime;
    float exitTime; // Normalized time (0-1)
    
    AnimationTransition() 
        : duration(0.2f), curve(BlendCurve::SmoothStep), hasExitTime(false), exitTime(0.9f) {}
};

struct AnimationState {
    std::string name;
    
    // Content (Single Animation or Blend Tree)
    enum class Type { Animation, BlendTree1D, BlendTree2D, BlendTree3D };
    Type type;
    int index; // Index into Animator's animation or blend tree arrays
    
    bool loop;
    float speed;
    
    std::vector<AnimationTransition> transitions;
    
    AnimationState(std::string n, int idx, Type t = Type::Animation) 
        : name(n), type(t), index(idx), loop(true), speed(1.0f) {}
};

// ============================================================================
// State Machine
// ============================================================================

class AnimationStateMachine {
public:
    AnimationStateMachine(Animator* animator);
    
    // Setup
    void AddState(const AnimationState& state);
    void SetDefaultState(const std::string& stateName);
    
    void AddTransition(const std::string& fromState, const std::string& toState, 
                       float duration = 0.2f, BlendCurve curve = BlendCurve::SmoothStep);
                       
    // Advanced transition setup (manual)
    AnimationTransition& AddTransition(const std::string& fromState, const std::string& toState);
    
    // Runtime
    void Update(float deltaTime);
    
    // Parameters
    void SetFloat(const std::string& name, float value);
    void SetInt(const std::string& name, int value);
    void SetBool(const std::string& name, bool value);
    void SetTrigger(const std::string& name);
    
    float GetFloat(const std::string& name) const;
    int GetInt(const std::string& name) const;
    bool GetBool(const std::string& name) const;
    
public:
    // Internal Parameter Storage
    struct Parameter {
        enum class Type { Float, Int, Bool, Trigger };
        Type type;
        
        float fValue;
        int iValue; // also stores bool as 0/1
        bool triggerActive;
        
        Parameter() : type(Type::Float), fValue(0), iValue(0), triggerActive(false) {}
    };
    
private:
    Animator* m_Animator;
    
    std::unordered_map<std::string, AnimationState> m_States;
    std::string m_CurrentStateName;
    std::string m_DefaultStateName;
    
    std::unordered_map<std::string, Parameter> m_Parameters;
    
    // Helpers
    bool CheckCondition(const AnimationCondition& condition);
    bool CheckTransition(const AnimationTransition& transition, float normalizedTime);
    void SwitchState(const std::string& newStateName, float blendTime, BlendCurve curve);
};
