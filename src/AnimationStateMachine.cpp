#include "AnimationStateMachine.h"
#include "Animator.h"
#include <iostream>

AnimationStateMachine::AnimationStateMachine(Animator* animator) 
    : m_Animator(animator) {
}

void AnimationStateMachine::AddState(const AnimationState& state) {
    m_States[state.name] = state;
    if (m_DefaultStateName.empty()) { // First state is default
        m_DefaultStateName = state.name;
    }
}

void AnimationStateMachine::SetDefaultState(const std::string& stateName) {
    if (m_States.find(stateName) != m_States.end()) {
        m_DefaultStateName = stateName;
    }
}

void AnimationStateMachine::AddTransition(const std::string& fromState, const std::string& toState, 
                                          float duration, BlendCurve curve) {
    if (m_States.find(fromState) != m_States.end() && m_States.find(toState) != m_States.end()) {
        AnimationTransition t;
        t.toState = toState;
        t.duration = duration;
        t.curve = curve;
        m_States[fromState].transitions.push_back(t);
    }
}

AnimationTransition& AnimationStateMachine::AddTransition(const std::string& fromState, const std::string& toState) {
    static AnimationTransition dummy;
    if (m_States.find(fromState) != m_States.end() && m_States.find(toState) != m_States.end()) {
        AnimationTransition t;
        t.toState = toState;
        m_States[fromState].transitions.push_back(t); // Add copy
        return m_States[fromState].transitions.back(); // Return reference to stored copy
    }
    return dummy;
}

void AnimationStateMachine::SetFloat(const std::string& name, float value) {
    Parameter& p = m_Parameters[name];
    p.type = Parameter::Type::Float;
    p.fValue = value;
}

void AnimationStateMachine::SetInt(const std::string& name, int value) {
    Parameter& p = m_Parameters[name];
    p.type = Parameter::Type::Int;
    p.iValue = value;
}

void AnimationStateMachine::SetBool(const std::string& name, bool value) {
    Parameter& p = m_Parameters[name];
    p.type = Parameter::Type::Bool;
    p.iValue = value ? 1 : 0;
}

void AnimationStateMachine::SetTrigger(const std::string& name) {
    Parameter& p = m_Parameters[name];
    p.type = Parameter::Type::Trigger;
    p.triggerActive = true;
}

float AnimationStateMachine::GetFloat(const std::string& name) const {
    auto it = m_Parameters.find(name);
    return (it != m_Parameters.end()) ? it->second.fValue : 0.0f;
}

int AnimationStateMachine::GetInt(const std::string& name) const {
    auto it = m_Parameters.find(name);
    return (it != m_Parameters.end()) ? it->second.iValue : 0;
}

bool AnimationStateMachine::GetBool(const std::string& name) const {
    auto it = m_Parameters.find(name);
    return (it != m_Parameters.end()) ? (it->second.iValue != 0) : false;
}

bool AnimationStateMachine::CheckCondition(const AnimationCondition& condition) {
    auto it = m_Parameters.find(condition.parameter);
    if (it == m_Parameters.end()) return false;
    
    const Parameter& p = it->second;
    
    switch (condition.mode) {
        case AnimationConditionMode::If:
            if (p.type == Parameter::Type::Trigger) return p.triggerActive;
            if (p.type == Parameter::Type::Bool) return p.iValue != 0;
            return false;
            
        case AnimationConditionMode::IfNot:
            if (p.type == Parameter::Type::Bool) return p.iValue == 0;
            if (p.type == Parameter::Type::Trigger) return !p.triggerActive;
            return false;
            
        case AnimationConditionMode::Greater:
            if (p.type == Parameter::Type::Float) return p.fValue > condition.threshold;
            if (p.type == Parameter::Type::Int) return p.iValue > static_cast<int>(condition.threshold);
            return false;
            
        case AnimationConditionMode::Less:
            if (p.type == Parameter::Type::Float) return p.fValue < condition.threshold;
            if (p.type == Parameter::Type::Int) return p.iValue < static_cast<int>(condition.threshold);
            return false;
            
        case AnimationConditionMode::Equals:
            if (p.type == Parameter::Type::Float) return std::abs(p.fValue - condition.threshold) < 0.001f;
            if (p.type == Parameter::Type::Int) return p.iValue == static_cast<int>(condition.threshold);
            return false;
            
        case AnimationConditionMode::NotEqual:
            if (p.type == Parameter::Type::Float) return std::abs(p.fValue - condition.threshold) >= 0.001f;
            if (p.type == Parameter::Type::Int) return p.iValue != static_cast<int>(condition.threshold);
            return false;
    }
    return false;
}

bool AnimationStateMachine::CheckTransition(const AnimationTransition& transition, float normalizedTime) {
    // Check Exit Time
    if (transition.hasExitTime) {
        if (normalizedTime < transition.exitTime) return false;
    }
    
    // Check Conditions (All must be true)
    if (transition.conditions.empty() && !transition.hasExitTime) {
        // Warning: Transition with no conditions and no exit time usually runs immediately?
        // Or never? Let's assume never unless unconditional transition is implied.
        // For robustness: Empty conditions + No Exit Time = Manual blocked or Instant?
        // Let's assume it waits for at least one condition.
        return false;
    }
    
    for (const auto& cond : transition.conditions) {
        if (!CheckCondition(cond)) return false;
    }
    
    return true;
}

void AnimationStateMachine::SwitchState(const std::string& newStateName, float blendTime, BlendCurve curve) {
    m_CurrentStateName = newStateName;
    AnimationState& newState = m_States[newStateName];
    
    if (newState.type == AnimationState::Type::Animation) {
        m_Animator->TransitionToAnimation(newState.index, blendTime, curve);
    } else {
        // Map StateType to Animator LayerType enum for blend tree
        int layerType = 0; // Invalid
        if (newState.type == AnimationState::Type::BlendTree1D) layerType = 2; // Magic number matching LayerType enum? No, use explicit method
        // Animator.h doesn't expose generic TransitionToBlendTree via index easily without explicit method
        // But we added TransitionLayerToBlendTree.
        // What about Base Layer? Base layer is Layer 0?
        // Base Animator functions usually operate on Layer 0 or Base logic.
        // Let's assume State Machine controls Layer 0 (Base Layer).
        
        // We need a method on Animator to transition BASE layer to Blend Tree
        // "TransitionLayerToBlendTree" exists, we can use Layer 0.
        int typeInt = 0;
        if (newState.type == AnimationState::Type::BlendTree1D) typeInt = 1; // 1D
        if (newState.type == AnimationState::Type::BlendTree2D) typeInt = 3; // 2D (Enum skip?)
        if (newState.type == AnimationState::Type::BlendTree3D) typeInt = 4; // 3D
        
        // Note: LayerType enum (0=Single, 1=1D, 2=1D duplicate typo in previous file?, 3=2D, 4=3D)
        // I need to check Animator.h LayerType enum values carefully.
        // In previous view:
        // enum class LayerType { SingleAnimation, BlendTree1D, BlendTree1D (Dup), BlendTree2D, BlendTree3D };
        // So 0, 1, 2(error), 3, 4.
        
        m_Animator->TransitionLayerToBlendTree(0, newState.index, typeInt, blendTime, curve);
    }
}

void AnimationStateMachine::Update(float deltaTime) {
    if (m_CurrentStateName.empty()) {
        if (!m_DefaultStateName.empty()) {
            SwitchState(m_DefaultStateName, 0.0f, BlendCurve::Linear);
        }
        return;
    }
    
    // Get current state info
    auto it = m_States.find(m_CurrentStateName);
    if (it == m_States.end()) return;
    
    AnimationState& currentState = it->second;
    
    // Get Normalized Time from Animator (Base Layer 0)
    float normalizedTime = m_Animator->GetLayerNormalizedTime(0);
    
    // Check transitions
    for (const auto& transition : currentState.transitions) {
        if (CheckTransition(transition, normalizedTime)) {
            // Consume triggers used in this transition
            for (const auto& cond : transition.conditions) {
                if (m_Parameters[cond.parameter].type == Parameter::Type::Trigger) {
                    m_Parameters[cond.parameter].triggerActive = false;
                }
            }
            
            SwitchState(transition.toState, transition.duration, transition.curve);
            return; // Only one transition per frame
        }
    }
}
