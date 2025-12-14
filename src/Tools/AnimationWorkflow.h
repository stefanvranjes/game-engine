#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "../Animation/Animator.h"
#include "../Animation/BlendTreeEditor.h"

namespace Tools {

enum class BlendCurve {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut
};

class AnimationWorkflow {
public:
    AnimationWorkflow();
    
    // Animation sequence builder
    void CreateAnimationSequence(const std::string& name, 
                                 const std::vector<std::string>& animPaths,
                                 float defaultSpeed = 1.0f);
    
    // State machine builder
    void SetupStateMachine(Animator* animator);
    void AddState(const std::string& stateName, int animIndex, bool loop = true);
    void AddTransition(const std::string& from, const std::string& to, 
                       float duration = 0.2f, BlendCurve curve = BlendCurve::Linear);
    
    // Blend tree builder
    int CreateLocomotionTree(int idle, int walk, int run);
    int CreateStrafingTree(const std::vector<int>& directionalAnims);
    
    // Validation
    bool ValidateAnimations(Animator* animator);
    void GenerateAnimationReport(const std::string& outputPath);
    
private:
    std::map<std::string, int> m_AnimationRegistry;
    std::map<std::string, std::vector<std::string>> m_StateTransitions;
};

} // namespace Tools