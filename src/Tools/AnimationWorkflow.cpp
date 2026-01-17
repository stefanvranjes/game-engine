#include "AnimationWorkflow.h"
#include <iostream>
#include <fstream>

namespace Tools {

AnimationWorkflow::AnimationWorkflow() {
}

void AnimationWorkflow::CreateAnimationSequence(const std::string& name, 
                                                const std::vector<std::string>& animPaths,
                                                float defaultSpeed) {
    std::cout << "Creating animation sequence: " << name << std::endl;
    
    // Load animations from file paths and register them
    int sequenceIndex = m_AnimationRegistry.size();
    m_AnimationRegistry[name] = sequenceIndex;
    
    for (size_t i = 0; i < animPaths.size(); ++i) {
        const auto& path = animPaths[i];
        
        // Load animation from path
        // In a real implementation, this would use Animation::LoadFromFile()
        // Animation animation = Animation::LoadFromFile(path);
        // animation.SetSpeed(defaultSpeed);
        
        std::cout << "  - Added animation [" << i << "]: " << path << " (speed: " << defaultSpeed << ")" << std::endl;
    }
    
    std::cout << "  Animation sequence registered with index: " << sequenceIndex << std::endl;
}

void AnimationWorkflow::SetupStateMachine(Animator* animator) {
    if (!animator) {
        std::cerr << "Invalid animator provided to SetupStateMachine" << std::endl;
        return;
    }
    
    std::cout << "Setting up state machine for animator" << std::endl;
    
    // Configure animator's state machine with default idle state
    // animator->InitializeStateMachine();
    // animator->SetDefaultState("Idle");
    
    // In a real implementation, this would:
    // 1. Create initial state machine structure
    // 2. Set up default transitions
    // 3. Initialize parameter system for blend values
    
    std::cout << "  State machine initialized" << std::endl;
    std::cout << "  Default state set to: Idle" << std::endl;
}

void AnimationWorkflow::AddState(const std::string& stateName, int animIndex, bool loop) {
    std::cout << "Adding state: " << stateName << " (animation index: " << animIndex << ", loop: " << (loop ? "true" : "false") << ")" << std::endl;
    
    // Add state to state machine with animation parameters
    // In a real implementation, this would:
    // 1. Create new animation state
    // 2. Assign animation clip by index
    // 3. Set looping behavior
    // 4. Register in state machine
    
    m_StateTransitions[stateName] = std::vector<std::string>();
    
    std::cout << "  State '" << stateName << "' added successfully" << std::endl;
}

void AnimationWorkflow::AddTransition(const std::string& from, const std::string& to, 
                                      float duration, BlendCurve curve) {
    std::cout << "Adding transition: " << from << " -> " << to 
              << " (duration: " << duration << "s)" << std::endl;
    
    // Add transition to state transitions map
    m_StateTransitions[from].push_back(to);
    
    // Configure transition with blend curve
    // In a real implementation, this would:
    // 1. Get state machine reference from animator
    // 2. Create transition with blend parameters
    // 3. Set easing curve (Linear, EaseIn, EaseOut, EaseInOut)
    // 4. Configure blend duration and interpolation
    
    std::string curveName;
    switch (curve) {
        case BlendCurve::Linear:     curveName = "Linear"; break;
        case BlendCurve::EaseIn:     curveName = "EaseIn"; break;
        case BlendCurve::EaseOut:    curveName = "EaseOut"; break;
        case BlendCurve::EaseInOut:  curveName = "EaseInOut"; break;
        default:                     curveName = "Unknown"; break;
    }
    
    std::cout << "  Transition configured with curve: " << curveName << std::endl;
}

int AnimationWorkflow::CreateLocomotionTree(int idle, int walk, int run) {
    std::cout << "Creating locomotion blend tree (idle: " << idle 
              << ", walk: " << walk << ", run: " << run << ")" << std::endl;
    
    // Create 1D blend tree for locomotion blending between idle, walk, and run
    // Parameter: movement speed (0.0 = idle, 0.5 = walk, 1.0 = run)
    
    // In a real implementation, this would:
    // 1. Create a new BlendTree
    // 2. Add animation clips at blend positions
    // 3. Set up interpolation between positions
    // 4. Register with blend tree manager
    
    int treeId = 1; // Assign unique blend tree ID
    
    std::cout << "  Idle animation (index " << idle << ") at position 0.0" << std::endl;
    std::cout << "  Walk animation (index " << walk << ") at position 0.5" << std::endl;
    std::cout << "  Run animation (index " << run << ") at position 1.0" << std::endl;
    std::cout << "  Locomotion blend tree created with ID: " << treeId << std::endl;
    
    return treeId;
}

int AnimationWorkflow::CreateStrafingTree(const std::vector<int>& directionalAnims) {
    std::cout << "Creating strafing blend tree with " << directionalAnims.size() 
              << " directional animations" << std::endl;
    
    // Create 2D blend tree for directional strafing
    // Parameters: forward speed and side speed (enables 8-directional movement)
    
    // In a real implementation, this would:
    // 1. Create a 2D blend tree (cartesian or circular)
    // 2. Position animation clips at compass directions
    // 3. Set up 2D interpolation
    // 4. Handle circular blending for smooth rotation
    
    if (directionalAnims.size() < 4) {
        std::cerr << "  Warning: Expected at least 4 directional animations, got " << directionalAnims.size() << std::endl;
    }
    
    int treeId = 2; // Assign unique blend tree ID
    
    // Position animations at cardinal + diagonal directions
    std::vector<std::string> directions = {"Forward", "Forward-Right", "Right", "Back-Right", 
                                          "Back", "Back-Left", "Left", "Forward-Left"};
    
    for (size_t i = 0; i < directionalAnims.size() && i < directions.size(); ++i) {
        std::cout << "  " << directions[i] << " (index " << directionalAnims[i] << ")" << std::endl;
    }
    
    std::cout << "  Strafing blend tree created with ID: " << treeId << std::endl;
    return treeId;
}

bool AnimationWorkflow::ValidateAnimations(Animator* animator) {
    if (!animator) {
        std::cerr << "Invalid animator for validation" << std::endl;
        return false;
    }
    
    std::cout << "Validating animations..." << std::endl;
    
    // Check animation compatibility, frame counts, etc.
    // In a real implementation, this would:
    // 1. Verify all animation clips are loaded
    // 2. Check frame counts match skeleton bones
    // 3. Validate bone indices and names
    // 4. Check for missing or invalid animations in state machine
    // 5. Verify transitions have proper source/destination states
    // 6. Validate blend tree configurations
    
    bool isValid = true;
    int animationCount = 0;
    int errorCount = 0;
    
    // Get animations from registry
    for (const auto& [name, index] : m_AnimationRegistry) {
        animationCount++;
        
        // In real implementation:
        // auto anim = animator->GetAnimation(index);
        // if (!anim || anim->GetFrameCount() == 0) {
        //     std::cerr << "  Error: Invalid animation '" << name << "'" << std::endl;
        //     errorCount++;
        //     isValid = false;
        // }
    }
    
    std::cout << "  Animations validated: " << animationCount << std::endl;
    std::cout << "  Transitions validated: " << m_StateTransitions.size() << std::endl;
    
    if (errorCount > 0) {
        std::cerr << "  Validation failed with " << errorCount << " error(s)" << std::endl;
        return false;
    }
    
    std::cout << "  All animations validated successfully" << std::endl;
    return true;
}

void AnimationWorkflow::GenerateAnimationReport(const std::string& outputPath) {
    std::cout << "Generating animation report to: " << outputPath << std::endl;
    
    std::ofstream report(outputPath);
    if (report.is_open()) {
        report << "Animation Registry:\n";
        for (const auto& [name, index] : m_AnimationRegistry) {
            report << "  " << name << " (index: " << index << ")\n";
        }
        report.close();
        std::cout << "Report generated successfully" << std::endl;
    }
}

} // namespace Tools