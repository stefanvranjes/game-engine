#include "Animation.h"
#include "Bone.h"
#include "BlendCurve.h"
#include "Math/Mat4.h"
#include "Math/Vec2.h"
#include "Math/Vec3.h"
#include <memory>
#include <vector>
#include "IK.h"
#include "AnimationStateMachine.h"
#include "BoneMask.h"

class BlendTree1D;
class BlendTree2D;
class BlendTree3D;

class Animator {
public:
    Animator();
    ~Animator();
    
    // Set skeleton
    void SetSkeleton(std::shared_ptr<Skeleton> skeleton);
    std::shared_ptr<Skeleton> GetSkeleton() const { return m_Skeleton; }
    
    // Animation management
    void AddAnimation(std::shared_ptr<Animation> animation);
    void PlayAnimation(int index, bool loop = true);
    void PlayAnimation(const std::string& name, bool loop = true);
    void Stop();
    void Pause();
    void Resume();
    
    // Update animation state
    void Update(float deltaTime);
    
    // Get final bone matrices for rendering
    const std::vector<Mat4>& GetBoneMatrices() const { return m_FinalBoneMatrices; }
    
    // State queries
    bool IsPlaying() const { return m_IsPlaying; }
    bool IsPaused() const { return m_IsPaused; }
    float GetCurrentTime() const { return m_CurrentTime; }
    
    // Playback control
    void SetPlaybackSpeed(float speed) { m_PlaybackSpeed = speed; }
    float GetPlaybackSpeed() const { return m_PlaybackSpeed; }
    
    // Animation blending/transitions
    void TransitionToAnimation(int animationIndex, float blendTime = -1.0f, BlendCurve curve = BlendCurve::Linear);
    void TransitionToAnimation(const std::string& name, float blendTime = -1.0f, BlendCurve curve = BlendCurve::Linear);
    void SetDefaultBlendTime(float time) { m_DefaultBlendTime = time; }
    float GetDefaultBlendTime() const { return m_DefaultBlendTime; }
    void SetDefaultBlendCurve(BlendCurve curve) { m_DefaultBlendCurve = curve; }
    BlendCurve GetDefaultBlendCurve() const { return m_DefaultBlendCurve; }
    bool IsBlending() const;
    float GetBlendProgress() const;
    
    // Get animation count
    int GetAnimationCount() const { return static_cast<int>(m_Animations.size()); }
    
    // Animation layers (partial blending)
    int AddLayer(int animationIndex, const class BoneMask& mask, float weight = 1.0f, bool additive = false);
    void RemoveLayer(int layerIndex);
    void SetLayerWeight(int layerIndex, float weight);
    void SetLayerAnimation(int layerIndex, int animationIndex);
    void SetLayerMask(int layerIndex, const class BoneMask& mask);
    int GetLayerCount() const;
    
    // Add Blend Tree Layers
    int AddBlendTreeLayer1D(int treeIndex, const class BoneMask& mask, float weight = 1.0f, bool additive = false);
    int AddBlendTreeLayer2D(int treeIndex, const class BoneMask& mask, float weight = 1.0f, bool additive = false);
    int AddBlendTreeLayer3D(int treeIndex, const class BoneMask& mask, float weight = 1.0f, bool additive = false);
    
    // Layer transitions (blend within a layer)
    void TransitionLayerToAnimation(int layerIndex, int animationIndex, float blendTime = -1.0f, BlendCurve curve = BlendCurve::Linear);
    void TransitionLayerToBlendTree(int layerIndex, int treeIndex, int type, float blendTime = -1.0f, BlendCurve curve = BlendCurve::Linear); // type cast from LayerType
    bool IsLayerBlending(int layerIndex) const;
    float GetLayerBlendProgress(int layerIndex) const;
    float GetLayerNormalizedTime(int layerIndex) const;
    
    // State Machine
    AnimationStateMachine* GetStateMachine() { return &m_StateMachine; }
    void UpdateStateMachine(float deltaTime) { m_StateMachine.Update(deltaTime); }
    
    // Convenience layer methods
    int AddUpperBodyLayer(int animationIndex, const std::string& spineBoneName, float weight = 1.0f);
    int AddAdditiveLayer(int animationIndex, const class BoneMask& mask, float weight = 1.0f);
    
    // Blend Tree management
    int CreateBlendTree1D();
    int CreateBlendTree2D();
    int CreateBlendTree3D();
    void AddBlendTreeNode1D(int treeIndex, int animationIndex, float parameter);
    void AddBlendTreeNode2D(int treeIndex, int animationIndex, Vec2 parameter);
    void AddBlendTreeNode3D(int treeIndex, int animationIndex, Vec3 parameter);
    void SetBlendTreeParameter1D(int treeIndex, float value);
    void SetBlendTreeParameterSmooth1D(int treeIndex, float value, float smoothTime);
    void SetBlendTreeParameter2D(int treeIndex, Vec2 value);
    void SetBlendTreeParameterSmooth2D(int treeIndex, Vec2 value, float smoothTime);
    void SetBlendTreeParameter3D(int treeIndex, Vec3 value);
    void SetBlendTreeParameterSmooth3D(int treeIndex, Vec3 value, float smoothTime);
    
    // Play blend tree on a layer
    // Overloads SetLayerAnimation to support blend trees
    void SetLayerBlendTree1D(int layerIndex, int treeIndex);
    void SetLayerBlendTree2D(int layerIndex, int treeIndex);
    void SetLayerBlendTree3D(int layerIndex, int treeIndex);
    

    
    // Editor Helpers
    int GetBlendTree1DCount() const { return static_cast<int>(m_BlendTrees1D.size()); }
    int GetBlendTree2DCount() const { return static_cast<int>(m_BlendTrees2D.size()); }
    int GetBlendTree3DCount() const { return static_cast<int>(m_BlendTrees3D.size()); }
    
    // Convenience Blend Tree methods
    int CreateLocomotionBlendTree1D(int idleAnim, int walkAnim, int runAnim);
    int CreateDirectionalBlendTree2D(const std::vector<int>& animIndices, const std::vector<Vec2>& positions);
    
    // Inverse Kinematics
    int AddIKChain(const std::string& rootBone, const std::string& effectorBone);
    void SetIKTarget(int chainIndex, Vec3 targetPos);
    void SetIKWeight(int chainIndex, float weight);
    
private:
    std::shared_ptr<Skeleton> m_Skeleton;
    std::vector<std::shared_ptr<Animation>> m_Animations;
    
    int m_CurrentAnimationIndex;
    float m_CurrentTime;
    bool m_IsPlaying;
    bool m_IsPaused;
    bool m_Loop;
    float m_PlaybackSpeed;
    
    // Bone transformation data
    std::vector<Mat4> m_LocalTransforms;   // Current local transforms for each bone
    std::vector<Mat4> m_GlobalTransforms;  // Global transforms in model space
    std::vector<Mat4> m_FinalBoneMatrices; // Final matrices for shader (global * inverseBindMatrix)
    
    // Blend state for animation transitions
    struct BlendState {
        bool isBlending;
        int fromAnimationIndex;
        int toAnimationIndex;
        float blendTime;
        float currentBlendTime;
        BlendCurve curve;
        
        BlendState() : isBlending(false), fromAnimationIndex(-1), toAnimationIndex(-1), 
                       blendTime(0.0f), currentBlendTime(0.0f), curve(BlendCurve::Linear) {}
    };
    
    BlendState m_BlendState;
    float m_DefaultBlendTime;
    BlendCurve m_DefaultBlendCurve;
    std::vector<Mat4> m_FromBoneMatrices;  // Cached matrices from source animation during blend
    
    // Animation layers for partial blending
    enum class LayerType {
        SingleAnimation,
        BlendTree1D,
        BlendTree2D,
        BlendTree3D
    };
    
    struct AnimationLayer {
        LayerType type;
        int animationIndex;        // Used if type == SingleAnimation
        int blendTreeIndex;        // Used if type == BlendTree...
        
        float currentTime;
        float weight;              // Overall layer weight [0,1]
        class BoneMask mask;      // Which bones this layer affects
        bool isAdditive;          // Additive or override mode
        bool isPlaying;
        bool loop;
        
        // Layer-specific blending
        // Layer-specific blending
        bool isBlending;
        
        // From state
        LayerType fromType;
        int fromAnimationIndex;
        int fromBlendTreeIndex;
        std::vector<Mat4> fromBoneMatrices;
        
        // To state
        LayerType toType;
        int toAnimationIndex;
        int toBlendTreeIndex;
        
        float blendTime;
        float currentBlendTime;
        BlendCurve blendCurve;
        
        AnimationLayer() : type(LayerType::SingleAnimation), animationIndex(-1), blendTreeIndex(-1), 
                          currentTime(0.0f), weight(1.0f), 
                          isAdditive(false), isPlaying(false), loop(true),
                          isBlending(false), 
                          fromType(LayerType::SingleAnimation), fromAnimationIndex(-1), fromBlendTreeIndex(-1),
                          toType(LayerType::SingleAnimation), toAnimationIndex(-1), toBlendTreeIndex(-1),
                          blendTime(0.0f), currentBlendTime(0.0f), blendCurve(BlendCurve::SmoothStep) {}
    };
    
    std::vector<AnimationLayer> m_Layers;
    std::vector<Mat4> m_LayerBoneMatrices;  // Temp storage for layer bone matrices
    
    // Blend Trees storage
    std::vector<std::unique_ptr<BlendTree1D>> m_BlendTrees1D;
    std::vector<std::unique_ptr<BlendTree2D>> m_BlendTrees2D;
    std::vector<std::unique_ptr<BlendTree3D>> m_BlendTrees3D;
    
    // Inverse Kinematics
    std::vector<IKChain> m_IKChains;

    // State Machine
    AnimationStateMachine m_StateMachine;
    
    // Update bone transforms based on current animation time
    void UpdateBoneTransforms();
    void UpdateLayerTransforms(AnimationLayer& layer, float deltaTime, std::vector<Mat4>& outMatrices);
    void BlendBoneMatrices(const std::vector<Mat4>& from, const std::vector<Mat4>& to, 
                           float t, std::vector<Mat4>& result);
    void BlendLayerMatrices(const std::vector<Mat4>& base, const std::vector<Mat4>& layer,
                           const class BoneMask& mask, float weight, bool additive, 
                           std::vector<Mat4>& result);
};

