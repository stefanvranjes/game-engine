#include "Animator.h"
#include "BlendCurve.h"
#include "BoneMask.h"
#include "BlendTree.h"
#include "IK.h"
#include <iostream>

Animator::Animator() 
    : m_CurrentAnimationIndex(-1)
    , m_CurrentTime(0.0f)
    , m_IsPlaying(false)
    , m_IsPaused(false)
    , m_Loop(true)
    , m_PlaybackSpeed(1.0f)
    , m_DefaultBlendTime(0.2f)
    , m_DefaultBlendCurve(BlendCurve::Linear)
    , m_StateMachine(this) { // Pass this to state machine
}

Animator::~Animator() {
}

void Animator::SetSkeleton(std::shared_ptr<Skeleton> skeleton) {
    m_Skeleton = skeleton;
    
    if (m_Skeleton) {
        int boneCount = m_Skeleton->GetBoneCount();
        m_LocalTransforms.resize(boneCount);
        m_GlobalTransforms.resize(boneCount);
        m_FinalBoneMatrices.resize(boneCount);
        m_FromBoneMatrices.resize(boneCount);  // For blending
        
        // Initialize with identity matrices
        for (int i = 0; i < boneCount; ++i) {
            m_LocalTransforms[i] = Mat4::Identity();
            m_GlobalTransforms[i] = Mat4::Identity();
            m_FinalBoneMatrices[i] = Mat4::Identity();
            m_FromBoneMatrices[i] = Mat4::Identity();
        }
    }
}

void Animator::AddAnimation(std::shared_ptr<Animation> animation) {
    m_Animations.push_back(animation);
}

void Animator::PlayAnimation(int index, bool loop) {
    if (index < 0 || index >= static_cast<int>(m_Animations.size())) {
        std::cerr << "Invalid animation index: " << index << std::endl;
        return;
    }
    
    m_CurrentAnimationIndex = index;
    m_CurrentTime = 0.0f;
    m_IsPlaying = true;
    m_IsPaused = false;
    m_Loop = loop;
}

void Animator::PlayAnimation(const std::string& name, bool loop) {
    for (size_t i = 0; i < m_Animations.size(); ++i) {
        if (m_Animations[i]->GetName() == name) {
            PlayAnimation(static_cast<int>(i), loop);
            return;
        }
    }
    std::cerr << "Animation not found: " << name << std::endl;
}

void Animator::Stop() {
    m_IsPlaying = false;
    m_IsPaused = false;
    m_CurrentTime = 0.0f;
}

void Animator::Pause() {
    if (m_IsPlaying) {
        m_IsPaused = true;
    }
}

void Animator::Resume() {
    if (m_IsPaused) {
        m_IsPaused = false;
    }
}

void Animator::Update(float deltaTime) {
    if (!m_IsPlaying || m_IsPaused || !m_Skeleton) {
        return;
    }
    
    // Handle blending
    if (m_BlendState.isBlending) {
        m_BlendState.currentBlendTime += deltaTime;
        float t = m_BlendState.currentBlendTime / m_BlendState.blendTime;
        
        if (t >= 1.0f) {
            // Blend complete, switch to target animation
            m_BlendState.isBlending = false;
            m_CurrentAnimationIndex = m_BlendState.toAnimationIndex;
            t = 1.0f;
        }
        
        // Apply easing curve to blend weight
        float easedT = EasingFunctions::Apply(t, m_BlendState.curve);
        
        // Update time for both animations
        Animation* fromAnim = m_Animations[m_BlendState.fromAnimationIndex].get();
        Animation* toAnim = m_Animations[m_BlendState.toAnimationIndex].get();
        
        m_CurrentTime += deltaTime * m_PlaybackSpeed;
        
        // Handle looping for target animation
        float toDuration = toAnim->GetDuration();
        if (m_CurrentTime >= toDuration) {
            if (m_Loop) {
                m_CurrentTime = fmod(m_CurrentTime, toDuration);
            } else {
                m_CurrentTime = toDuration;
                if (!m_BlendState.isBlending) {
                    m_IsPlaying = false;
                }
            }
        }
        
        // Determine sampling times for both animations
        float fromSamplingTime = m_CurrentTime;
        float toSamplingTime = m_CurrentTime;
        
        // Sync Logic: If animations are in the same SyncGroup, align their phases
        if (!fromAnim->GetSyncGroup().empty() && fromAnim->GetSyncGroup() == toAnim->GetSyncGroup()) {
            float fromDuration = fromAnim->GetDuration();
            
            if (fromDuration > 0.001f && toDuration > 0.001f) {
                // Calculate normalized phase of the source animation
                float fromPhase = fmod(fromSamplingTime, fromDuration) / fromDuration;
                
                // Set target time to match that phase
                toSamplingTime = fromPhase * toDuration;
            }
        }
        
        // Calculate bone matrices for both animations
        int boneCount = m_Skeleton->GetBoneCount();
        std::vector<Mat4> toBoneMatrices(boneCount);
        
        // Calculate matrices for 'from' animation
        for (int i = 0; i < boneCount; ++i) {
            const AnimationChannel* channel = fromAnim->GetChannelForBone(i);
            if (channel) {
                Vec3 position;
                Quaternion rotation;
                Vec3 scale;
                channel->GetTransform(fromSamplingTime, position, rotation, scale);
                
                Mat4 translation = Mat4::Translation(position);
                Mat4 rotationMat = rotation.ToMatrix();
                Mat4 scaleMat = Mat4::Scale(scale);
                m_LocalTransforms[i] = translation * rotationMat * scaleMat;
            } else {
                m_LocalTransforms[i] = m_Skeleton->GetBone(i).localTransform;
            }
        }
        
        m_Skeleton->CalculateGlobalTransforms(m_LocalTransforms, m_GlobalTransforms);
        for (int i = 0; i < boneCount; ++i) {
            m_FromBoneMatrices[i] = m_GlobalTransforms[i] * m_Skeleton->GetBone(i).inverseBindMatrix;
        }
        
        // Calculate matrices for 'to' animation
        for (int i = 0; i < boneCount; ++i) {
            const AnimationChannel* channel = toAnim->GetChannelForBone(i);
            if (channel) {
                Vec3 position;
                Quaternion rotation;
                Vec3 scale;
                channel->GetTransform(toSamplingTime, position, rotation, scale);
                
                Mat4 translation = Mat4::Translation(position);
                Mat4 rotationMat = rotation.ToMatrix();
                Mat4 scaleMat = Mat4::Scale(scale);
                m_LocalTransforms[i] = translation * rotationMat * scaleMat;
            } else {
                m_LocalTransforms[i] = m_Skeleton->GetBone(i).localTransform;
            }
        }
        
        m_Skeleton->CalculateGlobalTransforms(m_LocalTransforms, m_GlobalTransforms);
        for (int i = 0; i < boneCount; ++i) {
            toBoneMatrices[i] = m_GlobalTransforms[i] * m_Skeleton->GetBone(i).inverseBindMatrix;
        }
        
        // Blend the matrices using eased blend weight
        BlendBoneMatrices(m_FromBoneMatrices, toBoneMatrices, easedT, m_FinalBoneMatrices);
        
    } else {
        // Normal single animation update
        if (m_CurrentAnimationIndex < 0) {
            return;
        }
        
        Animation* currentAnim = m_Animations[m_CurrentAnimationIndex].get();
        float duration = currentAnim->GetDuration();
        
        // Update time
        m_CurrentTime += deltaTime * m_PlaybackSpeed;
        
        // Handle looping
        if (m_CurrentTime >= duration) {
            if (m_Loop) {
                m_CurrentTime = fmod(m_CurrentTime, duration);
            } else {
                m_CurrentTime = duration;
                m_IsPlaying = false;
            }
        }
        
        // Update bone transforms
        UpdateBoneTransforms();
        
        // Process animation layers
        if (!m_Layers.empty() && m_Skeleton) {
            int boneCount = m_Skeleton->GetBoneCount();
            m_LayerBoneMatrices.resize(boneCount);
            std::vector<Mat4> layerBlendedMatrices(boneCount);
            
            for (auto& layer : m_Layers) {
                if (!layer.isPlaying) {
                    continue;
                }
                
                // Handle layer-specific blending
                if (layer.isBlending) {
                    layer.currentBlendTime += deltaTime;
                    float t = layer.currentBlendTime / layer.blendTime;
                    
                    if (t >= 1.0f) {
                        // Layer blend complete
                        layer.isBlending = false;
                        layer.type = layer.toType;
                        if (layer.toType == LayerType::SingleAnimation) {
                            layer.animationIndex = layer.toAnimationIndex;
                        } else {
                            layer.blendTreeIndex = layer.toBlendTreeIndex;
                        }
                        t = 1.0f;
                    }
                    
                    // Apply easing curve
                    float easedT = EasingFunctions::Apply(t, layer.blendCurve);
                    
                    // Update time for target
                    // Note: BlendTrees track their own time inside Update(), but we might need
                    // to advance a dummy time if we want to support looping/duration checks easily for trees?
                    // Usually BlendTrees just run indefinitely or loop internally.
                    // For SingleAnimation target:
                    if (layer.toType == LayerType::SingleAnimation) {
                        Animation* toAnim = m_Animations[layer.toAnimationIndex].get();
                        layer.currentTime += deltaTime * m_PlaybackSpeed;
                        
                        float toDuration = toAnim->GetDuration();
                        if (layer.currentTime >= toDuration) {
                            if (layer.loop) {
                                layer.currentTime = fmod(layer.currentTime, toDuration);
                            } else {
                                layer.currentTime = toDuration;
                            }
                        }
                    }
                    
                    // Calculate matrices for target animation/tree
                    AnimationLayer tempLayer = layer;
                    tempLayer.type = layer.toType;
                    if (layer.toType == LayerType::SingleAnimation) {
                        tempLayer.animationIndex = layer.toAnimationIndex;
                        // Time already updated above for SingleAnim
                    } else {
                        tempLayer.blendTreeIndex = layer.toBlendTreeIndex;
                        // BlendTrees need the delta time passed to UpdateLayerTransforms to advance internal state
                        // BUT we don't want to advance state twice (once here, and once if it becomes active next frame).
                        // Since this is a temporary layer for blending, it's fine.
                    }
                    
                    // We pass deltaTime only if it's a BlendTree so it updates its nodes
                    float updateDelta = (layer.toType != LayerType::SingleAnimation) ? (deltaTime * m_PlaybackSpeed) : 0.0f;
                    UpdateLayerTransforms(tempLayer, updateDelta, m_LayerBoneMatrices);
                    
                    // Blend from cached 'from' matrices to 'to' matrices
                    BlendBoneMatrices(layer.fromBoneMatrices, m_LayerBoneMatrices, 
                                     easedT, layerBlendedMatrices);
                    
                } else {
                    // Normal layer update (no blending)
                    
                    if (layer.type == LayerType::SingleAnimation) {
                        if (layer.animationIndex < 0) {
                            continue;
                        }
                        
                        // Update layer time
                        Animation* layerAnim = m_Animations[layer.animationIndex].get();
                        layer.currentTime += deltaTime * m_PlaybackSpeed;
                        
                        // Handle looping
                        float layerDuration = layerAnim->GetDuration();
                        if (layer.currentTime >= layerDuration) {
                            if (layer.loop) {
                                layer.currentTime = fmod(layer.currentTime, layerDuration);
                            } else {
                                layer.currentTime = layerDuration;
                                layer.isPlaying = false;
                            }
                        }
                        
                        // Calculate layer bone matrices
                        UpdateLayerTransforms(layer, 0.0f, layerBlendedMatrices);
                    }
                    else {
                        // Blend Tree update
                        // Pass deltaTime so the tree can advance its internal time
                        UpdateLayerTransforms(layer, deltaTime * m_PlaybackSpeed, layerBlendedMatrices);
                    }
                }
                
                // Blend layer with current result
                BlendLayerMatrices(m_FinalBoneMatrices, layerBlendedMatrices,
                                  layer.mask, layer.weight, layer.isAdditive,
                                  m_FinalBoneMatrices);
            }
        }
    }
}

void Animator::UpdateBoneTransforms() {
    if (!m_Skeleton || m_CurrentAnimationIndex < 0) {
        return;
    }
    
    Animation* currentAnim = m_Animations[m_CurrentAnimationIndex].get();
    int boneCount = m_Skeleton->GetBoneCount();
    
    // Update local transforms from animation channels
    for (int i = 0; i < boneCount; ++i) {
        const AnimationChannel* channel = currentAnim->GetChannelForBone(i);
        
        if (channel) {
            Vec3 position;
            Quaternion rotation;
            Vec3 scale;
            channel->GetTransform(m_CurrentTime, position, rotation, scale);
            
            // Build transformation matrix: T * R * S
            Mat4 translation = Mat4::Translation(position);
            Mat4 rotationMat = rotation.ToMatrix();
            Mat4 scaleMat = Mat4::Scale(scale);
            
            m_LocalTransforms[i] = translation * rotationMat * scaleMat;
        } else {
            // No animation for this bone, use bind pose
            m_LocalTransforms[i] = m_Skeleton->GetBone(i).localTransform;
        }
    }
    
    // Calculate global transforms
    m_Skeleton->CalculateGlobalTransforms(m_LocalTransforms, m_GlobalTransforms);
    
    // Apply Inverse Kinematics
    for (auto& chain : m_IKChains) {
        if (chain.weight > 0.001f) {
            IKSolver::SolveFABRIK(chain, m_Skeleton.get(), m_GlobalTransforms);
        }
    }
    
    // Calculate final bone matrices (global * inverseBindMatrix)
    for (int i = 0; i < boneCount; ++i) {
        m_FinalBoneMatrices[i] = m_GlobalTransforms[i] * m_Skeleton->GetBone(i).inverseBindMatrix;
    }
}

void Animator::TransitionToAnimation(int animationIndex, float blendTime, BlendCurve curve) {
    if (animationIndex < 0 || animationIndex >= static_cast<int>(m_Animations.size())) {
        std::cerr << "Invalid animation index for transition: " << animationIndex << std::endl;
        return;
    }
    
    // If already transitioning to this animation, ignore
    if (m_BlendState.isBlending && m_BlendState.toAnimationIndex == animationIndex) {
        return;
    }
    
    // Use defaults if not specified
    if (blendTime < 0.0f) {
        blendTime = m_DefaultBlendTime;
    }
    if (curve == BlendCurve::Linear && m_DefaultBlendCurve != BlendCurve::Linear) {
        curve = m_DefaultBlendCurve;
    }
    
    // If not currently playing, just start the animation directly
    if (!m_IsPlaying || m_CurrentAnimationIndex < 0) {
        PlayAnimation(animationIndex, m_Loop);
        return;
    }
    
    // Setup blend state
    m_BlendState.isBlending = true;
    m_BlendState.fromAnimationIndex = m_CurrentAnimationIndex;
    m_BlendState.toAnimationIndex = animationIndex;
    m_BlendState.blendTime = blendTime;
    m_BlendState.currentBlendTime = 0.0f;
    m_BlendState.curve = curve;
    
    // Cache current bone matrices as 'from' state
    m_FromBoneMatrices = m_FinalBoneMatrices;
    
    std::cout << "Starting blend from animation " << m_BlendState.fromAnimationIndex 
              << " to " << m_BlendState.toAnimationIndex 
              << " over " << blendTime << "s with curve " << static_cast<int>(curve) << std::endl;
}

void Animator::TransitionToAnimation(const std::string& name, float blendTime, BlendCurve curve) {
    for (size_t i = 0; i < m_Animations.size(); ++i) {
        if (m_Animations[i]->GetName() == name) {
            TransitionToAnimation(static_cast<int>(i), blendTime, curve);
            return;
        }
    }
    std::cerr << "Animation not found for transition: " << name << std::endl;
}

bool Animator::IsBlending() const {
    return m_BlendState.isBlending;
}

float Animator::GetBlendProgress() const {
    if (!m_BlendState.isBlending || m_BlendState.blendTime <= 0.0f) {
        return 1.0f;
    }
    return std::min(1.0f, m_BlendState.currentBlendTime / m_BlendState.blendTime);
}

void Animator::BlendBoneMatrices(const std::vector<Mat4>& from, const std::vector<Mat4>& to, 
                                  float t, std::vector<Mat4>& result) {
    size_t count = std::min(from.size(), to.size());
    if (result.size() < count) {
        result.resize(count);
    }
    
    for (size_t i = 0; i < count; ++i) {
        // Decompose matrices to TRS
        Vec3 fromPos, toPos, fromScale, toScale;
        Quaternion fromRot, toRot;
        
        DecomposeMatrix(from[i], fromPos, fromRot, fromScale);
        DecomposeMatrix(to[i], toPos, toRot, toScale);
        
        // Interpolate components
        Vec3 blendPos;
        blendPos.x = fromPos.x + t * (toPos.x - fromPos.x);
        blendPos.y = fromPos.y + t * (toPos.y - fromPos.y);
        blendPos.z = fromPos.z + t * (toPos.z - fromPos.z);
        
        Quaternion blendRot = Quaternion::Slerp(fromRot, toRot, t);
        
        Vec3 blendScale;
        blendScale.x = fromScale.x + t * (toScale.x - fromScale.x);
        blendScale.y = fromScale.y + t * (toScale.y - fromScale.y);
        blendScale.z = fromScale.z + t * (toScale.z - fromScale.z);
        
        // Recompose matrix
        result[i] = ComposeMatrix(blendPos, blendRot, blendScale);
    }
}

// Animation layer management
int Animator::AddLayer(int animationIndex, const BoneMask& mask, float weight, bool additive) {
    if (animationIndex < 0 || animationIndex >= static_cast<int>(m_Animations.size())) {
        std::cerr << "Invalid animation index for layer: " << animationIndex << std::endl;
        return -1;
    }
    
    AnimationLayer layer;
    layer.animationIndex = animationIndex;
    layer.mask = mask;
    layer.weight = std::max(0.0f, std::min(1.0f, weight));
    layer.isAdditive = additive;
    layer.isPlaying = true;
    layer.loop = true;
    layer.currentTime = 0.0f;
    
    m_Layers.push_back(layer);
    
    std::cout << "Added animation layer " << (m_Layers.size() - 1) 
              << " with animation " << animationIndex 
              << " (" << (additive ? "additive" : "override") << ")" << std::endl;
    
    return static_cast<int>(m_Layers.size()) - 1;
}

void Animator::RemoveLayer(int layerIndex) {
    if (layerIndex >= 0 && layerIndex < static_cast<int>(m_Layers.size())) {
        m_Layers.erase(m_Layers.begin() + layerIndex);
        std::cout << "Removed animation layer " << layerIndex << std::endl;
    }
}

void Animator::SetLayerWeight(int layerIndex, float weight) {
    if (layerIndex >= 0 && layerIndex < static_cast<int>(m_Layers.size())) {
        m_Layers[layerIndex].weight = std::max(0.0f, std::min(1.0f, weight));
    }
}

void Animator::SetLayerAnimation(int layerIndex, int animationIndex) {
    if (layerIndex >= 0 && layerIndex < static_cast<int>(m_Layers.size())) {
        if (animationIndex >= 0 && animationIndex < static_cast<int>(m_Animations.size())) {
            m_Layers[layerIndex].animationIndex = animationIndex;
            m_Layers[layerIndex].currentTime = 0.0f;
        }
    }
}

void Animator::SetLayerMask(int layerIndex, const BoneMask& mask) {
    if (layerIndex >= 0 && layerIndex < static_cast<int>(m_Layers.size())) {
        m_Layers[layerIndex].mask = mask;
    }
}

int Animator::GetLayerCount() const {
    return static_cast<int>(m_Layers.size());
}

int Animator::AddUpperBodyLayer(int animationIndex, const std::string& spineBoneName, float weight) {
    if (!m_Skeleton) {
        std::cerr << "Cannot add upper body layer: no skeleton set" << std::endl;
        return -1;
    }
    
    BoneMask upperBodyMask = BoneMask::CreateUpperBody(m_Skeleton.get(), spineBoneName);
    return AddLayer(animationIndex, upperBodyMask, weight, false);
}

int Animator::AddAdditiveLayer(int animationIndex, const BoneMask& mask, float weight) {
    return AddLayer(animationIndex, mask, weight, true);
}

int Animator::AddBlendTreeLayer1D(int treeIndex, const BoneMask& mask, float weight, bool additive) {
    if (treeIndex < 0 || treeIndex >= static_cast<int>(m_BlendTrees1D.size())) {
        std::cerr << "Invalid 1D blend tree index for layer: " << treeIndex << std::endl;
        return -1;
    }
    
    AnimationLayer layer;
    layer.type = LayerType::BlendTree1D;
    layer.blendTreeIndex = treeIndex;
    layer.mask = mask;
    layer.weight = std::max(0.0f, std::min(1.0f, weight));
    layer.isAdditive = additive;
    layer.isPlaying = true;
    layer.loop = true;
    layer.currentTime = 0.0f;
    
    m_Layers.push_back(layer);
    return static_cast<int>(m_Layers.size()) - 1;
}

int Animator::AddBlendTreeLayer2D(int treeIndex, const BoneMask& mask, float weight, bool additive) {
    if (treeIndex < 0 || treeIndex >= static_cast<int>(m_BlendTrees2D.size())) {
        std::cerr << "Invalid 2D blend tree index for layer: " << treeIndex << std::endl;
        return -1;
    }
    
    AnimationLayer layer;
    layer.type = LayerType::BlendTree2D;
    layer.blendTreeIndex = treeIndex;
    layer.mask = mask;
    layer.weight = std::max(0.0f, std::min(1.0f, weight));
    layer.isAdditive = additive;
    layer.isPlaying = true;
    layer.loop = true;
    layer.currentTime = 0.0f;
    
    m_Layers.push_back(layer);
    return static_cast<int>(m_Layers.size()) - 1;
}

int Animator::AddBlendTreeLayer3D(int treeIndex, const BoneMask& mask, float weight, bool additive) {
    if (treeIndex < 0 || treeIndex >= static_cast<int>(m_BlendTrees3D.size())) {
        std::cerr << "Invalid 3D blend tree index for layer: " << treeIndex << std::endl;
        return -1;
    }
    
    AnimationLayer layer;
    layer.type = LayerType::BlendTree3D;
    layer.blendTreeIndex = treeIndex;
    layer.mask = mask;
    layer.weight = std::max(0.0f, std::min(1.0f, weight));
    layer.isAdditive = additive;
    layer.isPlaying = true;
    layer.loop = true;
    layer.currentTime = 0.0f;
    
    m_Layers.push_back(layer);
    return static_cast<int>(m_Layers.size()) - 1;
}

void Animator::UpdateLayerTransforms(AnimationLayer& layer, float deltaTime, std::vector<Mat4>& outMatrices) {
    if (!m_Skeleton) {
        return;
    }
    
    // Handle based on layer type
    if (layer.type == LayerType::SingleAnimation) {
        if (layer.animationIndex < 0 || layer.animationIndex >= static_cast<int>(m_Animations.size())) {
            return;
        }
        
        Animation* anim = m_Animations[layer.animationIndex].get();
        int boneCount = m_Skeleton->GetBoneCount();
        
        std::vector<Mat4> localTransforms(boneCount);
        std::vector<Mat4> globalTransforms(boneCount);
        
        // Note: For SingleAnimation, time is managed by Animator::Update loop
        // so we use layer.currentTime which is already updated.
        
        // Update local transforms from animation channels
        for (int i = 0; i < boneCount; ++i) {
            const AnimationChannel* channel = anim->GetChannelForBone(i);
            
            if (channel) {
                Vec3 position;
                Quaternion rotation;
                Vec3 scale;
                channel->GetTransform(layer.currentTime, position, rotation, scale);
                
                Mat4 translation = Mat4::Translation(position);
                Mat4 rotationMat = rotation.ToMatrix();
                Mat4 scaleMat = Mat4::Scale(scale);
                
                localTransforms[i] = translation * rotationMat * scaleMat;
            } else {
                localTransforms[i] = m_Skeleton->GetBone(i).localTransform;
            }
        }
        
        // Calculate global transforms
        m_Skeleton->CalculateGlobalTransforms(localTransforms, globalTransforms);
        
        // Calculate final bone matrices
        outMatrices.resize(boneCount);
        for (int i = 0; i < boneCount; ++i) {
            outMatrices[i] = globalTransforms[i] * m_Skeleton->GetBone(i).inverseBindMatrix;
        }
    } 
    else if (layer.type == LayerType::BlendTree1D) {
        if (layer.blendTreeIndex >= 0 && layer.blendTreeIndex < static_cast<int>(m_BlendTrees1D.size())) {
            m_BlendTrees1D[layer.blendTreeIndex]->Update(deltaTime, m_Animations, m_Skeleton.get(), outMatrices);
        }
    }
    else if (layer.type == LayerType::BlendTree2D) {
        if (layer.blendTreeIndex >= 0 && layer.blendTreeIndex < static_cast<int>(m_BlendTrees2D.size())) {
            m_BlendTrees2D[layer.blendTreeIndex]->Update(deltaTime, m_Animations, m_Skeleton.get(), outMatrices);
        }
    }
    else if (layer.type == LayerType::BlendTree3D) {
        if (layer.blendTreeIndex >= 0 && layer.blendTreeIndex < static_cast<int>(m_BlendTrees3D.size())) {
            m_BlendTrees3D[layer.blendTreeIndex]->Update(deltaTime, m_Animations, m_Skeleton.get(), outMatrices);
        }
    }
}

void Animator::BlendLayerMatrices(const std::vector<Mat4>& base, const std::vector<Mat4>& layer,
                                   const BoneMask& mask, float weight, bool additive,
                                   std::vector<Mat4>& result) {
    size_t count = std::min(base.size(), layer.size());
    if (result.size() < count) {
        result.resize(count);
    }
    
    for (size_t i = 0; i < count; ++i) {
        float maskWeight = mask.GetBoneWeight(static_cast<int>(i));
        float finalWeight = maskWeight * weight;
        
        if (finalWeight <= 0.0001f) {
            // No influence from layer
            result[i] = base[i];
        } else if (additive) {
            // Additive blending: add difference from identity
            // For simplicity, we'll decompose and add the differences
            Vec3 basePos, layerPos, baseScale, layerScale;
            Quaternion baseRot, layerRot;
            
            DecomposeMatrix(base[i], basePos, baseRot, baseScale);
            DecomposeMatrix(layer[i], layerPos, layerRot, layerScale);
            
            // Add weighted differences
            Vec3 finalPos = basePos + (layerPos * finalWeight);
            Vec3 finalScale = baseScale + ((layerScale - Vec3(1,1,1)) * finalWeight);
            
            // For rotation, blend towards layer rotation
            Quaternion finalRot = Quaternion::Slerp(baseRot, layerRot, finalWeight);
            
            result[i] = ComposeMatrix(finalPos, finalRot, finalScale);
        } else {
            // Override blending: blend towards layer
            Vec3 basePos, layerPos, baseScale, layerScale;
            Quaternion baseRot, layerRot;
            
            DecomposeMatrix(base[i], basePos, baseRot, baseScale);
            DecomposeMatrix(layer[i], layerPos, layerRot, layerScale);
            
            // Interpolate components
            Vec3 blendPos;
            blendPos.x = basePos.x + finalWeight * (layerPos.x - basePos.x);
            blendPos.y = basePos.y + finalWeight * (layerPos.y - basePos.y);
            blendPos.z = basePos.z + finalWeight * (layerPos.z - basePos.z);
            
            Quaternion blendRot = Quaternion::Slerp(baseRot, layerRot, finalWeight);
            
            Vec3 blendScale;
            blendScale.x = baseScale.x + finalWeight * (layerScale.x - baseScale.x);
            blendScale.y = baseScale.y + finalWeight * (layerScale.y - baseScale.y);
            blendScale.z = baseScale.z + finalWeight * (layerScale.z - baseScale.z);
            
            result[i] = ComposeMatrix(blendPos, blendRot, blendScale);
        }
    }
}

// Layer transition methods
void Animator::TransitionLayerToAnimation(int layerIndex, int animationIndex, float blendTime, BlendCurve curve) {
    if (layerIndex < 0 || layerIndex >= static_cast<int>(m_Layers.size())) {
        std::cerr << "Invalid layer index for transition: " << layerIndex << std::endl;
        return;
    }
    
    if (animationIndex < 0 || animationIndex >= static_cast<int>(m_Animations.size())) {
        std::cerr << "Invalid animation index for layer transition: " << animationIndex << std::endl;
        return;
    }
    
    AnimationLayer& layer = m_Layers[layerIndex];
    
    // If already transitioning to this animation, ignore
    if (layer.isBlending && layer.toAnimationIndex == animationIndex) {
        return;
    }
    
    // Use defaults if not specified
    if (blendTime < 0.0f) {
        blendTime = m_DefaultBlendTime;
    }
    if (curve == BlendCurve::Linear && m_DefaultBlendCurve != BlendCurve::Linear) {
        curve = m_DefaultBlendCurve;
    }
    
    // If layer not playing, just set the animation directly
    if (!layer.isPlaying || layer.animationIndex < 0) {
        layer.animationIndex = animationIndex;
        layer.currentTime = 0.0f;
        layer.isPlaying = true;
        return;
    }
    
    // Setup layer blend state
    layer.isBlending = true;
    layer.fromType = layer.type;
    layer.fromAnimationIndex = layer.animationIndex;
    layer.fromBlendTreeIndex = layer.blendTreeIndex;
    
    layer.toType = LayerType::SingleAnimation;
    layer.toAnimationIndex = animationIndex;
    layer.toBlendTreeIndex = -1;
    
    layer.blendTime = blendTime;
    layer.currentBlendTime = 0.0f;
    layer.blendCurve = curve;
    
    // Cache current bone matrices as 'from' state
    if (m_Skeleton) {
        int boneCount = m_Skeleton->GetBoneCount();
        layer.fromBoneMatrices.resize(boneCount);
        
        // Calculate current layer matrices
        std::vector<Mat4> currentMatrices;
        UpdateLayerTransforms(layer, 0.0f, currentMatrices);
        layer.fromBoneMatrices = currentMatrices;
    }
    
    std::cout << "Layer " << layerIndex << ": Starting blend from animation " 
              << layer.fromAnimationIndex << " to " << layer.toAnimationIndex 
              << " over " << blendTime << "s" << std::endl;
}

void Animator::TransitionLayerToBlendTree(int layerIndex, int treeIndex, int type, float blendTime, BlendCurve curve) {
    if (layerIndex < 0 || layerIndex >= static_cast<int>(m_Layers.size())) {
        std::cerr << "Invalid layer index for transition: " << layerIndex << std::endl;
        return;
    }
    
    LayerType targetType = static_cast<LayerType>(type);
    bool validTree = false;
    if (targetType == LayerType::BlendTree1D) validTree = (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees1D.size()));
    else if (targetType == LayerType::BlendTree2D) validTree = (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees2D.size()));
    else if (targetType == LayerType::BlendTree3D) validTree = (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees3D.size()));
    
    if (!validTree) {
        std::cerr << "Invalid blend tree index for layer transition: " << treeIndex << std::endl;
        return;
    }
    
    AnimationLayer& layer = m_Layers[layerIndex];
    
    // If already transitioning to this tree, ignore
    if (layer.isBlending && layer.toType == targetType && layer.toBlendTreeIndex == treeIndex) {
        return;
    }
    
    // Use defaults if not specified
    if (blendTime < 0.0f) {
        blendTime = m_DefaultBlendTime;
    }
    
     // If layer not playing, just set directly
    if (!layer.isPlaying) {
        layer.type = targetType;
        layer.blendTreeIndex = treeIndex;
        layer.currentTime = 0.0f;
        layer.isPlaying = true;
        return;
    }
    
    // Setup layer blend state
    layer.isBlending = true;
    layer.fromType = layer.type;
    layer.fromAnimationIndex = layer.animationIndex;
    layer.fromBlendTreeIndex = layer.blendTreeIndex;
    
    layer.toType = targetType;
    layer.toAnimationIndex = -1;
    layer.toBlendTreeIndex = treeIndex;
    
    layer.blendTime = blendTime;
    layer.currentBlendTime = 0.0f;
    layer.blendCurve = curve;
    
    // Cache current bone matrices as 'from' state
    if (m_Skeleton) {
        int boneCount = m_Skeleton->GetBoneCount();
        layer.fromBoneMatrices.resize(boneCount);
        
        // Calculate current layer matrices
        std::vector<Mat4> currentMatrices;
        UpdateLayerTransforms(layer, 0.0f, currentMatrices);
        layer.fromBoneMatrices = currentMatrices;
    }
}

bool Animator::IsLayerBlending(int layerIndex) const {
    if (layerIndex >= 0 && layerIndex < static_cast<int>(m_Layers.size())) {
        return m_Layers[layerIndex].isBlending;
    }
    return false;
}

float Animator::GetLayerBlendProgress(int layerIndex) const {
    if (layerIndex >= 0 && layerIndex < static_cast<int>(m_Layers.size())) {
        const AnimationLayer& layer = m_Layers[layerIndex];
        if (!layer.isBlending || layer.blendTime <= 0.0f) {
            return 1.0f;
        }
        return std::min(1.0f, layer.currentBlendTime / layer.blendTime);
    }
    return 1.0f;
}

// Blend Tree Management
int Animator::CreateBlendTree1D() {
    auto tree = std::make_unique<BlendTree1D>();
    m_BlendTrees1D.push_back(std::move(tree));
    return static_cast<int>(m_BlendTrees1D.size() - 1);
}

int Animator::CreateBlendTree2D() {
    auto tree = std::make_unique<BlendTree2D>();
    m_BlendTrees2D.push_back(std::move(tree));
    return static_cast<int>(m_BlendTrees2D.size() - 1);
}

int Animator::CreateBlendTree3D() {
    auto tree = std::make_unique<BlendTree3D>();
    m_BlendTrees3D.push_back(std::move(tree));
    return static_cast<int>(m_BlendTrees3D.size() - 1);
}

void Animator::AddBlendTreeNode1D(int treeIndex, int animationIndex, float parameter) {
    if (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees1D.size())) {
        m_BlendTrees1D[treeIndex]->AddNode(animationIndex, parameter);
    }
}

void Animator::AddBlendTreeNode2D(int treeIndex, int animationIndex, Vec2 parameter) {
    if (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees2D.size())) {
        m_BlendTrees2D[treeIndex]->AddNode(animationIndex, parameter);
    }
}

void Animator::AddBlendTreeNode3D(int treeIndex, int animationIndex, Vec3 parameter) {
    if (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees3D.size())) {
        m_BlendTrees3D[treeIndex]->AddNode(animationIndex, parameter);
    }
}

void Animator::SetBlendTreeParameter1D(int treeIndex, float value) {
    if (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees1D.size())) {
        m_BlendTrees1D[treeIndex]->SetParameter(value);
    }
}

void Animator::SetBlendTreeParameterSmooth1D(int treeIndex, float value, float smoothTime) {
    if (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees1D.size())) {
        m_BlendTrees1D[treeIndex]->SetParameterSmooth(value, smoothTime);
    }
}

void Animator::SetBlendTreeParameter2D(int treeIndex, Vec2 value) {
    if (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees2D.size())) {
        m_BlendTrees2D[treeIndex]->SetParameter(value);
    }
}

void Animator::SetBlendTreeParameterSmooth2D(int treeIndex, Vec2 value, float smoothTime) {
    if (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees2D.size())) {
        m_BlendTrees2D[treeIndex]->SetParameterSmooth(value, smoothTime);
    }
}

void Animator::SetBlendTreeParameter3D(int treeIndex, Vec3 value) {
    if (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees3D.size())) {
        m_BlendTrees3D[treeIndex]->SetParameter(value);
    }
}

void Animator::SetBlendTreeParameterSmooth3D(int treeIndex, Vec3 value, float smoothTime) {
    if (treeIndex >= 0 && treeIndex < static_cast<int>(m_BlendTrees3D.size())) {
        m_BlendTrees3D[treeIndex]->SetParameterSmooth(value, smoothTime);
    }
}

void Animator::SetLayerBlendTree1D(int layerIndex, int treeIndex) {
    if (layerIndex >= 0 && layerIndex < static_cast<int>(m_Layers.size())) {
        m_Layers[layerIndex].type = LayerType::BlendTree1D;
        m_Layers[layerIndex].blendTreeIndex = treeIndex;
        m_Layers[layerIndex].isPlaying = true;
    }
}

void Animator::SetLayerBlendTree2D(int layerIndex, int treeIndex) {
    if (layerIndex >= 0 && layerIndex < static_cast<int>(m_Layers.size())) {
        m_Layers[layerIndex].type = LayerType::BlendTree2D;
        m_Layers[layerIndex].blendTreeIndex = treeIndex;
        m_Layers[layerIndex].isPlaying = true;
    }
}

void Animator::SetLayerBlendTree3D(int layerIndex, int treeIndex) {
    if (layerIndex >= 0 && layerIndex < static_cast<int>(m_Layers.size())) {
        m_Layers[layerIndex].type = LayerType::BlendTree3D;
        m_Layers[layerIndex].blendTreeIndex = treeIndex;
        m_Layers[layerIndex].isPlaying = true;
    }
}

// Inverse Kinematics
int Animator::AddIKChain(const std::string& rootBone, const std::string& effectorBone) {
    if (!m_Skeleton) return -1;
    
    IKChain chain;
    chain.rootBoneName = rootBone;
    chain.effectorBoneName = effectorBone;
    chain.rootIndex = m_Skeleton->FindBoneIndex(rootBone);
    chain.effectorIndex = m_Skeleton->FindBoneIndex(effectorBone);
    chain.weight = 0.0f; // Disabled by default
    chain.targetPosition = Vec3(0, 0, 0);
    
    if (chain.rootIndex == -1 || chain.effectorIndex == -1) {
        std::cerr << "Invalid bones for IK chain: " << rootBone << " -> " << effectorBone << std::endl;
        return -1;
    }
    
    // Build chain indices (from Effector up to Root)
    int current = chain.effectorIndex;
    while (current != -1 && current != chain.rootIndex) {
        chain.chainIndices.push_back(current);
        const Bone& bone = m_Skeleton->GetBone(current);
        current = bone.parentIndex;
    }
    
    if (current == chain.rootIndex) {
        chain.chainIndices.push_back(current);
    } else {
        std::cerr << "Root bone is not an ancestor of effector bone for IK chain" << std::endl;
        return -1;
    }
    
    m_IKChains.push_back(chain);
    return static_cast<int>(m_IKChains.size()) - 1;
}

void Animator::SetIKTarget(int chainIndex, Vec3 targetPos) {
    if (chainIndex >= 0 && chainIndex < static_cast<int>(m_IKChains.size())) {
        m_IKChains[chainIndex].targetPosition = targetPos;
    }
}

void Animator::SetIKWeight(int chainIndex, float weight) {
    if (chainIndex >= 0 && chainIndex < static_cast<int>(m_IKChains.size())) {
        m_IKChains[chainIndex].weight = std::max(0.0f, std::min(1.0f, weight));
    }
}

int Animator::CreateLocomotionBlendTree1D(int idleAnim, int walkAnim, int runAnim) {
    int tree = CreateBlendTree1D();
    AddBlendTreeNode1D(tree, idleAnim, 0.0f);
    AddBlendTreeNode1D(tree, walkAnim, 0.5f);
    AddBlendTreeNode1D(tree, runAnim, 1.0f);
    return tree;
}

int Animator::CreateDirectionalBlendTree2D(const std::vector<int>& animIndices, const std::vector<Vec2>& positions) {
    if (animIndices.size() != positions.size()) {
        std::cerr << "Mismatch in animation indices and positions for 2D blend tree" << std::endl;
        return -1;
    }
    
    int tree = CreateBlendTree2D();
    for (size_t i = 0; i < animIndices.size(); ++i) {
        AddBlendTreeNode2D(tree, animIndices[i], positions[i]);
    }
    return tree;
}
