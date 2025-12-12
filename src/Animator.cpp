#include "Animator.h"
#include "BlendCurve.h"
#include "BoneMask.h"
#include <iostream>

Animator::Animator() 
    : m_CurrentAnimationIndex(-1)
    , m_CurrentTime(0.0f)
    , m_IsPlaying(false)
    , m_IsPaused(false)
    , m_Loop(true)
    , m_PlaybackSpeed(1.0f)
    , m_DefaultBlendTime(0.3f)
    , m_DefaultBlendCurve(BlendCurve::SmoothStep) {
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
        
        // Calculate bone matrices for both animations
        int boneCount = m_Skeleton->GetBoneCount();
        std::vector<Mat4> toBoneMatrices(boneCount);
        
        // Calculate matrices for 'from' animation (keep using cached time)
        for (int i = 0; i < boneCount; ++i) {
            const AnimationChannel* channel = fromAnim->GetChannelForBone(i);
            if (channel) {
                Vec3 position;
                Quaternion rotation;
                Vec3 scale;
                channel->GetTransform(m_CurrentTime, position, rotation, scale);
                
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
                channel->GetTransform(m_CurrentTime, position, rotation, scale);
                
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
                        layer.animationIndex = layer.toAnimationIndex;
                        t = 1.0f;
                    }
                    
                    // Apply easing curve
                    float easedT = EasingFunctions::Apply(t, layer.blendCurve);
                    
                    // Update time for target animation
                    Animation* toAnim = m_Animations[layer.toAnimationIndex].get();
                    layer.currentTime += deltaTime * m_PlaybackSpeed;
                    
                    // Handle looping for target
                    float toDuration = toAnim->GetDuration();
                    if (layer.currentTime >= toDuration) {
                        if (layer.loop) {
                            layer.currentTime = fmod(layer.currentTime, toDuration);
                        } else {
                            layer.currentTime = toDuration;
                            if (!layer.isBlending) {
                                layer.isPlaying = false;
                            }
                        }
                    }
                    
                    // Calculate matrices for target animation
                    AnimationLayer tempLayer = layer;
                    tempLayer.animationIndex = layer.toAnimationIndex;
                    UpdateLayerTransforms(tempLayer, m_LayerBoneMatrices);
                    
                    // Blend from cached 'from' matrices to 'to' matrices
                    BlendBoneMatrices(layer.fromBoneMatrices, m_LayerBoneMatrices, 
                                     easedT, layerBlendedMatrices);
                    
                } else {
                    // Normal layer update (no blending)
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
                    UpdateLayerTransforms(layer, layerBlendedMatrices);
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

void Animator::UpdateLayerTransforms(AnimationLayer& layer, std::vector<Mat4>& outMatrices) {
    if (!m_Skeleton || layer.animationIndex < 0) {
        return;
    }
    
    Animation* anim = m_Animations[layer.animationIndex].get();
    int boneCount = m_Skeleton->GetBoneCount();
    
    std::vector<Mat4> localTransforms(boneCount);
    std::vector<Mat4> globalTransforms(boneCount);
    
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
    layer.fromAnimationIndex = layer.animationIndex;
    layer.toAnimationIndex = animationIndex;
    layer.blendTime = blendTime;
    layer.currentBlendTime = 0.0f;
    layer.blendCurve = curve;
    
    // Cache current bone matrices as 'from' state
    if (m_Skeleton) {
        int boneCount = m_Skeleton->GetBoneCount();
        layer.fromBoneMatrices.resize(boneCount);
        
        // Calculate current layer matrices
        std::vector<Mat4> currentMatrices;
        UpdateLayerTransforms(layer, currentMatrices);
        layer.fromBoneMatrices = currentMatrices;
    }
    
    std::cout << "Layer " << layerIndex << ": Starting blend from animation " 
              << layer.fromAnimationIndex << " to " << layer.toAnimationIndex 
              << " over " << blendTime << "s" << std::endl;
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
