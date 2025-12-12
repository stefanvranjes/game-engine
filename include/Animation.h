#pragma once

#include "Math/Vec3.h"
#include "Math/Mat4.h"
#include <vector>
#include <string>

// Quaternion structure for rotations
struct Quaternion {
    float x, y, z, w;
    
    Quaternion() : x(0), y(0), z(0), w(1) {}
    Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    
    // Convert to rotation matrix
    Mat4 ToMatrix() const;
    
    // Spherical linear interpolation
    static Quaternion Slerp(const Quaternion& a, const Quaternion& b, float t);
    
    // Normalize quaternion
    void Normalize();
};

// Single keyframe for animation
struct Keyframe {
    float time;
    Vec3 position;
    Quaternion rotation;
    Vec3 scale;
    
    Keyframe() : time(0.0f), position(0, 0, 0), scale(1, 1, 1) {}
};

// Animation channel for a single bone
struct AnimationChannel {
    int boneIndex;
    std::vector<Keyframe> keyframes;
    
    AnimationChannel() : boneIndex(-1) {}
    
    // Get interpolated transform at given time
    void GetTransform(float time, Vec3& outPosition, Quaternion& outRotation, Vec3& outScale) const;
};

// Complete animation clip
class Animation {
public:
    Animation(const std::string& name = "Animation");
    ~Animation();
    
    void SetName(const std::string& name) { m_Name = name; }
    const std::string& GetName() const { return m_Name; }
    
    void SetDuration(float duration) { m_Duration = duration; }
    float GetDuration() const { return m_Duration; }
    
    void AddChannel(const AnimationChannel& channel);
    const std::vector<AnimationChannel>& GetChannels() const { return m_Channels; }
    
    // Get channel for specific bone
    const AnimationChannel* GetChannelForBone(int boneIndex) const;
    
private:
    std::string m_Name;
    float m_Duration;
    std::vector<AnimationChannel> m_Channels;
};

// Utility functions for matrix decomposition/composition (for blending)
void DecomposeMatrix(const Mat4& matrix, Vec3& translation, Quaternion& rotation, Vec3& scale);
Mat4 ComposeMatrix(const Vec3& translation, const Quaternion& rotation, const Vec3& scale);
