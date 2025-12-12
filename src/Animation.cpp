#include "Animation.h"
#include <cmath>
#include <algorithm>

// Quaternion implementation
Mat4 Quaternion::ToMatrix() const {
    Mat4 result = Mat4::Identity();
    
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;
    
    result.m[0][0] = 1.0f - 2.0f * (yy + zz);
    result.m[0][1] = 2.0f * (xy + wz);
    result.m[0][2] = 2.0f * (xz - wy);
    
    result.m[1][0] = 2.0f * (xy - wz);
    result.m[1][1] = 1.0f - 2.0f * (xx + zz);
    result.m[1][2] = 2.0f * (yz + wx);
    
    result.m[2][0] = 2.0f * (xz + wy);
    result.m[2][1] = 2.0f * (yz - wx);
    result.m[2][2] = 1.0f - 2.0f * (xx + yy);
    
    return result;
}

Quaternion Quaternion::Slerp(const Quaternion& a, const Quaternion& b, float t) {
    Quaternion result;
    
    // Compute dot product
    float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    
    // If negative dot, negate one quaternion to take shorter path
    Quaternion b_copy = b;
    if (dot < 0.0f) {
        b_copy.x = -b.x;
        b_copy.y = -b.y;
        b_copy.z = -b.z;
        b_copy.w = -b.w;
        dot = -dot;
    }
    
    // If quaternions are very close, use linear interpolation
    if (dot > 0.9995f) {
        result.x = a.x + t * (b_copy.x - a.x);
        result.y = a.y + t * (b_copy.y - a.y);
        result.z = a.z + t * (b_copy.z - a.z);
        result.w = a.w + t * (b_copy.w - a.w);
        result.Normalize();
        return result;
    }
    
    // Clamp dot to avoid acos domain errors
    dot = std::max(-1.0f, std::min(1.0f, dot));
    
    float theta_0 = std::acos(dot);
    float theta = theta_0 * t;
    float sin_theta = std::sin(theta);
    float sin_theta_0 = std::sin(theta_0);
    
    float s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
    float s1 = sin_theta / sin_theta_0;
    
    result.x = s0 * a.x + s1 * b_copy.x;
    result.y = s0 * a.y + s1 * b_copy.y;
    result.z = s0 * a.z + s1 * b_copy.z;
    result.w = s0 * a.w + s1 * b_copy.w;
    
    return result;
}

void Quaternion::Normalize() {
    float length = std::sqrt(x * x + y * y + z * z + w * w);
    if (length > 0.0f) {
        x /= length;
        y /= length;
        z /= length;
        w /= length;
    }
}

// AnimationChannel implementation
void AnimationChannel::GetTransform(float time, Vec3& outPosition, Quaternion& outRotation, Vec3& outScale) const {
    if (keyframes.empty()) {
        outPosition = Vec3(0, 0, 0);
        outRotation = Quaternion();
        outScale = Vec3(1, 1, 1);
        return;
    }
    
    // Find keyframes to interpolate between
    if (time <= keyframes[0].time) {
        outPosition = keyframes[0].position;
        outRotation = keyframes[0].rotation;
        outScale = keyframes[0].scale;
        return;
    }
    
    if (time >= keyframes.back().time) {
        outPosition = keyframes.back().position;
        outRotation = keyframes.back().rotation;
        outScale = keyframes.back().scale;
        return;
    }
    
    // Binary search for keyframe pair
    size_t nextFrame = 0;
    for (size_t i = 0; i < keyframes.size() - 1; ++i) {
        if (time >= keyframes[i].time && time < keyframes[i + 1].time) {
            nextFrame = i + 1;
            break;
        }
    }
    
    const Keyframe& k0 = keyframes[nextFrame - 1];
    const Keyframe& k1 = keyframes[nextFrame];
    
    float duration = k1.time - k0.time;
    float t = (duration > 0.0f) ? ((time - k0.time) / duration) : 0.0f;
    
    // Linear interpolation for position and scale
    outPosition.x = k0.position.x + t * (k1.position.x - k0.position.x);
    outPosition.y = k0.position.y + t * (k1.position.y - k0.position.y);
    outPosition.z = k0.position.z + t * (k1.position.z - k0.position.z);
    
    outScale.x = k0.scale.x + t * (k1.scale.x - k0.scale.x);
    outScale.y = k0.scale.y + t * (k1.scale.y - k0.scale.y);
    outScale.z = k0.scale.z + t * (k1.scale.z - k0.scale.z);
    
    // Spherical interpolation for rotation
    outRotation = Quaternion::Slerp(k0.rotation, k1.rotation, t);
}

// Animation implementation
Animation::Animation(const std::string& name) 
    : m_Name(name), m_SyncGroup(""), m_Duration(0.0f) {
}

Animation::~Animation() {
}

void Animation::AddChannel(const AnimationChannel& channel) {
    m_Channels.push_back(channel);
}

const AnimationChannel* Animation::GetChannelForBone(int boneIndex) const {
    for (const auto& channel : m_Channels) {
        if (channel.boneIndex == boneIndex) {
            return &channel;
        }
    }
    return nullptr;
}

// Matrix decomposition for blending
void DecomposeMatrix(const Mat4& matrix, Vec3& translation, Quaternion& rotation, Vec3& scale) {
    // Extract translation (last column)
    translation.x = matrix.m[3][0];
    translation.y = matrix.m[3][1];
    translation.z = matrix.m[3][2];
    
    // Extract scale (length of each column vector)
    Vec3 col0(matrix.m[0][0], matrix.m[0][1], matrix.m[0][2]);
    Vec3 col1(matrix.m[1][0], matrix.m[1][1], matrix.m[1][2]);
    Vec3 col2(matrix.m[2][0], matrix.m[2][1], matrix.m[2][2]);
    
    scale.x = std::sqrt(col0.x * col0.x + col0.y * col0.y + col0.z * col0.z);
    scale.y = std::sqrt(col1.x * col1.x + col1.y * col1.y + col1.z * col1.z);
    scale.z = std::sqrt(col2.x * col2.x + col2.y * col2.y + col2.z * col2.z);
    
    // Remove scale from rotation matrix
    Mat4 rotMatrix = matrix;
    if (scale.x > 0.0001f) {
        rotMatrix.m[0][0] /= scale.x;
        rotMatrix.m[0][1] /= scale.x;
        rotMatrix.m[0][2] /= scale.x;
    }
    if (scale.y > 0.0001f) {
        rotMatrix.m[1][0] /= scale.y;
        rotMatrix.m[1][1] /= scale.y;
        rotMatrix.m[1][2] /= scale.y;
    }
    if (scale.z > 0.0001f) {
        rotMatrix.m[2][0] /= scale.z;
        rotMatrix.m[2][1] /= scale.z;
        rotMatrix.m[2][2] /= scale.z;
    }
    
    // Convert rotation matrix to quaternion
    float trace = rotMatrix.m[0][0] + rotMatrix.m[1][1] + rotMatrix.m[2][2];
    
    if (trace > 0.0f) {
        float s = std::sqrt(trace + 1.0f) * 2.0f;
        rotation.w = 0.25f * s;
        rotation.x = (rotMatrix.m[2][1] - rotMatrix.m[1][2]) / s;
        rotation.y = (rotMatrix.m[0][2] - rotMatrix.m[2][0]) / s;
        rotation.z = (rotMatrix.m[1][0] - rotMatrix.m[0][1]) / s;
    } else if (rotMatrix.m[0][0] > rotMatrix.m[1][1] && rotMatrix.m[0][0] > rotMatrix.m[2][2]) {
        float s = std::sqrt(1.0f + rotMatrix.m[0][0] - rotMatrix.m[1][1] - rotMatrix.m[2][2]) * 2.0f;
        rotation.w = (rotMatrix.m[2][1] - rotMatrix.m[1][2]) / s;
        rotation.x = 0.25f * s;
        rotation.y = (rotMatrix.m[0][1] + rotMatrix.m[1][0]) / s;
        rotation.z = (rotMatrix.m[0][2] + rotMatrix.m[2][0]) / s;
    } else if (rotMatrix.m[1][1] > rotMatrix.m[2][2]) {
        float s = std::sqrt(1.0f + rotMatrix.m[1][1] - rotMatrix.m[0][0] - rotMatrix.m[2][2]) * 2.0f;
        rotation.w = (rotMatrix.m[0][2] - rotMatrix.m[2][0]) / s;
        rotation.x = (rotMatrix.m[0][1] + rotMatrix.m[1][0]) / s;
        rotation.y = 0.25f * s;
        rotation.z = (rotMatrix.m[1][2] + rotMatrix.m[2][1]) / s;
    } else {
        float s = std::sqrt(1.0f + rotMatrix.m[2][2] - rotMatrix.m[0][0] - rotMatrix.m[1][1]) * 2.0f;
        rotation.w = (rotMatrix.m[1][0] - rotMatrix.m[0][1]) / s;
        rotation.x = (rotMatrix.m[0][2] + rotMatrix.m[2][0]) / s;
        rotation.y = (rotMatrix.m[1][2] + rotMatrix.m[2][1]) / s;
        rotation.z = 0.25f * s;
    }
    
    rotation.Normalize();
}

// Matrix composition for blending
Mat4 ComposeMatrix(const Vec3& translation, const Quaternion& rotation, const Vec3& scale) {
    // Start with rotation matrix
    Mat4 result = rotation.ToMatrix();
    
    // Apply scale
    result.m[0][0] *= scale.x;
    result.m[0][1] *= scale.x;
    result.m[0][2] *= scale.x;
    
    result.m[1][0] *= scale.y;
    result.m[1][1] *= scale.y;
    result.m[1][2] *= scale.y;
    
    result.m[2][0] *= scale.z;
    result.m[2][1] *= scale.z;
    result.m[2][2] *= scale.z;
    
    // Apply translation
    result.m[3][0] = translation.x;
    result.m[3][1] = translation.y;
    result.m[3][2] = translation.z;
    
    return result;
}
