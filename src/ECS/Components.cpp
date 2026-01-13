#include "ECS/Components.h"
#include "Math/Mat4.h"
#include <cmath>

Mat4 TransformComponent::GetLocalMatrix() const {
    Mat4 translation = Mat4::Translate(m_Position);
    Mat4 rotation = Mat4::FromQuaternion(m_Rotation);
    Mat4 scale = Mat4::Scale(m_Scale);
    return translation * rotation * scale;
}

void TransformComponent::SetEulerAngles(const Vec3& angles) {
    // Convert euler angles (in degrees) to quaternion
    Vec3 rad = angles * 3.14159265f / 180.0f;
    
    float cy = std::cos(rad.z * 0.5f);
    float sy = std::sin(rad.z * 0.5f);
    float cp = std::cos(rad.y * 0.5f);
    float sp = std::sin(rad.y * 0.5f);
    float cr = std::cos(rad.x * 0.5f);
    float sr = std::sin(rad.x * 0.5f);

    m_Rotation.w = cy * cp * cr + sy * sp * sr;
    m_Rotation.x = cy * cp * sr - sy * sp * cr;
    m_Rotation.y = sy * cp * sr + cy * sp * cr;
    m_Rotation.z = sy * cp * cr - cy * sp * sr;
    
    m_IsDirty = true;
}

Vec3 TransformComponent::GetEulerAngles() const {
    // Convert quaternion to euler angles (in degrees)
    float sqx = m_Rotation.x * m_Rotation.x;
    float sqy = m_Rotation.y * m_Rotation.y;
    float sqz = m_Rotation.z * m_Rotation.z;
    float sqw = m_Rotation.w * m_Rotation.w;

    float pitch = std::asin(2.0f * (m_Rotation.w * m_Rotation.x - m_Rotation.y * m_Rotation.z));
    float yaw = std::atan2(2.0f * (m_Rotation.w * m_Rotation.y + m_Rotation.x * m_Rotation.z), 
                           1.0f - 2.0f * (sqy + sqz));
    float roll = std::atan2(2.0f * (m_Rotation.w * m_Rotation.z + m_Rotation.x * m_Rotation.y),
                            1.0f - 2.0f * (sqx + sqz));

    return Vec3(pitch, yaw, roll) * 180.0f / 3.14159265f;
}
