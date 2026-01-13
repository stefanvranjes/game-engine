#include "Math/Quat.h"
#include <cmath>

Quat Quat::FromAxisAngle(const Vec3& axis, float angle) {
    float halfAngle = angle * 0.5f;
    float s = std::sin(halfAngle);
    return Quat(axis.x * s, axis.y * s, axis.z * s, std::cos(halfAngle));
}

Quat Quat::FromEuler(float pitch, float yaw, float roll) {
    float cy = std::cos(yaw * 0.5f);
    float sy = std::sin(yaw * 0.5f);
    float cp = std::cos(pitch * 0.5f);
    float sp = std::sin(pitch * 0.5f);
    float cr = std::cos(roll * 0.5f);
    float sr = std::sin(roll * 0.5f);

    Quat q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
    return q;
}

Quat Quat::operator*(const Quat& other) const {
    return Quat(
        w * other.x + x * other.w + y * other.z - z * other.y,
        w * other.y - x * other.z + y * other.w + z * other.x,
        w * other.z + x * other.y - y * other.x + z * other.w,
        w * other.w - x * other.x - y * other.y - z * other.z
    );
}

Vec3 Quat::operator*(const Vec3& v) const {
    Vec3 qv(x, y, z);
    Vec3 uv = qv.Cross(v);
    Vec3 uuv = qv.Cross(uv);
    return v + (uv * w + uuv) * 2.0f;
}

Quat Quat::Conjugate() const {
    return Quat(-x, -y, -z, w);
}

Quat Quat::Inverse() const {
    float lenSq = x*x + y*y + z*z + w*w;
    if (lenSq > 0.0f) {
        float invLen = 1.0f / lenSq;
        return Quat(-x * invLen, -y * invLen, -z * invLen, w * invLen);
    }
    return Quat::Identity();
}

float Quat::Length() const {
    return std::sqrt(x*x + y*y + z*z + w*w);
}

Quat Quat::Normalized() const {
    float len = Length();
    if (len > 0.0f) {
        float invLen = 1.0f / len;
        return Quat(x * invLen, y * invLen, z * invLen, w * invLen);
    }
    return Quat::Identity();
}

Vec3 Quat::ToEuler() const {
    Vec3 euler;
    
    // Roll (x-axis rotation)
    float sinr_cosp = 2.0f * (w * x + y * z);
    float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
    euler.x = std::atan2(sinr_cosp, cosr_cosp);
    
    // Pitch (y-axis rotation)
    float sinp = 2.0f * (w * y - z * x);
    if (std::abs(sinp) >= 1.0f)
        euler.y = std::copysign(3.14159265f / 2.0f, sinp);
    else
        euler.y = std::asin(sinp);
    
    // Yaw (z-axis rotation)
    float siny_cosp = 2.0f * (w * z + x * y);
    float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    euler.z = std::atan2(siny_cosp, cosy_cosp);
    
    return euler;
}

Quat Quat::Slerp(const Quat& a, const Quat& b, float t) {
    float cosHalfTheta = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;

    Quat end = b;
    if (cosHalfTheta < 0.0f) {
        end = Quat(-b.x, -b.y, -b.z, -b.w);
        cosHalfTheta = -cosHalfTheta;
    }

    if (std::abs(cosHalfTheta) >= 1.0f) {
        return a;
    }

    float halfTheta = std::acos(cosHalfTheta);
    float sinHalfTheta = std::sqrt(1.0f - cosHalfTheta * cosHalfTheta);

    if (std::abs(sinHalfTheta) < 0.001f) {
        return Quat(
            a.x * 0.5f + end.x * 0.5f,
            a.y * 0.5f + end.y * 0.5f,
            a.z * 0.5f + end.z * 0.5f,
            a.w * 0.5f + end.w * 0.5f
        );
    }

    float ratioA = std::sin((1.0f - t) * halfTheta) / sinHalfTheta;
    float ratioB = std::sin(t * halfTheta) / sinHalfTheta;

    return Quat(
        a.x * ratioA + end.x * ratioB,
        a.y * ratioA + end.y * ratioB,
        a.z * ratioA + end.z * ratioB,
        a.w * ratioA + end.w * ratioB
    );
}
