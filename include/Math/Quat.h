#pragma once

#include "Vec3.h"

struct Quat {
    float x, y, z, w;
    
    Quat() : x(0), y(0), z(0), w(1) {}
    Quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    
    // Identity quaternion
    static Quat Identity() { return Quat(0, 0, 0, 1); }
    
    // Create from axis-angle
    static Quat FromAxisAngle(const Vec3& axis, float angle);
    
    // Create from Euler angles (in radians)
    static Quat FromEuler(float pitch, float yaw, float roll);
    
    // Quaternion operations
    Quat operator*(const Quat& other) const;
    Vec3 operator*(const Vec3& v) const;
    
    Quat Conjugate() const;
    Quat Inverse() const;
    float Length() const;
    Quat Normalized() const;
    
    // Convert to Euler angles
    Vec3 ToEuler() const;
};
