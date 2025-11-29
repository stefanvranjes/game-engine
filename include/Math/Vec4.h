#pragma once

#include <cmath>

class Vec4 {
public:
    float x, y, z, w;

    Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    // Vector addition
    Vec4 operator+(const Vec4& other) const {
        return Vec4(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    // Vector subtraction
    Vec4 operator-(const Vec4& other) const {
        return Vec4(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    // Scalar multiplication
    Vec4 operator*(float scalar) const {
        return Vec4(x * scalar, y * scalar, z * scalar, w * scalar);
    }

    // Scalar division
    Vec4 operator/(float scalar) const {
        return Vec4(x / scalar, y / scalar, z / scalar, w / scalar);
    }

    // Dot product
    float Dot(const Vec4& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }

    // Length
    float Length() const {
        return std::sqrt(x * x + y * y + z * z + w * w);
    }

    // Normalize
    Vec4 Normalized() const {
        float len = Length();
        if (len > 0.0f) {
            return *this / len;
        }
        return Vec4(0, 0, 0, 0);
    }
};
