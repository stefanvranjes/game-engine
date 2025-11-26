#pragma once

struct Vec2 {
    float x, y;

    Vec2() : x(0), y(0) {}
    Vec2(float _x, float _y) : x(_x), y(_y) {}

    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }

    Vec2 operator-(const Vec2& other) const {
        return Vec2(x - other.x, y - other.y);
    }

    Vec2 operator*(float scalar) const {
        return Vec2(x * scalar, y * scalar);
    }

    float Dot(const Vec2& other) const {
        return x * other.x + y * other.y;
    }

    float Length() const {
        return std::sqrt(x * x + y * y);
    }

    Vec2 Normalized() const {
        float len = Length();
        if (len > 0) {
            return Vec2(x / len, y / len);
        }
        return Vec2(0, 0);
    }
};
