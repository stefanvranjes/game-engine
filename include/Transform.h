#pragma once

#include "Math/Vec3.h"
#include "Math/Mat4.h"

class Transform {
public:
    Vec3 position;
    Vec3 rotation; // Euler angles in degrees
    Vec3 scale;

    Transform()
        : position(0, 0, 0)
        , rotation(0, 0, 0)
        , scale(1, 1, 1)
    {}

    Transform(const Vec3& pos, const Vec3& rot = Vec3(0, 0, 0), const Vec3& scl = Vec3(1, 1, 1))
        : position(pos)
        , rotation(rot)
        , scale(scl)
    {}

    Mat4 GetModelMatrix() const {
        Mat4 translation = Mat4::Translate(position);
        Mat4 rotationX = Mat4::RotateX(rotation.x * 3.14159f / 180.0f);
        Mat4 rotationY = Mat4::RotateY(rotation.y * 3.14159f / 180.0f);
        Mat4 rotationZ = Mat4::RotateZ(rotation.z * 3.14159f / 180.0f);
        Mat4 scaling = Mat4::Scale(scale);

        // Combine: Translation * Rotation * Scale
        return translation * rotationY * rotationX * rotationZ * scaling;
    }

    Mat4 GetMatrix() const {
        return GetModelMatrix();
    }
};
