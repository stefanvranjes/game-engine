#pragma once

#include "Math/Vec3.h"
#include "Math/Mat4.h"

struct GLFWwindow;

class Camera {
public:
    Camera(const Vec3& position = Vec3(0, 0, 3), float fov = 45.0f, float aspect = 16.0f / 9.0f);
    
    Mat4 GetViewMatrix() const;
    Mat4 GetProjectionMatrix() const;
    
    void ProcessInput(GLFWwindow* window, float deltaTime);
    void SetAspectRatio(float aspect);
    
    Vec3 GetPosition() const { return m_Position; }
    void SetPosition(const Vec3& position) { m_Position = position; }
    
private:
    Vec3 m_Position;
    Vec3 m_Front;
    Vec3 m_Up;
    Vec3 m_Right;
    Vec3 m_WorldUp;
    
    float m_Yaw;
    float m_Pitch;
    
    float m_FOV;
    float m_AspectRatio;
    float m_NearPlane;
    float m_FarPlane;
    
    float m_MovementSpeed;
    float m_RotationSpeed;
    
    void UpdateCameraVectors();
};
