#pragma once

#include "Math/Vec3.h"
#include "Math/Vec2.h"
#include "Math/Mat4.h"

struct Ray;
struct GLFWwindow;

class Camera {
public:
    Camera(const Vec3& position = Vec3(0, 0, 3), float fov = 45.0f, float aspect = 16.0f / 9.0f);
    
    Mat4 GetViewMatrix() const;
    Mat4 GetProjectionMatrix() const;
    Mat4 GetPreviousViewProjection() const { return m_PrevViewProjection; }
    
    void SetJitter(const Vec2& jitter) { m_Jitter = jitter; }
    Vec2 GetJitter() const { return m_Jitter; }
    
    void ProcessInput(GLFWwindow* window, float deltaTime);
    void SetAspectRatio(float aspect);
    
    Vec3 GetPosition() const { return m_Position; }
    void SetPosition(const Vec3& position) { m_Position = position; }
    Vec3 GetFront() const { return m_Front; }
    
    void UpdateMatrices();
    
    // Frustum planes: Left, Right, Bottom, Top, Near, Far
    void GetFrustumPlanes(Vec4 planes[6]) const;
    
    /**
     * @brief Create ray from screen coordinates
     * @param screenX Screen X coordinate (0 to screenWidth)
     * @param screenY Screen Y coordinate (0 to screenHeight)
     * @param screenWidth Screen width in pixels
     * @param screenHeight Screen height in pixels
     * @return Ray from camera through screen point
     */
    Ray ScreenPointToRay(float screenX, float screenY, int screenWidth, int screenHeight) const;
    
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
    
    Vec2 m_Jitter;
    Mat4 m_PrevViewProjection;
    
    void UpdateCameraVectors();

public:
    float GetFOV() const { return m_FOV; }
    float GetNearPlane() const { return m_NearPlane; }
    float GetFarPlane() const { return m_FarPlane; }
    float GetAspectRatio() const { return m_AspectRatio; }
};
