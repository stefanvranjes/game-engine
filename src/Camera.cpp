#include "Camera.h"
#include <GLFW/glfw3.h>
#include <cmath>

#define PI 3.14159265359f

Camera::Camera(const Vec3& position, float fov, float aspect)
    : m_Position(position)
    , m_WorldUp(0, 1, 0)
    , m_Yaw(-90.0f)
    , m_Pitch(0.0f)
    , m_FOV(fov)
    , m_AspectRatio(aspect)
    , m_NearPlane(0.1f)
    , m_FarPlane(100.0f)
    , m_MovementSpeed(2.5f)
    , m_RotationSpeed(45.0f)
{
    UpdateCameraVectors();
}

Mat4 Camera::GetViewMatrix() const {
    return Mat4::LookAt(m_Position, m_Position + m_Front, m_Up);
}

Mat4 Camera::GetProjectionMatrix() const {
    return Mat4::Perspective(m_FOV * PI / 180.0f, m_AspectRatio, m_NearPlane, m_FarPlane);
}

void Camera::ProcessInput(GLFWwindow* window, float deltaTime) {
    float velocity = m_MovementSpeed * deltaTime;
    float rotation = m_RotationSpeed * deltaTime;

    // Movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        m_Position += m_Front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        m_Position -= m_Front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        m_Position -= m_Right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        m_Position += m_Right * velocity;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        m_Position += m_WorldUp * velocity;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        m_Position -= m_WorldUp * velocity;

    // Rotation
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        m_Yaw -= rotation;
        UpdateCameraVectors();
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        m_Yaw += rotation;
        UpdateCameraVectors();
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        m_Pitch += rotation;
        if (m_Pitch > 89.0f) m_Pitch = 89.0f;
        UpdateCameraVectors();
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        m_Pitch -= rotation;
        if (m_Pitch < -89.0f) m_Pitch = -89.0f;
        UpdateCameraVectors();
    }
}

void Camera::SetAspectRatio(float aspect) {
    m_AspectRatio = aspect;
}

void Camera::UpdateCameraVectors() {
    // Calculate new front vector
    Vec3 front;
    front.x = std::cos(m_Yaw * PI / 180.0f) * std::cos(m_Pitch * PI / 180.0f);
    front.y = std::sin(m_Pitch * PI / 180.0f);
    front.z = std::sin(m_Yaw * PI / 180.0f) * std::cos(m_Pitch * PI / 180.0f);
    m_Front = front.Normalized();
    
    // Recalculate right and up vectors
    m_Right = m_Front.Cross(m_WorldUp).Normalized();
    m_Up = m_Right.Cross(m_Front).Normalized();
}
