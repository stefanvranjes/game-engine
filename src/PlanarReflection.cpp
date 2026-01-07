#include "PlanarReflection.h"
#include "Camera.h"
#include "GLExtensions.h"
#include <iostream>

PlanarReflection::PlanarReflection()
    : m_PlanePoint(0.0f, 0.0f, 0.0f)
    , m_PlaneNormal(0.0f, 1.0f, 0.0f) // Default: horizontal plane facing up
{
}

PlanarReflection::~PlanarReflection() {
    Cleanup();
}

void PlanarReflection::Cleanup() {
    if (m_FBO) {
        glDeleteFramebuffers(1, &m_FBO);
        m_FBO = 0;
    }
    if (m_ReflectionTexture) {
        glDeleteTextures(1, &m_ReflectionTexture);
        m_ReflectionTexture = 0;
    }
    if (m_DepthRBO) {
        glDeleteRenderbuffers(1, &m_DepthRBO);
        m_DepthRBO = 0;
    }
    m_Initialized = false;
}

bool PlanarReflection::Init(int width, int height) {
    if (width <= 0 || height <= 0) {
        std::cerr << "PlanarReflection::Init: Invalid dimensions " << width << "x" << height << std::endl;
        return false;
    }

    m_Width = width;
    m_Height = height;

    CreateResources();

    std::cout << "PlanarReflection initialized at " << m_Width << "x" << m_Height << std::endl;
    return m_Initialized;
}

void PlanarReflection::CreateResources() {
    Cleanup();

    // Create framebuffer
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    // Create reflection color texture
    glGenTextures(1, &m_ReflectionTexture);
    glBindTexture(GL_TEXTURE_2D, m_ReflectionTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_Width, m_Height, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_ReflectionTexture, 0);

    // Create depth renderbuffer
    glGenRenderbuffers(1, &m_DepthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, m_DepthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_Width, m_Height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_DepthRBO);

    // Check framebuffer completeness
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "PlanarReflection: Framebuffer incomplete, status: " << status << std::endl;
        Cleanup();
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    m_Initialized = true;
}

void PlanarReflection::Resize(int width, int height) {
    if (width <= 0 || height <= 0) return;
    if (width == m_Width && height == m_Height) return;

    m_Width = width;
    m_Height = height;
    CreateResources();

    std::cout << "PlanarReflection resized to " << m_Width << "x" << m_Height << std::endl;
}

void PlanarReflection::SetPlane(const Vec3& point, const Vec3& normal) {
    m_PlanePoint = point;
    m_PlaneNormal = normal.Normalized();
}

Mat4 PlanarReflection::GetReflectedView(const Camera* camera) const {
    if (!camera) {
        return Mat4::Identity();
    }

    // Get camera properties
    Vec3 camPos = camera->GetPosition();
    Vec3 camFront = camera->GetFront();
    
    // Compute reflection of camera position across the plane
    // Distance from camera to plane
    float d = (camPos - m_PlanePoint).Dot(m_PlaneNormal);
    
    // Reflected position: P' = P - 2 * d * N
    Vec3 reflectedPos = camPos - m_PlaneNormal * (2.0f * d);
    
    // Reflect the look direction
    // For direction: D' = D - 2 * (D . N) * N
    float dn = camFront.Dot(m_PlaneNormal);
    Vec3 reflectedFront = camFront - m_PlaneNormal * (2.0f * dn);
    
    // Compute up vector - also needs to be reflected
    Vec3 worldUp(0.0f, 1.0f, 0.0f);
    float upDot = worldUp.Dot(m_PlaneNormal);
    Vec3 reflectedUp = worldUp - m_PlaneNormal * (2.0f * upDot);
    
    // For a horizontal water plane (normal = 0,1,0), the up vector reflection
    // simply flips Y: (0,1,0) -> (0,-1,0)
    // This is correct for maintaining proper orientation
    
    // Build look-at matrix from reflected position
    Vec3 target = reflectedPos + reflectedFront;
    
    return Mat4::LookAt(reflectedPos, target, reflectedUp);
}

Vec4 PlanarReflection::GetClipPlane(const Mat4& /*view*/) const {
    // Plane equation in world space: N.x * x + N.y * y + N.z * z + D = 0
    // where D = -N . P (P is a point on the plane)
    float d = -m_PlaneNormal.Dot(m_PlanePoint);
    Vec4 planeWorld(m_PlaneNormal.x, m_PlaneNormal.y, m_PlaneNormal.z, d);
    
    // Transform plane to view space
    // For a plane, we need to use the inverse transpose of the view matrix
    // But for clip planes, we want to transform points, so we use:
    // plane_view = (M^-1)^T * plane_world
    
    // However, for OpenGL clip planes, we need the plane in clip space
    // The standard approach: transform plane by inverse-transpose of MVP
    
    // For simplicity with glClipDistance, we'll return the world-space plane
    // and let the vertex shader handle the transformation
    
    // Offset slightly above the plane to avoid z-fighting
    float offset = 0.001f;
    return Vec4(m_PlaneNormal.x, m_PlaneNormal.y, m_PlaneNormal.z, d + offset);
}

void PlanarReflection::BindForWriting() {
    if (!m_Initialized) return;
    
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glViewport(0, 0, m_Width, m_Height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void PlanarReflection::BindForReading(int textureUnit) {
    if (!m_Initialized) return;
    
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_2D, m_ReflectionTexture);
}

void PlanarReflection::Unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
