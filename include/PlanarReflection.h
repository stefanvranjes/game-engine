#pragma once

#include "Math/Vec3.h"
#include "Math/Vec4.h"
#include "Math/Mat4.h"

class Camera;

/**
 * @brief Manages planar reflections for flat reflective surfaces (water, mirrors, etc.)
 * 
 * This class handles:
 * - Reflection framebuffer creation and management
 * - Reflected camera matrix computation
 * - Oblique clip plane calculation for proper clipping
 */
class PlanarReflection {
public:
    PlanarReflection();
    ~PlanarReflection();

    /**
     * @brief Initialize the reflection framebuffer
     * @param width Initial texture width
     * @param height Initial texture height
     * @return true if initialization successful
     */
    bool Init(int width, int height);

    /**
     * @brief Resize the reflection texture
     * @param width New width
     * @param height New height
     */
    void Resize(int width, int height);

    /**
     * @brief Set the reflection plane
     * @param point A point on the plane (typically water surface position)
     * @param normal The plane normal (typically (0,1,0) for horizontal water)
     */
    void SetPlane(const Vec3& point, const Vec3& normal);

    /**
     * @brief Compute the view matrix for reflected camera
     * @param camera The main camera to reflect
     * @return View matrix from reflected camera position
     */
    Mat4 GetReflectedView(const Camera* camera) const;

    /**
     * @brief Get the clip plane in view space for oblique frustum clipping
     * @param view The view matrix being used
     * @return Clip plane as Vec4 (A, B, C, D) where Ax + By + Cz + D = 0
     */
    Vec4 GetClipPlane(const Mat4& view) const;
    
    /**
     * @brief Get the clip plane in world space
     */
    Vec4 GetClipPlane() const {
        return Vec4(m_PlaneNormal.x, m_PlaneNormal.y, m_PlaneNormal.z, -m_PlaneNormal.Dot(m_PlanePoint));
    }

    /**
     * @brief Check if reflection is currently rendering
     */
    bool IsRendering() const { return m_IsRendering; }
    void SetRendering(bool rendering) { m_IsRendering = rendering; }

    /**
     * @brief Bind the reflection FBO for rendering
     */
    void BindForWriting();

    /**
     * @brief Bind the reflection texture for reading
     * @param textureUnit The texture unit to bind to
     */
    void BindForReading(int textureUnit);

    /**
     * @brief Unbind framebuffer (return to default)
     */
    void Unbind();

    /**
     * @brief Get the reflection texture ID
     * @return OpenGL texture ID
     */
    unsigned int GetReflectionTexture() const { return m_ReflectionTexture; }

    /**
     * @brief Get the current texture width
     */
    int GetWidth() const { return m_Width; }

    /**
     * @brief Get the current texture height
     */
    int GetHeight() const { return m_Height; }

    // Public settings
    float m_ReflectionDistortionStrength = 0.02f;  ///< Distortion amount for wave effect
    float m_ReflectionClarity = 1.0f;              ///< Blend factor with environment
    bool m_Enabled = true;                          ///< Enable/disable planar reflections

private:
    void Cleanup();
    void CreateResources();

    unsigned int m_FBO = 0;               ///< Framebuffer object
    unsigned int m_ReflectionTexture = 0; ///< Color attachment for reflection
    unsigned int m_DepthRBO = 0;          ///< Depth renderbuffer

    Vec3 m_PlanePoint;   ///< A point on the reflection plane
    Vec3 m_PlaneNormal;  ///< Normal of the reflection plane

    int m_Width = 0;
    int m_Height = 0;
    bool m_Initialized = false;
    bool m_IsRendering = false;
};
