#pragma once

#include "GameObject.h"
#include "Shader.h"
#include "Camera.h"
#include "Light.h"
#include <memory>

class PreviewRenderer {
public:
    PreviewRenderer();
    ~PreviewRenderer();

    bool Init(int width = 512, int height = 512);
    void RenderPreview(GameObject* object, Shader* shader);
    void Shutdown();

    unsigned int GetTextureID() const { return m_ColorTexture; }
    int GetWidth() const { return m_Width; }
    int GetHeight() const { return m_Height; }

private:
    unsigned int m_FBO;
    unsigned int m_ColorTexture;
    unsigned int m_DepthRenderbuffer;
    
    int m_Width;
    int m_Height;
    
    std::unique_ptr<Camera> m_Camera;
    std::unique_ptr<Shader> m_Shader;
    float m_RotationAngle;
};
