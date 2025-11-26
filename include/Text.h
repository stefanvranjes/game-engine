#pragma once

#include <string>
#include <memory>
#include "Shader.h"
#include "Texture.h"
#include "Math/Vec3.h"

class Text {
public:
    Text();
    ~Text();

    bool Init(const std::string& fontAtlasPath, int screenWidth, int screenHeight);
    void RenderText(const std::string& text, float x, float y, float scale, const Vec3& color);
    
    void UpdateProjection(int screenWidth, int screenHeight);

private:
    void SetupRenderData();

    std::unique_ptr<Shader> m_Shader;
    std::unique_ptr<Texture> m_FontAtlas;
    
    unsigned int m_VAO;
    unsigned int m_VBO;
    
    int m_ScreenWidth;
    int m_ScreenHeight;
    
    // Font atlas configuration (16x6 grid for ASCII 32-126)
    static constexpr int ATLAS_COLS = 16;
    static constexpr int ATLAS_ROWS = 6;
    static constexpr int CHAR_WIDTH = 8;  // pixels per character in atlas
    static constexpr int CHAR_HEIGHT = 16;
};
