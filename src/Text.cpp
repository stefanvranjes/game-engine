#include "Text.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <iostream>

Text::Text()
    : m_VAO(0)
    , m_VBO(0)
    , m_ScreenWidth(800)
    , m_ScreenHeight(600)
{
}

Text::~Text() {
    if (m_VAO != 0) glDeleteVertexArrays(1, &m_VAO);
    if (m_VBO != 0) glDeleteBuffers(1, &m_VBO);
}

bool Text::Init(const std::string& fontAtlasPath, int screenWidth, int screenHeight) {
    m_ScreenWidth = screenWidth;
    m_ScreenHeight = screenHeight;

    // Load UI shader
    m_Shader = std::make_unique<Shader>();
    if (!m_Shader->LoadFromFiles("shaders/ui.vert", "shaders/ui.frag")) {
        std::cerr << "Failed to load UI shaders" << std::endl;
        return false;
    }

    // Load font atlas
    m_FontAtlas = std::make_unique<Texture>();
    if (!m_FontAtlas->LoadFromFile(fontAtlasPath)) {
        std::cerr << "Failed to load font atlas: " << fontAtlasPath << std::endl;
        return false;
    }

    // Setup orthographic projection
    m_Shader->Use();
    
    // Create orthographic projection matrix (0, width, height, 0)
    float projection[16] = {
        2.0f / m_ScreenWidth, 0.0f, 0.0f, 0.0f,
        0.0f, -2.0f / m_ScreenHeight, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 1.0f
    };
    m_Shader->SetMat4("projection", projection);

    SetupRenderData();

    std::cout << "Text system initialized" << std::endl;
    return true;
}

void Text::SetupRenderData() {
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);

    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Text::RenderText(const std::string& text, float x, float y, float scale, const Vec3& color) {
    // Enable blending for text transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    m_Shader->Use();
    m_Shader->SetVec3("textColor", color.x, color.y, color.z);
    
    // Ensure we're using texture unit 0 (IBL may have changed active texture)
    glActiveTexture(GL_TEXTURE0);
    m_FontAtlas->Bind(0);

    glBindVertexArray(m_VAO);

    float atlasCharWidth = 1.0f / ATLAS_COLS;
    float atlasCharHeight = 1.0f / ATLAS_ROWS;

    for (char c : text) {
        if (c < 32 || c > 126) continue; // Skip non-printable characters

        int charIndex = c - 32; // ASCII 32 is first character in atlas
        int col = charIndex % ATLAS_COLS;
        int row = charIndex / ATLAS_COLS;

        float xpos = x;
        float ypos = y;
        float w = CHAR_WIDTH * scale;
        float h = CHAR_HEIGHT * scale;

        // Texture coordinates
        float texX = col * atlasCharWidth;
        float texY = row * atlasCharHeight;

        float vertices[6][4] = {
            { xpos,     ypos + h,   texX,                   texY + atlasCharHeight },
            { xpos,     ypos,       texX,                   texY },
            { xpos + w, ypos,       texX + atlasCharWidth,  texY },

            { xpos,     ypos + h,   texX,                   texY + atlasCharHeight },
            { xpos + w, ypos,       texX + atlasCharWidth,  texY },
            { xpos + w, ypos + h,   texX + atlasCharWidth,  texY + atlasCharHeight }
        };

        glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        x += w; // Advance to next character position
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);
}

void Text::UpdateProjection(int screenWidth, int screenHeight) {
    m_ScreenWidth = screenWidth;
    m_ScreenHeight = screenHeight;

    m_Shader->Use();
    float projection[16] = {
        2.0f / m_ScreenWidth, 0.0f, 0.0f, 0.0f,
        0.0f, -2.0f / m_ScreenHeight, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 1.0f
    };
    m_Shader->SetMat4("projection", projection);
}
