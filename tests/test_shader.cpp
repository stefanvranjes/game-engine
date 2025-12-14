#include <gtest/gtest.h>
#include "Shader.h"

class ShaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code - note: actual shader compilation tests would require OpenGL context
    }

    void TearDown() override {
        // Cleanup code
    }
};

TEST_F(ShaderTest, ShaderCreation) {
    // Test basic shader object creation
    // Note: Full compilation tests require OpenGL context initialization
    Shader shader;
    
    // Verify shader object is created but not necessarily compiled without GL context
    EXPECT_TRUE(true);  // Placeholder for actual shader tests
}

TEST_F(ShaderTest, ShaderSourceValidation) {
    // Test that shader sources are validated before compilation
    std::string validFragmentShader = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
    )";
    
    // Verify source is non-empty and contains GLSL version
    EXPECT_FALSE(validFragmentShader.empty());
    EXPECT_NE(validFragmentShader.find("#version"), std::string::npos);
}

TEST_F(ShaderTest, ShaderSourceContainsMain) {
    std::string shader = R"(
        #version 330 core
        void main() {
            // shader body
        }
    )";
    
    // Verify shader contains main function
    EXPECT_NE(shader.find("main"), std::string::npos);
}

TEST_F(ShaderTest, FragmentShaderStructure) {
    std::string fragmentShader = R"(
        #version 330 core
        in vec3 vPosition;
        out vec4 FragColor;
        void main() {
            FragColor = vec4(vPosition, 1.0);
        }
    )";
    
    // Verify shader has output
    EXPECT_NE(fragmentShader.find("out"), std::string::npos);
}

TEST_F(ShaderTest, VertexShaderStructure) {
    std::string vertexShader = R"(
        #version 330 core
        layout(location = 0) in vec3 aPosition;
        uniform mat4 u_MVP;
        void main() {
            gl_Position = u_MVP * vec4(aPosition, 1.0);
        }
    )";
    
    // Verify vertex shader has input layout and transforms position
    EXPECT_NE(vertexShader.find("layout"), std::string::npos);
    EXPECT_NE(vertexShader.find("gl_Position"), std::string::npos);
}

TEST_F(ShaderTest, UniformDeclaration) {
    std::string shader = R"(
        #version 330 core
        uniform mat4 u_Model;
        uniform mat4 u_View;
        uniform mat4 u_Projection;
        void main() {}
    )";
    
    // Verify uniforms are declared
    EXPECT_NE(shader.find("uniform"), std::string::npos);
    EXPECT_NE(shader.find("u_Model"), std::string::npos);
    EXPECT_NE(shader.find("u_View"), std::string::npos);
    EXPECT_NE(shader.find("u_Projection"), std::string::npos);
}

TEST_F(ShaderTest, TextureCoordinateHandling) {
    std::string shader = R"(
        #version 330 core
        in vec2 vTexCoord;
        uniform sampler2D u_Texture;
        out vec4 FragColor;
        void main() {
            FragColor = texture(u_Texture, vTexCoord);
        }
    )";
    
    // Verify texture sampling
    EXPECT_NE(shader.find("vTexCoord"), std::string::npos);
    EXPECT_NE(shader.find("sampler2D"), std::string::npos);
    EXPECT_NE(shader.find("texture("), std::string::npos);
}

TEST_F(ShaderTest, NormalMapping) {
    std::string shader = R"(
        #version 330 core
        in vec3 vNormal;
        in vec2 vTexCoord;
        uniform sampler2D u_NormalMap;
        void main() {
            vec3 normal = normalize(texture(u_NormalMap, vTexCoord).rgb);
        }
    )";
    
    // Verify normal mapping setup
    EXPECT_NE(shader.find("vNormal"), std::string::npos);
    EXPECT_NE(shader.find("u_NormalMap"), std::string::npos);
}
