#include "Shader.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

Shader::Shader() : m_ProgramID(0), m_IsComputeShader(false) {
}

Shader::~Shader() {
    if (m_ProgramID != 0) {
        glDeleteProgram(m_ProgramID);
    }
}

std::string Shader::ReadFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filepath << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool Shader::CompileShader(unsigned int shader, const std::string& source, const std::string& type) {
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed (" << type << "):\n" << infoLog << std::endl;
        return false;
    }

    return true;
}

bool Shader::LinkProgram() {
    glLinkProgram(m_ProgramID);

    int success;
    glGetProgramiv(m_ProgramID, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(m_ProgramID, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
        return false;
    }

    return true;
}

bool Shader::LoadFromFiles(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath) {
    // Read shader source code
    std::string vertexCode = ReadFile(vertexPath);
    std::string fragmentCode = ReadFile(fragmentPath);
    std::string geometryCode;
    
    if (!geometryPath.empty()) {
        geometryCode = ReadFile(geometryPath);
    }

    if (vertexCode.empty() || fragmentCode.empty() || (!geometryPath.empty() && geometryCode.empty())) {
        return false;
    }

    // Create shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    unsigned int geometryShader = 0;
    
    if (!geometryPath.empty()) {
        geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
    }

    // Compile shaders
    if (!CompileShader(vertexShader, vertexCode, "VERTEX")) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        if (geometryShader) glDeleteShader(geometryShader);
        return false;
    }

    if (!CompileShader(fragmentShader, fragmentCode, "FRAGMENT")) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        if (geometryShader) glDeleteShader(geometryShader);
        return false;
    }
    
    if (!geometryPath.empty()) {
        if (!CompileShader(geometryShader, geometryCode, "GEOMETRY")) {
            glDeleteShader(vertexShader);
            glDeleteShader(fragmentShader);
            glDeleteShader(geometryShader);
            return false;
        }
    }

    // Create program and link
    m_ProgramID = glCreateProgram();
    glAttachShader(m_ProgramID, vertexShader);
    glAttachShader(m_ProgramID, fragmentShader);
    if (!geometryPath.empty()) {
        glAttachShader(m_ProgramID, geometryShader);
    }

    if (!LinkProgram()) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        if (geometryShader) glDeleteShader(geometryShader);
        glDeleteProgram(m_ProgramID);
        m_ProgramID = 0;
        return false;
    }

    // Clean up shaders (they're linked into the program now)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    if (geometryShader) glDeleteShader(geometryShader);

    // Store paths and timestamp
    m_VertexPath = vertexPath;
    m_FragmentPath = fragmentPath;
    m_GeometryPath = geometryPath;
    
    m_LastWriteTime = GetFileTimestamp(vertexPath);
    m_LastWriteTime = std::max(m_LastWriteTime, GetFileTimestamp(fragmentPath));
    if (!geometryPath.empty()) {
        m_LastWriteTime = std::max(m_LastWriteTime, GetFileTimestamp(geometryPath));
    }
    
    m_IsComputeShader = false;

    std::cout << "Shader program created successfully" << std::endl;
    return true;
}

long long Shader::GetFileTimestamp(const std::string& filepath) {
    struct stat result;
    if (stat(filepath.c_str(), &result) == 0) {
        return result.st_mtime;
    }
    return 0;
}

void Shader::CheckForUpdates() {
    if (m_IsComputeShader) {
        // Check compute shader for updates
        long long currentTime = GetFileTimestamp(m_ComputePath);
        
        if (currentTime > m_LastWriteTime) {
            std::cout << "Reloading compute shader..." << std::endl;
            
            unsigned int oldProgram = m_ProgramID;
            
            if (LoadComputeShader(m_ComputePath)) {
                glDeleteProgram(oldProgram);
                std::cout << "Compute shader reloaded successfully!" << std::endl;
            } else {
                m_ProgramID = oldProgram;
                std::cerr << "Compute shader reload failed! Keeping old shader." << std::endl;
                m_LastWriteTime = currentTime;
            }
        }
    } else {
        // Check regular shaders for updates
        long long currentMaxTime = GetFileTimestamp(m_VertexPath);
        currentMaxTime = std::max(currentMaxTime, GetFileTimestamp(m_FragmentPath));
        if (!m_GeometryPath.empty()) {
            currentMaxTime = std::max(currentMaxTime, GetFileTimestamp(m_GeometryPath));
        }
        
        if (currentMaxTime > m_LastWriteTime) {
            std::cout << "Reloading shader..." << std::endl;
            
            // Keep old ID in case reload fails
            unsigned int oldProgram = m_ProgramID;
            
            // Attempt reload
            if (LoadFromFiles(m_VertexPath, m_FragmentPath, m_GeometryPath)) {
                // Success! Delete old program
                glDeleteProgram(oldProgram);
                std::cout << "Shader reloaded successfully!" << std::endl;
            } else {
                // Failed! Restore old program
                m_ProgramID = oldProgram;
                std::cerr << "Shader reload failed! Keeping old shader." << std::endl;
                // Update timestamp anyway to avoid loop
                m_LastWriteTime = currentMaxTime;
            }
        }
    }
}

bool Shader::LoadComputeShader(const std::string& computePath) {
    // Check if compute shaders are supported
    if (!glDispatchCompute || !glMemoryBarrier) {
        std::cerr << "Compute shaders not supported (OpenGL 4.3+ required)" << std::endl;
        return false;
    }
    
    // Read shader source code
    std::string computeCode = ReadFile(computePath);
    
    if (computeCode.empty()) {
        return false;
    }
    
    // Create compute shader
    unsigned int computeShader = glCreateShader(GL_COMPUTE_SHADER);
    
    // Compile shader
    if (!CompileShader(computeShader, computeCode, "COMPUTE")) {
        glDeleteShader(computeShader);
        return false;
    }
    
    // Create program and link
    m_ProgramID = glCreateProgram();
    glAttachShader(m_ProgramID, computeShader);
    
    if (!LinkProgram()) {
        glDeleteShader(computeShader);
        glDeleteProgram(m_ProgramID);
        m_ProgramID = 0;
        return false;
    }
    
    // Clean up shader (it's linked into the program now)
    glDeleteShader(computeShader);
    
    // Store path and timestamp
    m_ComputePath = computePath;
    m_LastWriteTime = GetFileTimestamp(computePath);
    m_IsComputeShader = true;
    
    std::cout << "Compute shader program created successfully" << std::endl;
    return true;
}

void Shader::Dispatch(unsigned int numGroupsX, unsigned int numGroupsY, unsigned int numGroupsZ) const {
    if (!m_IsComputeShader) {
        std::cerr << "Error: Dispatch called on non-compute shader" << std::endl;
        return;
    }
    
    if (!glDispatchCompute) {
        std::cerr << "Error: glDispatchCompute not available" << std::endl;
        return;
    }
    
    glDispatchCompute(numGroupsX, numGroupsY, numGroupsZ);
}

void Shader::GetWorkGroupSize(int& x, int& y, int& z) const {
    if (!m_IsComputeShader) {
        std::cerr << "Error: GetWorkGroupSize called on non-compute shader" << std::endl;
        x = y = z = 0;
        return;
    }
    
    if (!glGetIntegeri_v) {
        std::cerr << "Error: glGetIntegeri_v not available" << std::endl;
        x = y = z = 0;
        return;
    }
    
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &x);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &y);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &z);
}

void Shader::Use() const {
    glUseProgram(m_ProgramID);
}

int Shader::GetUniformLocation(const std::string& name) {
    if (m_UniformLocationCache.find(name) != m_UniformLocationCache.end()) {
        return m_UniformLocationCache[name];
    }

    int location = glGetUniformLocation(m_ProgramID, name.c_str());
    m_UniformLocationCache[name] = location;
    return location;
}

void Shader::SetInt(const std::string& name, int value) {
    glUniform1i(GetUniformLocation(name), value);
}

void Shader::SetFloat(const std::string& name, float value) {
    glUniform1f(GetUniformLocation(name), value);
}

void Shader::SetVec2(const std::string& name, float x, float y) {
    glUniform2f(GetUniformLocation(name), x, y);
}

void Shader::SetVec3(const std::string& name, float x, float y, float z) {
    glUniform3f(GetUniformLocation(name), x, y, z);
}

void Shader::SetVec4(const std::string& name, float x, float y, float z, float w) {
    glUniform4f(GetUniformLocation(name), x, y, z, w);
}

void Shader::SetMat4(const std::string& name, const float* value) {
    glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, value);
}
