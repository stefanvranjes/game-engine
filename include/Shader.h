#pragma once

#include <string>
#include <unordered_map>

class Shader {
public:
    Shader();
    ~Shader();

    bool LoadFromFiles(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath = "");
    void Use() const;
    
    // Uniform setters
    void SetInt(const std::string& name, int value);
    void SetFloat(const std::string& name, float value);
    void SetVec3(const std::string& name, float x, float y, float z);
    void SetVec4(const std::string& name, float x, float y, float z, float w);
    void SetMat4(const std::string& name, const float* value);

    unsigned int GetProgramID() const { return m_ProgramID; }

private:
    bool CompileShader(unsigned int shader, const std::string& source, const std::string& type);
    bool LinkProgram();
    std::string ReadFile(const std::string& filepath);
    int GetUniformLocation(const std::string& name);

    unsigned int m_ProgramID;
    std::unordered_map<std::string, int> m_UniformLocationCache;
};
