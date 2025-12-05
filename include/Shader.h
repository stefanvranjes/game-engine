#pragma once

#include <string>
#include <unordered_map>

class Shader {
public:
    Shader();
    ~Shader();

    bool LoadFromFiles(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath = "");
    bool LoadComputeShader(const std::string& computePath);
    void Use() const;
    void Dispatch(unsigned int numGroupsX, unsigned int numGroupsY, unsigned int numGroupsZ) const;
    
    // Uniform setters
    void SetInt(const std::string& name, int value);
    void SetFloat(const std::string& name, float value);
    void SetVec2(const std::string& name, float x, float y);
    void SetVec3(const std::string& name, float x, float y, float z);
    void SetVec4(const std::string& name, float x, float y, float z, float w);
    void SetMat4(const std::string& name, const float* value);

    unsigned int GetProgramID() const { return m_ProgramID; }
    bool IsComputeShader() const { return m_IsComputeShader; }
    void GetWorkGroupSize(int& x, int& y, int& z) const;
    
    // Hot-Reload
    void CheckForUpdates();

private:
    bool CompileShader(unsigned int shader, const std::string& source, const std::string& type);
    bool LinkProgram();
    std::string ReadFile(const std::string& filepath);
    int GetUniformLocation(const std::string& name);
    long long GetFileTimestamp(const std::string& filepath);

    unsigned int m_ProgramID;
    std::unordered_map<std::string, int> m_UniformLocationCache;
    bool m_IsComputeShader;
    
    // Hot-Reload State
    std::string m_VertexPath;
    std::string m_FragmentPath;
    std::string m_GeometryPath;
    std::string m_ComputePath;
    long long m_LastWriteTime;
};
