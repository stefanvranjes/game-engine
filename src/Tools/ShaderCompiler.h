#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include "../Rendering/Shader.h"

namespace Tools {

class ShaderCompiler {
public:
    ShaderCompiler();
    
    // Compile individual shader
    bool CompileShader(const std::string& vertPath, 
                       const std::string& fragPath,
                       std::shared_ptr<Shader>& outShader);
    
    // Hot reload on file change
    void EnableHotReload(bool enable) { m_HotReloadEnabled = enable; }
    void CheckAndReloadShaders();
    
    // Shader preprocessor
    std::string PreprocessShader(const std::string& source, 
                                 const std::string& defines = "");
    
    // Validation & optimization
    bool ValidateShader(const std::string& shaderSource);
    std::string OptimizeShader(const std::string& shaderSource);
    
    // Register shader for tracking
    void RegisterShader(const std::string& name, std::shared_ptr<Shader> shader);

private:
    struct ShaderFile {
        std::string vertPath;
        std::string fragPath;
        std::filesystem::file_time_type lastModified;
        std::shared_ptr<Shader> shader;
    };
    
    std::unordered_map<std::string, ShaderFile> m_TrackedShaders;
    bool m_HotReloadEnabled;
};

} // namespace Tools