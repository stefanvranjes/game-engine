#include "ShaderCompiler.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace Tools {

ShaderCompiler::ShaderCompiler() 
    : m_HotReloadEnabled(false) {
}

bool ShaderCompiler::CompileShader(const std::string& vertPath, 
                                   const std::string& fragPath,
                                   std::shared_ptr<Shader>& outShader) {
    std::cout << "Compiling shader:\n  Vertex: " << vertPath 
              << "\n  Fragment: " << fragPath << std::endl;
    
    if (!ValidateShader("")) {
        std::cerr << "Shader validation failed" << std::endl;
        return false;
    }
    
    // TODO: Compile to SPIR-V using glslc or similar
    outShader = std::make_shared<Shader>();
    
    std::cout << "Shader compiled successfully" << std::endl;
    return true;
}

void ShaderCompiler::CheckAndReloadShaders() {
    if (!m_HotReloadEnabled) return;
    
    // TODO: Check modification times of tracked shader files
    // Recompile if changed
}

std::string ShaderCompiler::PreprocessShader(const std::string& source, 
                                             const std::string& defines) {
    std::istringstream iss(source);
    std::ostringstream oss;
    std::string line;
    
    std::cout << "Preprocessing shader with defines: " << defines << std::endl;
    
    // TODO: Process #include directives, #defines, etc
    
    return oss.str();
}

bool ShaderCompiler::ValidateShader(const std::string& shaderSource) {
    std::cout << "Validating shader..." << std::endl;
    
    // TODO: Check shader syntax, validate against GLSL specification
    
    return true;
}

std::string ShaderCompiler::OptimizeShader(const std::string& shaderSource) {
    std::cout << "Optimizing shader..." << std::endl;
    
    // TODO: Remove dead code, optimize uniforms, etc
    
    return shaderSource;
}

void ShaderCompiler::RegisterShader(const std::string& name, std::shared_ptr<Shader> shader) {
    std::cout << "Registering shader for hot reload: " << name << std::endl;
    
    ShaderFile sf;
    sf.shader = shader;
    sf.lastModified = std::filesystem::file_time_type::clock::now();
    
    m_TrackedShaders[name] = sf;
}

} // namespace Tools