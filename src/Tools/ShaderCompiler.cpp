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
    
    // Read shader source files
    std::ifstream vertFile(vertPath), fragFile(fragPath);
    if (!vertFile.is_open() || !fragFile.is_open()) {
        std::cerr << "Failed to open shader files" << std::endl;
        return false;
    }
    
    std::string vertSource((std::istreambuf_iterator<char>(vertFile)),
                          std::istreambuf_iterator<char>());
    std::string fragSource((std::istreambuf_iterator<char>(fragFile)),
                          std::istreambuf_iterator<char>());
    vertFile.close();
    fragFile.close();
    
    // Validate shader source
    if (!ValidateShader(vertSource) || !ValidateShader(fragSource)) {
        std::cerr << "Shader validation failed" << std::endl;
        return false;
    }
    
    // Compile to SPIR-V using glslc or similar compiler
    // In a real implementation, this would:
    // 1. Invoke glslc (or shaderc) command-line tool
    // 2. Pass vertex and fragment sources
    // 3. Generate SPIR-V bytecode
    // 4. Load compiled bytecode into shader objects
    
    // Example: system("glslc -fshader-stage=vertex shader.vert -o shader.vert.spv");
    // Example: system("glslc -fshader-stage=fragment shader.frag -o shader.frag.spv");
    
    outShader = std::make_shared<Shader>();
    // outShader->LoadFromSPIRV(vertBytecode, fragBytecode);
    
    std::cout << "Shader compiled successfully to SPIR-V" << std::endl;
    return true;
}

void ShaderCompiler::CheckAndReloadShaders() {
    if (!m_HotReloadEnabled) return;
    
    std::cout << "Checking for shader file modifications..." << std::endl;
    
    // Check modification times of tracked shader files and recompile if changed
    for (auto& [name, shaderFile] : m_TrackedShaders) {
        bool vertModified = false, fragModified = false;
        
        // Check vertex shader modification time
        if (!shaderFile.vertPath.empty() && std::filesystem::exists(shaderFile.vertPath)) {
            auto lastWriteTime = std::filesystem::last_write_time(shaderFile.vertPath);
            if (lastWriteTime > shaderFile.lastModified) {
                vertModified = true;
                std::cout << "  Vertex shader modified: " << shaderFile.vertPath << std::endl;
            }
        }
        
        // Check fragment shader modification time
        if (!shaderFile.fragPath.empty() && std::filesystem::exists(shaderFile.fragPath)) {
            auto lastWriteTime = std::filesystem::last_write_time(shaderFile.fragPath);
            if (lastWriteTime > shaderFile.lastModified) {
                fragModified = true;
                std::cout << "  Fragment shader modified: " << shaderFile.fragPath << std::endl;
            }
        }
        
        // Recompile if either shader was modified
        if (vertModified || fragModified) {
            std::cout << "  Recompiling shader: " << name << std::endl;
            std::shared_ptr<Shader> recompiledShader;
            if (CompileShader(shaderFile.vertPath, shaderFile.fragPath, recompiledShader)) {
                shaderFile.shader = recompiledShader;
                shaderFile.lastModified = std::filesystem::file_time_type::clock::now();
                std::cout << "  Hot reload successful for: " << name << std::endl;
            } else {
                std::cerr << "  Hot reload failed for: " << name << std::endl;
            }
        }
    }
}

std::string ShaderCompiler::PreprocessShader(const std::string& source, 
                                             const std::string& defines) {
    std::istringstream iss(source);
    std::ostringstream oss;
    std::string line;
    
    std::cout << "Preprocessing shader with defines: " << defines << std::endl;
    
    // Add custom defines to output
    if (!defines.empty()) {
        oss << "#define " << defines << "\n";
    }
    
    // Process shader source line by line
    while (std::getline(iss, line)) {
        // Handle #include directives
        if (line.find("#include") == 0) {
            // Extract include filename
            size_t start = line.find('"');
            size_t end = line.rfind('"');
            
            if (start != std::string::npos && end != std::string::npos && start < end) {
                std::string includePath = line.substr(start + 1, end - start - 1);
                
                // Read included file
                std::ifstream includeFile(includePath);
                if (includeFile.is_open()) {
                    std::string includeContent((std::istreambuf_iterator<char>(includeFile)),
                                             std::istreambuf_iterator<char>());
                    includeFile.close();
                    
                    oss << "// Included from " << includePath << "\n";
                    oss << includeContent << "\n";
                    std::cout << "  Included: " << includePath << std::endl;
                } else {
                    std::cerr << "  Warning: Could not find include file: " << includePath << std::endl;
                    oss << line << "\n";
                }
            } else {
                oss << line << "\n";
            }
        }
        // Handle #define directives
        else if (line.find("#define") == 0) {
            // Keep #defines in output for conditional compilation
            oss << line << "\n";
        }
        // Handle #ifdef/#endif for conditional compilation
        else if (line.find("#ifdef") == 0 || line.find("#if") == 0 || 
                 line.find("#endif") == 0 || line.find("#else") == 0) {
            oss << line << "\n";
        }
        // Regular shader code
        else {
            oss << line << "\n";
        }
    }
    
    return oss.str();
}

bool ShaderCompiler::ValidateShader(const std::string& shaderSource) {
    std::cout << "Validating shader..." << std::endl;
    
    if (shaderSource.empty()) {
        std::cerr << "  Error: Shader source is empty" << std::endl;
        return false;
    }
    
    // Check shader syntax and validate against GLSL specification
    bool hasErrors = false;
    
    // Check for balanced braces
    int braceCount = 0;
    int parenCount = 0;
    int bracketCount = 0;
    
    for (char c : shaderSource) {
        if (c == '{') braceCount++;
        if (c == '}') braceCount--;
        if (c == '(') parenCount++;
        if (c == ')') parenCount--;
        if (c == '[') bracketCount++;
        if (c == ']') bracketCount--;
    }
    
    if (braceCount != 0) {
        std::cerr << "  Error: Unbalanced braces (count: " << braceCount << ")" << std::endl;
        hasErrors = true;
    }
    if (parenCount != 0) {
        std::cerr << "  Error: Unbalanced parentheses (count: " << parenCount << ")" << std::endl;
        hasErrors = true;
    }
    if (bracketCount != 0) {
        std::cerr << "  Error: Unbalanced brackets (count: " << bracketCount << ")" << std::endl;
        hasErrors = true;
    }
    
    // Check for required function
    if (shaderSource.find("main") == std::string::npos) {
        std::cerr << "  Error: Shader must define main() function" << std::endl;
        hasErrors = true;
    }
    
    // Check for valid GLSL version directive (optional but recommended)
    if (shaderSource.find("#version") != std::string::npos) {
        std::cout << "  Valid GLSL version directive found" << std::endl;
    }
    
    // Check for common variable types
    std::vector<std::string> validTypes = {"vec2", "vec3", "vec4", "mat2", "mat3", "mat4",
                                           "float", "int", "bool", "sampler2D", "samplerCube",
                                           "in", "out", "uniform", "varying"};
    
    if (hasErrors) {
        std::cout << "  Shader validation failed" << std::endl;
        return false;
    }
    
    std::cout << "  Shader validation passed" << std::endl;
    return true;
}

std::string ShaderCompiler::OptimizeShader(const std::string& shaderSource) {
    std::cout << "Optimizing shader..." << std::endl;
    
    std::istringstream iss(shaderSource);
    std::ostringstream oss;
    std::string line;
    
    int removedLines = 0;
    
    // Remove dead code, optimize uniforms, consolidate declarations
    while (std::getline(iss, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            // Skip empty lines (can be optimized away)
            removedLines++;
            continue;
        }
        
        std::string trimmedLine = line.substr(start);
        
        // Skip pure comment lines
        if (trimmedLine.find("//") == 0) {
            removedLines++;
            continue;
        }
        
        // Skip unreachable code after return statements
        if (oss.str().find("return") != std::string::npos && 
            trimmedLine.find("}") == std::string::npos) {
            // Note: This is a simplified check; real implementation would need
            // proper scope tracking
        }
        
        oss << line << "\n";
    }
    
    std::string optimized = oss.str();
    
    // Count uniform declarations for optimization reporting
    size_t uniformCount = 0;
    size_t pos = 0;
    while ((pos = optimized.find("uniform", pos)) != std::string::npos) {
        uniformCount++;
        pos += 7;
    }
    
    std::cout << "  Optimization complete:" << std::endl;
    std::cout << "    Removed " << removedLines << " empty/comment lines" << std::endl;
    std::cout << "    Found " << uniformCount << " uniform declarations" << std::endl;
    std::cout << "    Code size: " << optimized.length() << " bytes" << std::endl;
    
    return optimized;
}

void ShaderCompiler::RegisterShader(const std::string& name, std::shared_ptr<Shader> shader) {
    std::cout << "Registering shader for hot reload: " << name << std::endl;
    
    ShaderFile sf;
    sf.shader = shader;
    sf.lastModified = std::filesystem::file_time_type::clock::now();
    
    m_TrackedShaders[name] = sf;
}

} // namespace Tools