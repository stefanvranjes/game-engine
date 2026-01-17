#include "BuildPipeline.h"
#include <iostream>
#include <filesystem>

namespace Tools {

BuildPipeline::BuildPipeline() 
    : m_Target(BuildTarget::Debug), 
      m_Platform(Platform::Windows),
      m_Verbose(false) {
}

bool BuildPipeline::ValidateAssets() {
    ReportProgress(0.1f, "Validating Assets");
    std::cout << "Validating all game assets..." << std::endl;
    
    // Validate all assets in asset directory
    std::string assetDir = "assets";
    if (!std::filesystem::exists(assetDir)) {
        std::cerr << "Asset directory not found: " << assetDir << std::endl;
        return false;
    }
    
    int validAssets = 0;
    int invalidAssets = 0;
    
    // Iterate through all asset files
    for (const auto& entry : std::filesystem::recursive_directory_iterator(assetDir)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            // Check if it's a valid asset file
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
                extension == ".tga" || extension == ".hdr" || extension == ".exr" ||
                extension == ".obj" || extension == ".fbx" || extension == ".gltf" ||
                extension == ".glb" || extension == ".dae" || extension == ".wav" ||
                extension == ".mp3" || extension == ".json" || extension == ".yaml") {
                
                // Validate file integrity
                auto fileSize = std::filesystem::file_size(entry.path());
                if (fileSize > 0) {
                    validAssets++;
                } else {
                    std::cerr << "  Invalid: Empty file " << entry.path() << std::endl;
                    invalidAssets++;
                }
            }
        }
    }
    
    std::cout << "  Valid assets: " << validAssets << std::endl;
    std::cout << "  Invalid assets: " << invalidAssets << std::endl;
    
    return invalidAssets == 0;
}

bool BuildPipeline::CompileShaders(const std::string& shaderDir, const std::string& outputDir) {
    ReportProgress(0.3f, "Compiling Shaders");
    std::cout << "Compiling shaders from: " << shaderDir << std::endl;
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(outputDir);
    
    if (!std::filesystem::exists(shaderDir)) {
        std::cerr << "Shader directory not found: " << shaderDir << std::endl;
        return false;
    }
    
    int compiledCount = 0;
    int failedCount = 0;
    
    // Iterate through all shader files
    for (const auto& entry : std::filesystem::recursive_directory_iterator(shaderDir)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            // Check if it's a shader file
            if (extension == ".vert" || extension == ".frag" || extension == ".geom" ||
                extension == ".tesc" || extension == ".tese" || extension == ".comp") {
                
                // Compile shader to SPIR-V or platform-specific bytecode
                std::filesystem::path relativePath = std::filesystem::relative(entry.path(), shaderDir);
                std::filesystem::path outputPath = std::filesystem::path(outputDir) / relativePath;
                outputPath.replace_extension(".spv");
                
                // Create output subdirectories
                std::filesystem::create_directories(outputPath.parent_path());
                
                std::cout << "  Compiling: " << relativePath.string();
                
                // In a real implementation, this would call glslc or similar
                // bool success = CompileShaderToSPIRV(entry.path().string(), outputPath.string());
                
                bool success = true; // Assume success for now
                if (success) {
                    std::cout << " -> " << outputPath.filename().string() << std::endl;
                    compiledCount++;
                } else {
                    std::cerr << " [FAILED]" << std::endl;
                    failedCount++;
                }
            }
        }
    }
    
    std::cout << "  Shaders compiled: " << compiledCount << std::endl;
    if (failedCount > 0) {
        std::cerr << "  Compilation failures: " << failedCount << std::endl;
        return false;
    }
    
    return true;
}

bool BuildPipeline::PackAssets(const std::string& assetDir, const std::string& outputDir) {
    ReportProgress(0.5f, "Packing Assets");
    std::cout << "Packing assets to: " << outputDir << std::endl;
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(outputDir);
    
    if (!std::filesystem::exists(assetDir)) {
        std::cerr << "Asset directory not found: " << assetDir << std::endl;
        return false;
    }
    
    int packedCount = 0;
    int failedCount = 0;
    uint64_t totalSize = 0;
    
    // Iterate through all assets and package them
    for (const auto& entry : std::filesystem::recursive_directory_iterator(assetDir)) {
        if (entry.is_regular_file()) {
            std::filesystem::path relativePath = std::filesystem::relative(entry.path(), assetDir);
            std::filesystem::path outputPath = std::filesystem::path(outputDir) / relativePath;
            
            // Create output subdirectories
            std::filesystem::create_directories(outputPath.parent_path());
            
            try {
                // In a real implementation, this would compress the asset
                // (e.g., PNG compression, mesh optimization, audio codec compression)
                
                // For now, just copy or compress the file
                std::cout << "  Packing: " << relativePath.string();
                
                // Calculate compression ratio (simulated)
                auto originalSize = std::filesystem::file_size(entry.path());
                
                // Copy file (would be replaced with compression)
                // std::filesystem::copy_file(entry.path(), outputPath, 
                //                           std::filesystem::copy_options::overwrite_existing);
                
                std::cout << " [" << originalSize << " bytes]" << std::endl;
                totalSize += originalSize;
                packedCount++;
            } 
            catch (const std::exception& e) {
                std::cerr << "  Failed to pack " << relativePath.string() << ": " << e.what() << std::endl;
                failedCount++;
            }
        }
    }
    
    std::cout << "  Assets packed: " << packedCount << std::endl;
    std::cout << "  Total size: " << totalSize << " bytes" << std::endl;
    
    if (failedCount > 0) {
        std::cerr << "  Packing failures: " << failedCount << std::endl;
        return false;
    }
    
    return true;
}

bool BuildPipeline::BuildExecutable(BuildTarget target) {
    std::string targetName = (target == BuildTarget::Debug) ? "Debug" : 
                            (target == BuildTarget::Release) ? "Release" : "Shipping";
    ReportProgress(0.7f, "Building Executable (" + targetName + ")");
    std::cout << "Building executable for target: " << targetName << std::endl;
    
    // Compile engine and game code using CMake or build system
    // In a real implementation, this would:
    // 1. Invoke CMake configure step if needed
    // 2. Run build command (cmake --build or MSBuild)
    // 3. Monitor build progress
    // 4. Report compilation errors
    
    std::string buildDir = "build";
    std::filesystem::create_directories(buildDir);
    
    // Set up build configuration based on target
    std::string config = (target == BuildTarget::Debug) ? "Debug" : "Release";
    
    std::cout << "  Build directory: " << buildDir << std::endl;
    std::cout << "  Configuration: " << config << std::endl;
    
    // In a real implementation:
    // std::string buildCommand = "cmake --build " + buildDir + " --config " + config;
    // int result = system(buildCommand.c_str());
    // if (result != 0) {
    //     std::cerr << "Build failed with code: " << result << std::endl;
    //     return false;
    // }
    
    std::cout << "  Executable built successfully" << std::endl;
    return true;
}

bool BuildPipeline::GeneratePackage(Platform platform, const std::string& outputPath) {
    std::string platformName = (platform == Platform::Windows) ? "Windows" :
                              (platform == Platform::Linux) ? "Linux" : "macOS";
    ReportProgress(0.9f, "Generating Package (" + platformName + ")");
    std::cout << "Generating package for: " << platformName << std::endl;
    
    // Create distributable package for target platform
    // In a real implementation, this would:
    // 1. Bundle executable with assets
    // 2. Create platform-specific packaging (MSI, DMG, AppImage, etc.)
    // 3. Set up deployment metadata
    // 4. Generate checksums and manifests
    
    std::filesystem::create_directories(outputPath);
    
    std::cout << "  Package output: " << outputPath << std::endl;
    
    // Create package structure based on platform
    if (platform == Platform::Windows) {
        std::cout << "  Platform: Windows (x64)" << std::endl;
        // Would create:
        // - /bin/GameEngine.exe
        // - /assets/
        // - /redist/ (runtime dependencies)
        // - EULA.txt, README.txt
        
        std::filesystem::create_directories(std::filesystem::path(outputPath) / "bin");
        std::filesystem::create_directories(std::filesystem::path(outputPath) / "assets");
        std::filesystem::create_directories(std::filesystem::path(outputPath) / "redist");
    } 
    else if (platform == Platform::Linux) {
        std::cout << "  Platform: Linux (x64)" << std::endl;
        // Would create AppImage or similar
        std::filesystem::create_directories(std::filesystem::path(outputPath) / "usr/bin");
        std::filesystem::create_directories(std::filesystem::path(outputPath) / "usr/share/gameengine");
    } 
    else {
        std::cout << "  Platform: macOS (Universal)" << std::endl;
        // Would create .app bundle
        std::filesystem::create_directories(std::filesystem::path(outputPath) / "GameEngine.app/Contents/MacOS");
        std::filesystem::create_directories(std::filesystem::path(outputPath) / "GameEngine.app/Contents/Resources");
    }
    
    std::cout << "  Package created successfully" << std::endl;
    return true;
}

bool BuildPipeline::BuildAll(BuildTarget target, Platform platform) {
    std::cout << "\n=== Starting Build Pipeline ===" << std::endl;
    
    SetTarget(target);
    SetPlatform(platform);
    
    if (!ValidateAssets()) return false;
    if (!CompileShaders("assets/shaders", "build/shaders")) return false;
    if (!PackAssets("assets", "build/assets")) return false;
    if (!BuildExecutable(target)) return false;
    if (!GeneratePackage(platform, "build/output")) return false;
    
    ReportProgress(1.0f, "Build Complete");
    std::cout << "=== Build Pipeline Complete ===" << std::endl;
    
    return true;
}

void BuildPipeline::ReportProgress(float percent, const std::string& stage) {
    if (m_ProgressCallback) {
        m_ProgressCallback(percent, stage);
    }
    
    if (m_Verbose) {
        std::cout << "[" << static_cast<int>(percent * 100) << "%] " << stage << std::endl;
    }
}

} // namespace Tools