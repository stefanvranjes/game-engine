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
    
    // TODO: Validate all assets in asset directory
    
    return true;
}

bool BuildPipeline::CompileShaders(const std::string& shaderDir, const std::string& outputDir) {
    ReportProgress(0.3f, "Compiling Shaders");
    std::cout << "Compiling shaders from: " << shaderDir << std::endl;
    
    // TODO: Compile all shaders to SPIR-V or platform-specific format
    
    return true;
}

bool BuildPipeline::PackAssets(const std::string& assetDir, const std::string& outputDir) {
    ReportProgress(0.5f, "Packing Assets");
    std::cout << "Packing assets to: " << outputDir << std::endl;
    
    // TODO: Compress and package all assets
    
    return true;
}

bool BuildPipeline::BuildExecutable(BuildTarget target) {
    std::string targetName = (target == BuildTarget::Debug) ? "Debug" : 
                            (target == BuildTarget::Release) ? "Release" : "Shipping";
    ReportProgress(0.7f, "Building Executable (" + targetName + ")");
    std::cout << "Building executable for target: " << targetName << std::endl;
    
    // TODO: Compile engine and game code
    // Use CMake or Visual Studio project files
    
    return true;
}

bool BuildPipeline::GeneratePackage(Platform platform, const std::string& outputPath) {
    std::string platformName = (platform == Platform::Windows) ? "Windows" :
                              (platform == Platform::Linux) ? "Linux" : "macOS";
    ReportProgress(0.9f, "Generating Package (" + platformName + ")");
    std::cout << "Generating package for: " << platformName << std::endl;
    
    // TODO: Create distributable package for target platform
    
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