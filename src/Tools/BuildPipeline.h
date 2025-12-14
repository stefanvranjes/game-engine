#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace Tools {

enum class BuildTarget {
    Debug,
    Release,
    Shipping
};

enum class Platform {
    Windows,
    Linux,
    macOS
};

class BuildPipeline {
public:
    BuildPipeline();
    
    // Pipeline stages
    bool ValidateAssets();
    bool CompileShaders(const std::string& shaderDir, const std::string& outputDir);
    bool PackAssets(const std::string& assetDir, const std::string& outputDir);
    bool BuildExecutable(BuildTarget target);
    bool GeneratePackage(Platform platform, const std::string& outputPath);
    
    // Configuration
    void SetTarget(BuildTarget target) { m_Target = target; }
    void SetPlatform(Platform platform) { m_Platform = platform; }
    void SetVerbose(bool verbose) { m_Verbose = verbose; }
    
    // Callbacks for progress tracking
    void SetProgressCallback(std::function<void(float, const std::string&)> callback) {
        m_ProgressCallback = callback;
    }
    
    // Full build
    bool BuildAll(BuildTarget target, Platform platform);

private:
    BuildTarget m_Target;
    Platform m_Platform;
    bool m_Verbose;
    std::function<void(float, const std::string&)> m_ProgressCallback;
    
    void ReportProgress(float percent, const std::string& stage);
};

} // namespace Tools