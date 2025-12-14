#pragma once

#include <memory>
#include "AssetImporter.h"
#include "AnimationWorkflow.h"
#include "BuildPipeline.h"
#include "LevelEditor.h"
#include "ShaderCompiler.h"
#include "PerformanceProfiler.h"

namespace Tools {

class WorkflowManager {
public:
    static WorkflowManager& Get();
    
    // Initialize all tools
    void Initialize();
    void Shutdown();
    
    // Tool accessors
    ShaderCompiler* GetShaderCompiler() { return m_ShaderCompiler.get(); }
    LevelEditor* GetLevelEditor() { return m_LevelEditor.get(); }
    BuildPipeline* GetBuildPipeline() { return m_BuildPipeline.get(); }
    AnimationWorkflow* GetAnimationWorkflow() { return m_AnimationWorkflow.get(); }
    PerformanceProfiler& GetProfiler() { return PerformanceProfiler::Get(); }
    
    // Update called each frame
    void Update();

private:
    std::unique_ptr<ShaderCompiler> m_ShaderCompiler;
    std::unique_ptr<LevelEditor> m_LevelEditor;
    std::unique_ptr<BuildPipeline> m_BuildPipeline;
    std::unique_ptr<AnimationWorkflow> m_AnimationWorkflow;
};

} // namespace Tools