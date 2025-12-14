#include "WorkflowManager.h"
#include <iostream>

namespace Tools {

WorkflowManager& WorkflowManager::Get() {
    static WorkflowManager instance;
    return instance;
}

void WorkflowManager::Initialize() {
    std::cout << "Initializing Tools & Workflow Manager..." << std::endl;
    
    m_ShaderCompiler = std::make_unique<ShaderCompiler>();
    m_ShaderCompiler->EnableHotReload(true);
    
    m_LevelEditor = std::make_unique<LevelEditor>();
    
    m_BuildPipeline = std::make_unique<BuildPipeline>();
    m_BuildPipeline->SetVerbose(true);
    
    m_AnimationWorkflow = std::make_unique<AnimationWorkflow>();
    
    std::cout << "Tools & Workflow Manager initialized successfully" << std::endl;
}

void WorkflowManager::Shutdown() {
    std::cout << "Shutting down Tools & Workflow Manager..." << std::endl;
    
    m_ShaderCompiler.reset();
    m_LevelEditor.reset();
    m_BuildPipeline.reset();
    m_AnimationWorkflow.reset();
}

void WorkflowManager::Update() {
    // Check for shader hot reload
    if (m_ShaderCompiler) {
        m_ShaderCompiler->CheckAndReloadShaders();
    }
    
    // Update performance profiler
    PerformanceProfiler::Get().EndFrame();
}

} // namespace Tools