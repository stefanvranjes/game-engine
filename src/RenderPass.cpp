#include "RenderPass.h"

RenderPass::RenderPass(const std::string& name, PassType type)
    : m_Name(name), m_Type(type), m_Enabled(true), m_DebugMode(false) {
}

void RenderPass::SetShader(std::unique_ptr<Shader> shader) {
    m_Shader = std::move(shader);
}

RenderPipeline::RenderPipeline(const std::string& name)
    : m_Name(name) {
}

void RenderPipeline::AddPass(std::unique_ptr<RenderPass> pass) {
    if (pass) {
        m_Passes.push_back(std::move(pass));
    }
}

void RenderPipeline::RemovePass(const std::string& passName) {
    auto it = std::find_if(m_Passes.begin(), m_Passes.end(),
        [&passName](const std::unique_ptr<RenderPass>& pass) {
            return pass->GetName() == passName;
        });
    
    if (it != m_Passes.end()) {
        m_Passes.erase(it);
    }
}

RenderPass* RenderPipeline::GetPass(const std::string& passName) const {
    auto it = std::find_if(m_Passes.begin(), m_Passes.end(),
        [&passName](const std::unique_ptr<RenderPass>& pass) {
            return pass->GetName() == passName;
        });
    
    if (it != m_Passes.end()) {
        return it->get();
    }
    return nullptr;
}

void RenderPipeline::Execute(const RenderPass::PassContext& context) {
    for (auto& pass : m_Passes) {
        if (pass->IsEnabled()) {
            pass->Execute(context);
        }
    }
}
