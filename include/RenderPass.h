#pragma once

#include <string>
#include <memory>
#include <glm/glm.hpp>
#include "Shader.h"
#include "Camera.h"

/**
 * @class RenderPass
 * @brief Abstract base class for Scriptable Render Pipeline (SRP)-like render passes
 * 
 * Each render pass represents a distinct stage in the rendering pipeline (shadows, 
 * geometry, lighting, post-processing, etc.). This follows Unity's SRP design pattern
 * to make the renderer more modular and extensible.
 */
class RenderPass {
public:
    enum class PassType {
        Shadow,              // Shadow map generation
        Geometry,            // G-Buffer fill (deferred)
        Lighting,            // Lighting computation
        Transparent,         // Forward-rendered transparency
        PostProcessing,      // Post-processing effects
        Composite,           // Final output composition
        Custom               // User-defined passes
    };

    struct PassContext {
        Camera* camera;
        int viewportWidth;
        int viewportHeight;
        float deltaTime;
        glm::mat4 viewMatrix;
        glm::mat4 projectionMatrix;
        void* userData; // For custom data per pass
    };

    explicit RenderPass(const std::string& name, PassType type);
    virtual ~RenderPass() = default;

    // Core lifecycle methods
    virtual bool Initialize() = 0;
    virtual void Execute(const PassContext& context) = 0;
    virtual void Shutdown() = 0;

    // Shader management
    virtual void SetShader(std::unique_ptr<Shader> shader);
    Shader* GetShader() const { return m_Shader.get(); }

    // State queries
    const std::string& GetName() const { return m_Name; }
    PassType GetType() const { return m_Type; }
    bool IsEnabled() const { return m_Enabled; }
    void SetEnabled(bool enabled) { m_Enabled = enabled; }

    // Debug rendering
    virtual void SetDebugMode(bool enabled) { m_DebugMode = enabled; }
    bool GetDebugMode() const { return m_DebugMode; }

protected:
    std::string m_Name;
    PassType m_Type;
    std::unique_ptr<Shader> m_Shader;
    bool m_Enabled;
    bool m_DebugMode;
};

/**
 * @class RenderPipeline
 * @brief Orchestrates a sequence of RenderPass instances
 * 
 * The pipeline manages the execution order of passes and provides a clean
 * abstraction for multi-pass rendering with dependency management.
 */
class RenderPipeline {
public:
    RenderPipeline(const std::string& name);
    ~RenderPipeline() = default;

    // Pass management
    void AddPass(std::unique_ptr<RenderPass> pass);
    void RemovePass(const std::string& passName);
    RenderPass* GetPass(const std::string& passName) const;

    // Execution
    void Execute(const RenderPass::PassContext& context);

    // Configuration
    const std::string& GetName() const { return m_Name; }
    size_t GetPassCount() const { return m_Passes.size(); }

private:
    std::string m_Name;
    std::vector<std::unique_ptr<RenderPass>> m_Passes;
};
