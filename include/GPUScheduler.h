#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include "RenderBackend.h"
#include "glm/glm.hpp"

/**
 * @brief Render pass type for scheduling decisions
 */
enum class RenderPassType {
    Geometry,        // G-Buffer generation - heavy GPU load
    Lighting,        // Deferred lighting - medium load
    ShadowMap,       // Shadow map rendering - high load
    PostProcessing,  // Post-process effects - low to medium
    Transparent,     // Forward rendering - variable
    Compute,         // Compute shader dispatches
    Transfer,        // CPU-GPU transfers
    Present          // Final presentation
};

/**
 * @brief Estimated resource requirements for a render pass
 */
struct RenderPassRequirements {
    RenderPassType type;
    std::string name;
    
    // Estimated execution time on GPU (ms)
    float estimatedGPUTime = 0.0f;
    
    // Memory requirements
    uint64_t peakMemoryUsage = 0;
    uint64_t bandwidthRequired = 0;
    
    // Threading hints
    bool needsAsync = false;
    bool supportsSplitFrame = true;
    
    // Dependencies
    std::vector<std::string> dependsOn;
};

/**
 * @brief Single node in the render graph
 */
class RenderGraphNode {
public:
    std::string name;
    RenderPassRequirements requirements;
    
    // Execution state
    uint32_t assignedGPU = 0;
    float estimatedCompletionTime = 0.0f;
    
    // Callbacks
    std::function<void()> onExecute;
    std::function<void()> onComplete;
    
    explicit RenderGraphNode(const std::string& nodeName)
        : name(nodeName) {}
};

/**
 * @brief Directed acyclic graph (DAG) of render passes
 * 
 * Manages render pass dependencies, schedules execution across GPUs,
 * and handles synchronization between passes.
 */
class RenderGraph {
public:
    /**
     * @brief Add a render pass to the graph
     * @param name Unique name for this pass
     * @param requirements Resource and timing requirements
     * @param callback Function to execute this pass
     */
    void AddPass(
        const std::string& name,
        const RenderPassRequirements& requirements,
        std::function<void()> callback);

    /**
     * @brief Add dependency between two passes
     * Pass B cannot start until pass A completes
     */
    void AddDependency(const std::string& passA, const std::string& passB);

    /**
     * @brief Enable split-frame rendering (multi-GPU optimization)
     * Each GPU renders different screen regions
     */
    void EnableSplitFrame(bool enable) { m_UseSplitFrame = enable; }

    /**
     * @brief Enable alternate-frame rendering (multi-GPU optimization)
     * GPU0 renders even frames, GPU1 renders odd frames
     */
    void EnableAlternateFrame(bool enable) { m_UseAlternateFrame = enable; }

    /**
     * @brief Compile graph structure and optimize for multi-GPU execution
     * Call after all passes and dependencies are defined
     */
    void Compile(uint32_t gpuCount);

    /**
     * @brief Execute the entire render graph
     * Respects dependencies and distributes work across available GPUs
     */
    void Execute();

    /**
     * @brief Wait for all pending passes to complete
     */
    void Flush();

    /**
     * @brief Get total estimated frame time
     */
    float GetEstimatedFrameTime() const { return m_EstimatedFrameTime; }

    /**
     * @brief Get load per GPU as percentage (0-100)
     */
    std::vector<float> GetGPULoads() const { return m_GPULoads; }

    /**
     * @brief Reset profiling data for next frame
     */
    void ResetMetrics();

    /**
     * @brief Get detailed execution statistics
     */
    struct ExecutionStats {
        float totalTime = 0.0f;
        float longestPass = 0.0f;
        float loadBalance = 0.0f; // 0.0 = perfect, 1.0 = terrible
        std::map<std::string, float> perPassTimes;
    };
    
    ExecutionStats GetStats() const { return m_Stats; }

private:
    struct GraphNodeInternal {
        std::shared_ptr<RenderGraphNode> node;
        std::vector<std::string> dependencies;
        std::vector<std::string> dependents;
        bool executed = false;
        std::chrono::high_resolution_clock::time_point startTime;
    };

    std::map<std::string, GraphNodeInternal> m_Nodes;
    std::vector<float> m_GPULoads;
    float m_EstimatedFrameTime = 0.0f;
    ExecutionStats m_Stats;

    bool m_UseSplitFrame = false;
    bool m_UseAlternateFrame = false;
    uint32_t m_GPUCount = 1;

    // Topological sort for dependency resolution
    std::vector<std::string> TopoSort();

    // Assign passes to GPUs based on load
    void BalanceLoad(const std::vector<std::string>& sortedPasses);
};

/**
 * @brief Manages GPU device selection and load balancing
 * 
 * Automatically detects GPUs, profiles their capabilities,
 * and distributes rendering workload for optimal performance.
 */
class GPUScheduler {
public:
    explicit GPUScheduler(RenderBackend* backend);
    ~GPUScheduler();

    /**
     * @brief Initialize GPU detection and profiling
     */
    void Init();

    /**
     * @brief Get number of available GPUs
     */
    uint32_t GetGPUCount() const { return m_GPUCount; }

    /**
     * @brief Get info about a specific GPU
     */
    const GPUDeviceInfo& GetGPUInfo(uint32_t index) const;

    /**
     * @brief Get aggregate GPU power (TFlops) for scheduling
     */
    float GetTotalTFlops() const { return m_TotalTFlops; }

    /**
     * @brief Select GPU based on workload characteristics
     * @param workloadType Type of workload (geometry, lighting, etc.)
     * @param estimatedTime Estimated GPU time required (ms)
     * @return GPU index to use
     */
    uint32_t SelectGPU(RenderPassType workloadType, float estimatedTime);

    /**
     * @brief Mark GPU workload complete for a pass
     * Updates load tracking for next frame
     */
    void MarkPassComplete(uint32_t gpuIndex, float elapsedTime);

    /**
     * @brief Get current estimated GPU utilization
     * @return Vector of utilization percentages (0-100) per GPU
     */
    std::vector<float> GetGPUUtilizations() const;

    /**
     * @brief Get suggested rendering strategy for multi-GPU
     */
    enum class Strategy {
        Single,        // Single GPU (baseline)
        SplitFrame,    // Each GPU renders screen region
        AlternateFrame // Each GPU renders alternate frames
    };
    
    Strategy RecommendStrategy() const;

    /**
     * @brief Enable or disable multi-GPU optimization
     */
    void SetMultiGPUEnabled(bool enabled) { m_MultiGPUEnabled = enabled; }

    /**
     * @brief Force specific GPU count (for testing)
     */
    void SetForcedGPUCount(uint32_t count);

    /**
     * @brief Get access to render graph for advanced scheduling
     */
    RenderGraph* GetRenderGraph() { return m_RenderGraph.get(); }

private:
    struct GPUMetrics {
        GPUDeviceInfo info;
        float currentLoad = 0.0f;
        float estimatedCompleteTime = 0.0f; // Absolute time, not elapsed
        bool isAvailable = true;
    };

    RenderBackend* m_Backend = nullptr;
    uint32_t m_GPUCount = 0;
    float m_TotalTFlops = 0.0f;
    std::vector<GPUMetrics> m_GPUs;
    std::unique_ptr<RenderGraph> m_RenderGraph;
    
    bool m_MultiGPUEnabled = true;
    uint32_t m_ForcedGPUCount = 0; // 0 = auto-detect

    void DetectGPUs();
    void ProfileGPUCapabilities();
};

