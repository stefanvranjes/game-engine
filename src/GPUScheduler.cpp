#include "GPUScheduler.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

// ============= RenderGraphNode Implementation =============

// ============= RenderGraph Implementation =============

void RenderGraph::AddPass(
    const std::string& name,
    const RenderPassRequirements& requirements,
    std::function<void()> callback)
{
    auto node = std::make_shared<RenderGraphNode>(name);
    node->requirements = requirements;
    node->onExecute = callback;

    GraphNodeInternal internal;
    internal.node = node;
    m_Nodes[name] = internal;

    SPDLOG_DEBUG("Added render pass: {}", name);
}

void RenderGraph::AddDependency(const std::string& passA, const std::string& passB) {
    if (m_Nodes.find(passA) == m_Nodes.end() || m_Nodes.find(passB) == m_Nodes.end()) {
        SPDLOG_WARN("Invalid dependency: {} -> {}", passA, passB);
        return;
    }

    m_Nodes[passA].dependents.push_back(passB);
    m_Nodes[passB].dependencies.push_back(passA);
}

void RenderGraph::Compile(uint32_t gpuCount) {
    m_GPUCount = gpuCount;
    m_GPULoads.resize(gpuCount, 0.0f);

    // Topologically sort passes
    auto sortedPasses = TopoSort();

    // Calculate estimated frame time
    float maxLoad = 0.0f;
    for (const auto& passName : sortedPasses) {
        maxLoad += m_Nodes[passName].node->requirements.estimatedGPUTime;
    }
    m_EstimatedFrameTime = maxLoad;

    // Balance load across GPUs
    BalanceLoad(sortedPasses);

    SPDLOG_INFO("Render graph compiled: {} passes, {} GPUs, ~{:.2f}ms estimated frame time",
                m_Nodes.size(), gpuCount, m_EstimatedFrameTime);
}

void RenderGraph::Execute() {
    auto sortedPasses = TopoSort();

    for (const auto& passName : sortedPasses) {
        auto& nodeInternal = m_Nodes[passName];
        auto& node = nodeInternal.node;

        // Wait for dependencies
        for (const auto& depName : nodeInternal.dependencies) {
            if (!m_Nodes[depName].executed) {
                SPDLOG_WARN("Dependency not executed: {}", depName);
            }
        }

        // Record execution time
        nodeInternal.startTime = std::chrono::high_resolution_clock::now();

        // Execute the pass
        if (node->onExecute) {
            node->onExecute();
        }

        // Calculate elapsed time
        auto endTime = std::chrono::high_resolution_clock::now();
        float elapsedMs = std::chrono::duration<float, std::milli>(endTime - nodeInternal.startTime).count();
        m_Stats.perPassTimes[passName] = elapsedMs;

        nodeInternal.executed = true;
        m_Stats.totalTime += elapsedMs;

        SPDLOG_TRACE("Executed pass: {} ({:.3f}ms on GPU{})",
                     passName, elapsedMs, node->assignedGPU);
    }

    // Calculate load balance metric
    if (!m_GPULoads.empty()) {
        float avgLoad = m_Stats.totalTime / m_GPUCount;
        float maxLoad = *std::max_element(m_GPULoads.begin(), m_GPULoads.end());
        m_Stats.loadBalance = (maxLoad - avgLoad) / avgLoad; // 0 = perfect, higher = worse
    }
}

void RenderGraph::Flush() {
    // All passes executed, nothing more to do
    // In a real implementation, would wait for GPU commands to complete
}

void RenderGraph::ResetMetrics() {
    for (auto& node : m_Nodes) {
        node.second.executed = false;
    }
    m_Stats = ExecutionStats();
    std::fill(m_GPULoads.begin(), m_GPULoads.end(), 0.0f);
}

std::vector<std::string> RenderGraph::TopoSort() {
    std::vector<std::string> sorted;
    std::map<std::string, int> inDegree;

    // Calculate in-degrees
    for (const auto& node : m_Nodes) {
        inDegree[node.first] = node.second.dependencies.size();
    }

    // Find all nodes with in-degree 0
    std::vector<std::string> queue;
    for (const auto& node : m_Nodes) {
        if (inDegree[node.first] == 0) {
            queue.push_back(node.first);
        }
    }

    // Kahn's algorithm
    while (!queue.empty()) {
        std::string current = queue.front();
        queue.erase(queue.begin());
        sorted.push_back(current);

        for (const auto& dependent : m_Nodes[current].dependents) {
            inDegree[dependent]--;
            if (inDegree[dependent] == 0) {
                queue.push_back(dependent);
            }
        }
    }

    if (sorted.size() != m_Nodes.size()) {
        SPDLOG_ERROR("Render graph has cycles!");
    }

    return sorted;
}

void RenderGraph::BalanceLoad(const std::vector<std::string>& sortedPasses) {
    std::fill(m_GPULoads.begin(), m_GPULoads.end(), 0.0f);

    for (const auto& passName : sortedPasses) {
        auto& node = m_Nodes[passName].node;
        float passTime = node->requirements.estimatedGPUTime;

        // Find GPU with least current load
        uint32_t selectedGPU = 0;
        float minLoad = m_GPULoads[0];

        for (uint32_t i = 1; i < m_GPUCount; i++) {
            if (m_GPULoads[i] < minLoad) {
                minLoad = m_GPULoads[i];
                selectedGPU = i;
            }
        }

        // Check if pass can be split across GPUs
        if (node->requirements.supportsSplitFrame && m_UseSplitFrame && m_GPUCount > 1) {
            // Distribute evenly
            for (uint32_t i = 0; i < m_GPUCount; i++) {
                m_GPULoads[i] += passTime / m_GPUCount;
            }
            node->assignedGPU = 0; // Mark as multi-GPU
        } else {
            m_GPULoads[selectedGPU] += passTime;
            node->assignedGPU = selectedGPU;
        }
    }
}

// ============= GPUScheduler Implementation =============

GPUScheduler::GPUScheduler(RenderBackend* backend)
    : m_Backend(backend),
      m_RenderGraph(std::make_unique<RenderGraph>())
{
}

GPUScheduler::~GPUScheduler() = default;

void GPUScheduler::Init() {
    DetectGPUs();
    ProfileGPUCapabilities();

    SPDLOG_INFO("GPU Scheduler initialized with {} GPU(s)", m_GPUCount);
    SPDLOG_INFO("Total GPU compute power: {:.2f} TFlops", m_TotalTFlops);

    for (uint32_t i = 0; i < m_GPUCount; i++) {
        SPDLOG_INFO("  GPU {}: {} ({:.2f} TFlops)",
                    i, m_GPUs[i].info.name, m_GPUs[i].info.estimatedPeakTFlops);
    }
}

const GPUDeviceInfo& GPUScheduler::GetGPUInfo(uint32_t index) const {
    if (index >= m_GPUs.size()) {
        static GPUDeviceInfo dummy;
        return dummy;
    }
    return m_GPUs[index].info;
}

uint32_t GPUScheduler::SelectGPU(RenderPassType workloadType, float estimatedTime) {
    if (!m_MultiGPUEnabled || m_GPUCount <= 1) {
        return 0;
    }

    uint32_t selectedGPU = 0;
    float minLoad = m_GPUs[0].currentLoad;

    for (uint32_t i = 1; i < m_GPUCount; i++) {
        if (!m_GPUs[i].isAvailable) continue;
        if (m_GPUs[i].currentLoad < minLoad) {
            minLoad = m_GPUs[i].currentLoad;
            selectedGPU = i;
        }
    }

    // Update estimated load
    m_GPUs[selectedGPU].estimatedCompleteTime =
        std::max(m_GPUs[selectedGPU].estimatedCompleteTime,
                 (float)std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now().time_since_epoch()).count() + estimatedTime);

    return selectedGPU;
}

void GPUScheduler::MarkPassComplete(uint32_t gpuIndex, float elapsedTime) {
    if (gpuIndex >= m_GPUs.size()) return;
    m_GPUs[gpuIndex].currentLoad -= elapsedTime;
    m_GPUs[gpuIndex].currentLoad = std::max(0.0f, m_GPUs[gpuIndex].currentLoad);
}

std::vector<float> GPUScheduler::GetGPUUtilizations() const {
    std::vector<float> utilizations;
    for (const auto& gpu : m_GPUs) {
        float util = 0.0f;
        if (gpu.info.estimatedPeakTFlops > 0.0f) {
            util = (gpu.currentLoad / 16.67f) * 100.0f; // Assuming 60 FPS = 16.67ms budget
            util = std::min(100.0f, util);
        }
        utilizations.push_back(util);
    }
    return utilizations;
}

GPUScheduler::Strategy GPUScheduler::RecommendStrategy() const {
    if (m_GPUCount <= 1) {
        return Strategy::Single;
    }

    // If we have 2 GPUs of similar performance, suggest split-frame
    if (m_GPUCount == 2) {
        float perf0 = m_GPUs[0].info.estimatedPeakTFlops;
        float perf1 = m_GPUs[1].info.estimatedPeakTFlops;
        float ratio = std::max(perf0, perf1) / std::min(perf0, perf1);

        if (ratio < 1.3f) { // Within 30% performance
            return Strategy::SplitFrame;
        }
    }

    // Default to alternate frame for multi-GPU setups
    return Strategy::AlternateFrame;
}

void GPUScheduler::SetForcedGPUCount(uint32_t count) {
    m_ForcedGPUCount = count;
    if (count > 0) {
        m_GPUCount = std::min(count, (uint32_t)m_GPUs.size());
    }
}

void GPUScheduler::DetectGPUs() {
    if (m_ForcedGPUCount > 0) {
        m_GPUCount = std::min(m_ForcedGPUCount, m_Backend->GetDeviceCount());
    } else {
        m_GPUCount = m_Backend->GetDeviceCount();
    }

    m_GPUs.clear();
    m_TotalTFlops = 0.0f;

    for (uint32_t i = 0; i < m_GPUCount; i++) {
        GPUMetrics metric;
        metric.info = m_Backend->GetDeviceInfo(i);
        metric.isAvailable = true;
        m_GPUs.push_back(metric);
        m_TotalTFlops += metric.info.estimatedPeakTFlops;
    }
}

void GPUScheduler::ProfileGPUCapabilities() {
    for (auto& gpu : m_GPUs) {
        // In a real implementation, would run benchmarks to measure actual performance
        // For now, estimate based on device properties
        if (gpu.info.estimatedPeakTFlops == 0.0f) {
            // Default estimate: varies by GPU type
            gpu.info.estimatedPeakTFlops = 50.0f; // Conservative estimate
        }
    }
}

