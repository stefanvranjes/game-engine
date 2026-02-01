#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <glm/glm.hpp>
#include "imgui/imgui.h"

// Forward declarations
class IScriptSystem;
class LuaJitScriptSystem;
class AngelScriptSystem;
class WasmScriptSystem;
class WrenScriptSystem;
class GDScriptSystem;

/**
 * @class ScriptingProfilerUI
 * @brief ImGui-based profiling UI for all scripting systems
 * 
 * Provides comprehensive performance profiling visualization for:
 * - LuaJIT (with JIT coverage metrics)
 * - AngelScript (with optimization stats)
 * - WebAssembly (WASM)
 * - Wren
 * - GDScript
 * - And other scripting languages
 * 
 * Features:
 * - Real-time execution time tracking
 * - Memory usage monitoring
 * - Per-function performance statistics
 * - Call graph visualization
 * - Frame-by-frame profiling
 * - Export to JSON/CSV
 * - Historical data visualization (charts)
 * 
 * Usage:
 * ```cpp
 * auto profiler_ui = std::make_unique<ScriptingProfilerUI>();
 * profiler_ui->Init();
 * 
 * // In RenderEditorUI():
 * profiler_ui->Render();
 * 
 * // In Update():
 * profiler_ui->Update(deltaTime);
 * ```
 */
class ScriptingProfilerUI {
public:
    ScriptingProfilerUI();
    ~ScriptingProfilerUI();

    /**
     * Initialize the profiler UI
     */
    void Init();

    /**
     * Shutdown the profiler UI
     */
    void Shutdown();

    /**
     * Update profiler data (call every frame)
     */
    void Update(float deltaTime);

    /**
     * Render the profiler UI
     * Should be called from RenderEditorUI()
     */
    void Render();

    /**
     * Check if profiler window is open
     */
    bool IsOpen() const { return m_IsOpen; }

    /**
     * Set profiler window visibility
     */
    void SetOpen(bool open) { m_IsOpen = open; }

    /**
     * Show the profiler UI
     */
    void Show() { m_IsOpen = true; }

    /**
     * Hide the profiler UI
     */
    void Hide() { m_IsOpen = false; }

    /**
     * Toggle profiler visibility
     */
    void Toggle() { m_IsOpen = !m_IsOpen; }

    /**
     * Enable/disable profiling for specific language
     */
    void SetLanguageProfilingEnabled(const std::string& language, bool enabled);

    /**
     * Check if profiling is enabled for specific language
     */
    bool IsLanguageProfilingEnabled(const std::string& language) const;

    /**
     * Clear all profiling data
     */
    void ClearData();

    /**
     * Export profiling data to JSON file
     */
    void ExportToJSON(const std::string& filepath);

    /**
     * Export profiling data to CSV file
     */
    void ExportToCSV(const std::string& filepath);

    /**
     * Pause/resume profiling collection
     */
    void SetPaused(bool paused) { m_Paused = paused; }

    /**
     * Check if profiling is paused
     */
    bool IsPaused() const { return m_Paused; }

    /**
     * Get maximum history samples to keep
     */
    size_t GetMaxHistorySamples() const { return m_MaxHistorySamples; }

    /**
     * Set maximum history samples to keep (default: 1000)
     */
    void SetMaxHistorySamples(size_t samples) { m_MaxHistorySamples = samples; }

private:
    // Main render functions
    void RenderMainWindow();
    void RenderOverviewPanel();
    void RenderLanguageDetailsPanel();
    void RenderPerformanceChartsPanel();
    void RenderMemoryStatsPanel();
    void RenderCallGraphPanel();
    void RenderSettingsPanel();

    // Language-specific rendering
    void RenderLuaJitStats();
    void RenderAngelScriptStats();
    void RenderWasmStats();
    void RenderWrenStats();
    void RenderGDScriptStats();
    void RenderOtherLanguageStats();

    // Helper rendering functions
    void RenderExecutionTimeChart(const std::string& language, float width, float height);
    void RenderMemoryUsageChart(const std::string& language, float width, float height);
    void RenderFunctionCallList(const std::string& language);
    void RenderCallGraph(const std::string& language);
    void RenderToolbar();
    void RenderLanguageTabs();
    void RenderLegend();

    // Data collection
    void CollectFrameData();
    void UpdateHistoryData(const std::string& language);
    void SampleLanguageMetrics(const std::string& language);

    // Performance analysis helpers
    struct FunctionStats {
        std::string name;
        uint64_t callCount = 0;
        double totalTime = 0.0;   // milliseconds
        double minTime = 0.0;
        double maxTime = 0.0;
        double avgTime = 0.0;
        size_t samples = 0;
    };

    struct LanguageStats {
        std::string language;
        double totalExecutionTime = 0.0;
        uint64_t totalCallCount = 0;
        size_t memoryUsage = 0;
        bool isAvailable = false;
        bool profilingEnabled = false;
        std::vector<FunctionStats> functionStats;
        std::vector<float> executionTimeHistory;
        std::vector<float> memoryHistory;
        std::chrono::high_resolution_clock::time_point lastSampleTime;
    };

    // Member variables
    bool m_IsOpen = false;
    bool m_Paused = false;
    int m_SelectedLanguageIndex = 0;
    size_t m_MaxHistorySamples = 1000;
    float m_FrameTime = 0.0f;
    float m_RefreshRate = 0.1f;  // Update metrics every 100ms
    float m_TimeSinceLastUpdate = 0.0f;

    std::unordered_map<std::string, LanguageStats> m_LanguageStats;
    std::vector<std::string> m_AvailableLanguages;

    // UI state
    static const ImVec4 CHART_COLOR_EXEC_TIME;
    static const ImVec4 CHART_COLOR_MEMORY;
    static const ImVec4 CHART_COLOR_CALLS;
    static const ImVec4 COLOR_HEADER_BG;
    static const ImVec4 COLOR_ROW_BG;

    // Temporary UI data
    std::unordered_map<std::string, bool> m_ExpandedFunctions;
    bool m_ShowExecutionChart = true;
    bool m_ShowMemoryChart = true;
    bool m_ShowCallGraph = false;
    bool m_AutoRefresh = true;
    bool m_SortByTime = true;
};
