#include "ScriptingProfilerUI.h"
#include "LuaJitScriptSystem.h"
#include "AngelScriptSystem.h"
#include "Wasm/WasmScriptSystem.h"
#include "WrenScriptSystem.h"
#include "GDScriptSystem.h"
#include "IScriptSystem.h"
#include <imgui.h>
#include <imgui_stdlib.h>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Color scheme for charts and UI
const ImVec4 ScriptingProfilerUI::CHART_COLOR_EXEC_TIME(0.2f, 0.8f, 0.2f, 0.8f);  // Green
const ImVec4 ScriptingProfilerUI::CHART_COLOR_MEMORY(0.2f, 0.6f, 1.0f, 0.8f);      // Blue
const ImVec4 ScriptingProfilerUI::CHART_COLOR_CALLS(1.0f, 0.8f, 0.2f, 0.8f);       // Yellow
const ImVec4 ScriptingProfilerUI::COLOR_HEADER_BG(0.3f, 0.3f, 0.3f, 0.7f);         // Dark gray
const ImVec4 ScriptingProfilerUI::COLOR_ROW_BG(0.15f, 0.15f, 0.15f, 0.5f);         // Darker gray

ScriptingProfilerUI::ScriptingProfilerUI()
    : m_IsOpen(false), m_Paused(false), m_SelectedLanguageIndex(0),
      m_MaxHistorySamples(1000), m_FrameTime(0.0f), m_RefreshRate(0.1f),
      m_TimeSinceLastUpdate(0.0f), m_ShowExecutionChart(true),
      m_ShowMemoryChart(true), m_ShowCallGraph(false),
      m_AutoRefresh(true), m_SortByTime(true)
{
}

ScriptingProfilerUI::~ScriptingProfilerUI()
{
    Shutdown();
}

void ScriptingProfilerUI::Init()
{
    // Initialize available languages
    m_AvailableLanguages = {
        "LuaJIT",
        "AngelScript",
        "WASM",
        "Wren",
        "GDScript"
    };

    // Initialize language stats
    for (const auto& lang : m_AvailableLanguages) {
        LanguageStats stats;
        stats.language = lang;
        stats.profilingEnabled = true;
        stats.lastSampleTime = std::chrono::high_resolution_clock::now();
        m_LanguageStats[lang] = stats;
    }

    // Initialize LuaJIT profiling
    try {
        auto& luaJit = LuaJitScriptSystem::GetInstance();
        luaJit.SetProfilingEnabled(true);
        m_LanguageStats["LuaJIT"].isAvailable = true;
    } catch (...) {
        m_LanguageStats["LuaJIT"].isAvailable = false;
    }
}

void ScriptingProfilerUI::Shutdown()
{
    m_LanguageStats.clear();
    m_AvailableLanguages.clear();
}

void ScriptingProfilerUI::Update(float deltaTime)
{
    if (m_Paused || !m_AutoRefresh) {
        return;
    }

    m_FrameTime = deltaTime;
    m_TimeSinceLastUpdate += deltaTime;

    if (m_TimeSinceLastUpdate >= m_RefreshRate) {
        CollectFrameData();
        m_TimeSinceLastUpdate = 0.0f;
    }
}

void ScriptingProfilerUI::Render()
{
    if (!m_IsOpen) {
        return;
    }

    RenderMainWindow();
}

void ScriptingProfilerUI::SetLanguageProfilingEnabled(const std::string& language, bool enabled)
{
    if (m_LanguageStats.find(language) != m_LanguageStats.end()) {
        m_LanguageStats[language].profilingEnabled = enabled;

        // Enable profiling on actual system
        if (language == "LuaJIT") {
            try {
                auto& luaJit = LuaJitScriptSystem::GetInstance();
                luaJit.SetProfilingEnabled(enabled);
            } catch (...) {}
        }
    }
}

bool ScriptingProfilerUI::IsLanguageProfilingEnabled(const std::string& language) const
{
    auto it = m_LanguageStats.find(language);
    if (it != m_LanguageStats.end()) {
        return it->second.profilingEnabled;
    }
    return false;
}

void ScriptingProfilerUI::ClearData()
{
    for (auto& pair : m_LanguageStats) {
        pair.second.functionStats.clear();
        pair.second.executionTimeHistory.clear();
        pair.second.memoryHistory.clear();
        pair.second.totalCallCount = 0;
        pair.second.totalExecutionTime = 0.0;
    }
    m_ExpandedFunctions.clear();
}

void ScriptingProfilerUI::ExportToJSON(const std::string& filepath)
{
    json output = json::object();
    output["exported_at"] = std::chrono::system_clock::now().time_since_epoch().count();
    output["frame_time_ms"] = m_FrameTime;

    json languages = json::array();
    for (const auto& pair : m_LanguageStats) {
        const auto& stats = pair.second;
        json lang_data = json::object({
            {"language", stats.language},
            {"total_execution_time_ms", stats.totalExecutionTime},
            {"total_calls", stats.totalCallCount},
            {"memory_usage_bytes", stats.memoryUsage},
            {"profiling_enabled", stats.profilingEnabled}
        });

        json functions = json::array();
        for (const auto& func : stats.functionStats) {
            functions.push_back(json::object({
                {"name", func.name},
                {"call_count", func.callCount},
                {"total_time_ms", func.totalTime},
                {"min_time_ms", func.minTime},
                {"max_time_ms", func.maxTime},
                {"avg_time_ms", func.avgTime},
                {"samples", func.samples}
            }));
        }
        lang_data["functions"] = functions;

        languages.push_back(lang_data);
    }
    output["languages"] = languages;

    try {
        std::ofstream file(filepath);
        if (file.is_open()) {
            file << output.dump(2);
            file.close();
        }
    } catch (const std::exception& e) {
        // Handle error silently
    }
}

void ScriptingProfilerUI::ExportToCSV(const std::string& filepath)
{
    try {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return;
        }

        // Write header
        file << "Language,Function,CallCount,TotalTime_ms,MinTime_ms,MaxTime_ms,AvgTime_ms\n";

        // Write data rows
        for (const auto& pair : m_LanguageStats) {
            const auto& stats = pair.second;
            for (const auto& func : stats.functionStats) {
                file << stats.language << ","
                     << func.name << ","
                     << func.callCount << ","
                     << std::fixed << std::setprecision(4) << func.totalTime << ","
                     << func.minTime << ","
                     << func.maxTime << ","
                     << func.avgTime << "\n";
            }
        }

        file.close();
    } catch (const std::exception& e) {
        // Handle error silently
    }
}

void ScriptingProfilerUI::CollectFrameData()
{
    if (m_Paused) {
        return;
    }

    // Collect metrics from LuaJIT
    SampleLanguageMetrics("LuaJIT");
    SampleLanguageMetrics("AngelScript");
    SampleLanguageMetrics("WASM");
}

void ScriptingProfilerUI::SampleLanguageMetrics(const std::string& language)
{
    if (m_LanguageStats.find(language) == m_LanguageStats.end()) {
        return;
    }

    auto& stats = m_LanguageStats[language];

    if (language == "LuaJIT") {
        try {
            auto& luaJit = LuaJitScriptSystem::GetInstance();
            auto profStats = luaJit.GetProfilingStats();
            
            stats.totalExecutionTime = profStats.totalExecutionTime / 1000.0;  // Convert to ms
            stats.totalCallCount = profStats.callCount;

            // Record history
            if (stats.executionTimeHistory.size() >= m_MaxHistorySamples) {
                stats.executionTimeHistory.erase(stats.executionTimeHistory.begin());
            }
            stats.executionTimeHistory.push_back(stats.totalExecutionTime);
        } catch (...) {
            stats.isAvailable = false;
        }
    }
}

void ScriptingProfilerUI::RenderMainWindow()
{
    ImGui::SetNextWindowSize(ImVec2(1200, 700), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Scripting Profiler", &m_IsOpen, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    RenderToolbar();
    ImGui::Separator();

    // Main content with tabs
    if (ImGui::BeginTabBar("ProfilerTabs")) {
        if (ImGui::BeginTabItem("Overview")) {
            RenderOverviewPanel();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Language Details")) {
            RenderLanguageDetailsPanel();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Performance Charts")) {
            RenderPerformanceChartsPanel();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Memory Stats")) {
            RenderMemoryStatsPanel();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Call Graph")) {
            RenderCallGraphPanel();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Settings")) {
            RenderSettingsPanel();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

void ScriptingProfilerUI::RenderToolbar()
{
    ImGui::BeginGroup();
    
    // Play/Pause button
    if (m_Paused) {
        if (ImGui::Button("Resume", ImVec2(80, 0))) {
            m_Paused = false;
        }
    } else {
        if (ImGui::Button("Pause", ImVec2(80, 0))) {
            m_Paused = true;
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Clear Data", ImVec2(100, 0))) {
        ClearData();
    }

    ImGui::SameLine();
    if (ImGui::Button("Export JSON", ImVec2(100, 0))) {
        ExportToJSON("profiler_data.json");
    }

    ImGui::SameLine();
    if (ImGui::Button("Export CSV", ImVec2(100, 0))) {
        ExportToCSV("profiler_data.csv");
    }

    ImGui::SameLine();
    ImGui::Checkbox("Auto Refresh", &m_AutoRefresh);

    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    ImGui::SliderFloat("Refresh Rate (s)", &m_RefreshRate, 0.01f, 1.0f, "%.3f");

    ImGui::EndGroup();
}

void ScriptingProfilerUI::RenderOverviewPanel()
{
    ImGui::Text("Frame Time: %.2f ms (%.1f FPS)", m_FrameTime * 1000.0f, 1.0f / m_FrameTime);
    ImGui::Separator();

    // Language overview table
    if (ImGui::BeginTable("LanguageOverview", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Language");
        ImGui::TableSetupColumn("Status");
        ImGui::TableSetupColumn("Exec Time (ms)");
        ImGui::TableSetupColumn("Calls");
        ImGui::TableSetupColumn("Memory (MB)");
        ImGui::TableHeadersRow();

        for (const auto& langName : m_AvailableLanguages) {
            auto it = m_LanguageStats.find(langName);
            if (it == m_LanguageStats.end()) {
                continue;
            }

            const auto& stats = it->second;

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "%s", langName.c_str());

            ImGui::TableSetColumnIndex(1);
            if (stats.isAvailable) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Available");
            } else {
                ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "Unavailable");
            }

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.3f", stats.totalExecutionTime);

            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%llu", stats.totalCallCount);

            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.2f", stats.memoryUsage / (1024.0f * 1024.0f));
        }

        ImGui::EndTable();
    }
}

void ScriptingProfilerUI::RenderLanguageDetailsPanel()
{
    RenderLanguageTabs();
}

void ScriptingProfilerUI::RenderLanguageTabs()
{
    if (ImGui::BeginTabBar("LanguageTabs")) {
        if (ImGui::BeginTabItem("LuaJIT")) {
            RenderLuaJitStats();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("AngelScript")) {
            RenderAngelScriptStats();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("WASM")) {
            RenderWasmStats();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Wren")) {
            RenderWrenStats();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("GDScript")) {
            RenderGDScriptStats();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void ScriptingProfilerUI::RenderLuaJitStats()
{
    auto it = m_LanguageStats.find("LuaJIT");
    if (it == m_LanguageStats.end() || !it->second.isAvailable) {
        ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "LuaJIT not available");
        return;
    }

    auto& stats = it->second;

    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "LuaJIT Performance Metrics");
    ImGui::Separator();

    // Try to get live stats from LuaJIT
    try {
        auto& luaJit = LuaJitScriptSystem::GetInstance();
        auto profStats = luaJit.GetProfilingStats();

        ImGui::Text("Total Execution Time: %.3f ms", profStats.totalExecutionTime / 1000.0);
        ImGui::Text("Call Count: %llu", profStats.callCount);
        ImGui::Text("Avg Time Per Call: %.2f μs", profStats.avgExecutionTime);

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "JIT Statistics");
        ImGui::Text("Active Traces: %u", profStats.activeTraces);
        ImGui::Text("JIT Compiled Functions: %u", profStats.jitCompiledFunctions);
        ImGui::ProgressBar(profStats.jitCoveragePercent / 100.0f, ImVec2(-1.0f, 0.0f));
        ImGui::SameLine();
        ImGui::Text("JIT Coverage: %.1f%%", profStats.jitCoveragePercent);
    } catch (const std::exception& e) {
        ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "Error reading stats");
    }
}

void ScriptingProfilerUI::RenderAngelScriptStats()
{
    auto it = m_LanguageStats.find("AngelScript");
    if (it == m_LanguageStats.end()) {
        ImGui::Text("AngelScript profiling not available");
        return;
    }

    auto& stats = it->second;

    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "AngelScript Performance Metrics");
    ImGui::Separator();

    ImGui::Text("Total Execution Time: %.3f ms", stats.totalExecutionTime);
    ImGui::Text("Call Count: %llu", stats.totalCallCount);
    ImGui::Text("Memory Usage: %.2f MB", stats.memoryUsage / (1024.0f * 1024.0f));

    ImGui::Separator();
    ImGui::Text("Note: AngelScript profiling requires integration with AngelScriptSystem");
}

void ScriptingProfilerUI::RenderWasmStats()
{
    auto it = m_LanguageStats.find("WASM");
    if (it == m_LanguageStats.end()) {
        ImGui::Text("WASM profiling not available");
        return;
    }

    auto& stats = it->second;

    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "WebAssembly Performance Metrics");
    ImGui::Separator();

    ImGui::Text("Total Execution Time: %.3f ms", stats.totalExecutionTime);
    ImGui::Text("Call Count: %llu", stats.totalCallCount);
    ImGui::Text("Memory Usage: %.2f MB", stats.memoryUsage / (1024.0f * 1024.0f));

    ImGui::Separator();
    ImGui::Text("Note: WASM profiling requires integration with WasmScriptSystem");
}

void ScriptingProfilerUI::RenderWrenStats()
{
    auto it = m_LanguageStats.find("Wren");
    if (it == m_LanguageStats.end()) {
        ImGui::Text("Wren profiling not available");
        return;
    }

    auto& stats = it->second;

    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Wren Performance Metrics");
    ImGui::Separator();

    ImGui::Text("Total Execution Time: %.3f ms", stats.totalExecutionTime);
    ImGui::Text("Call Count: %llu", stats.totalCallCount);
    ImGui::Text("Memory Usage: %.2f MB", stats.memoryUsage / (1024.0f * 1024.0f));
}

void ScriptingProfilerUI::RenderGDScriptStats()
{
    auto it = m_LanguageStats.find("GDScript");
    if (it == m_LanguageStats.end()) {
        ImGui::Text("GDScript profiling not available");
        return;
    }

    auto& stats = it->second;

    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "GDScript Performance Metrics");
    ImGui::Separator();

    ImGui::Text("Total Execution Time: %.3f ms", stats.totalExecutionTime);
    ImGui::Text("Call Count: %llu", stats.totalCallCount);
    ImGui::Text("Memory Usage: %.2f MB", stats.memoryUsage / (1024.0f * 1024.0f));
}

void ScriptingProfilerUI::RenderOtherLanguageStats()
{
    ImGui::Text("Additional language profiling details here");
}

void ScriptingProfilerUI::RenderPerformanceChartsPanel()
{
    ImGui::Checkbox("Show Execution Time Chart", &m_ShowExecutionChart);
    ImGui::SameLine();
    ImGui::Checkbox("Show Memory Chart", &m_ShowMemoryChart);

    if (m_ShowExecutionChart) {
        ImGui::Text("Execution Time History");
        auto it = m_LanguageStats.find("LuaJIT");
        if (it != m_LanguageStats.end() && !it->second.executionTimeHistory.empty()) {
            const auto& history = it->second.executionTimeHistory;
            ImGui::PlotLines(
                "##ExecutionTimeChart",
                history.data(),
                static_cast<int>(history.size()),
                0,
                "Execution Time (ms)",
                0.0f,
                FLT_MAX,
                ImVec2(0, 200)
            );
        } else {
            ImGui::Text("No execution time data available");
        }
    }

    if (m_ShowMemoryChart) {
        ImGui::Text("Memory Usage History");
        auto it = m_LanguageStats.find("LuaJIT");
        if (it != m_LanguageStats.end() && !it->second.memoryHistory.empty()) {
            const auto& history = it->second.memoryHistory;
            ImGui::PlotLines(
                "##MemoryChart",
                history.data(),
                static_cast<int>(history.size()),
                0,
                "Memory (MB)",
                0.0f,
                FLT_MAX,
                ImVec2(0, 200)
            );
        } else {
            ImGui::Text("No memory data available");
        }
    }
}

void ScriptingProfilerUI::RenderMemoryStatsPanel()
{
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Memory Statistics");
    ImGui::Separator();

    if (ImGui::BeginTable("MemoryStats", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Language");
        ImGui::TableSetupColumn("Memory (MB)");
        ImGui::TableSetupColumn("Profiling Enabled");
        ImGui::TableHeadersRow();

        for (const auto& pair : m_LanguageStats) {
            const auto& stats = pair.second;

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", stats.language.c_str());

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.2f", stats.memoryUsage / (1024.0f * 1024.0f));

            ImGui::TableSetColumnIndex(2);
            if (stats.profilingEnabled) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Yes");
            } else {
                ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "No");
            }
        }

        ImGui::EndTable();
    }
}

void ScriptingProfilerUI::RenderCallGraphPanel()
{
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Call Graph Visualization");
    ImGui::Separator();
    ImGui::Text("Call graph data visualization would be rendered here");
    ImGui::Text("Currently displaying aggregated function call data");
}

void ScriptingProfilerUI::RenderSettingsPanel()
{
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Profiler Settings");
    ImGui::Separator();

    ImGui::Checkbox("Auto Refresh##Settings", &m_AutoRefresh);
    ImGui::SliderFloat("Refresh Rate (s)##Settings", &m_RefreshRate, 0.01f, 1.0f, "%.3f");
    ImGui::SliderInt("Max History Samples", reinterpret_cast<int*>(&m_MaxHistorySamples), 100, 10000);

    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Language Profiling");

    for (auto& pair : m_LanguageStats) {
        auto key = pair.first + "##checkbox";
        ImGui::Checkbox(key.c_str(), &pair.second.profilingEnabled);
    }

    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Export Options");

    static char filepath[256] = "profiler_data";
    ImGui::InputText("Export Filepath", filepath, sizeof(filepath));

    if (ImGui::Button("Export JSON##Settings", ImVec2(120, 0))) {
        std::string fullpath = std::string(filepath) + ".json";
        ExportToJSON(fullpath);
    }

    ImGui::SameLine();
    if (ImGui::Button("Export CSV##Settings", ImVec2(120, 0))) {
        std::string fullpath = std::string(filepath) + ".csv";
        ExportToCSV(fullpath);
    }
}

void ScriptingProfilerUI::RenderFunctionCallList(const std::string& language)
{
    auto it = m_LanguageStats.find(language);
    if (it == m_LanguageStats.end()) {
        return;
    }

    auto& stats = it->second;

    if (ImGui::BeginTable("FunctionCalls", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Function Name");
        ImGui::TableSetupColumn("Calls");
        ImGui::TableSetupColumn("Total (ms)");
        ImGui::TableSetupColumn("Avg (ms)");
        ImGui::TableSetupColumn("Max (ms)");
        ImGui::TableHeadersRow();

        for (const auto& func : stats.functionStats) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", func.name.c_str());

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%llu", func.callCount);

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.3f", func.totalTime);

            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.3f", func.avgTime);

            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.3f", func.maxTime);
        }

        ImGui::EndTable();
    }
}

void ScriptingProfilerUI::RenderCallGraph(const std::string& language)
{
    ImGui::Text("Call graph for: %s", language.c_str());
}

void ScriptingProfilerUI::UpdateHistoryData(const std::string& language)
{
    auto it = m_LanguageStats.find(language);
    if (it == m_LanguageStats.end()) {
        return;
    }

    auto& stats = it->second;

    if (stats.executionTimeHistory.size() >= m_MaxHistorySamples) {
        stats.executionTimeHistory.erase(stats.executionTimeHistory.begin());
    }
    stats.executionTimeHistory.push_back(stats.totalExecutionTime);

    if (stats.memoryHistory.size() >= m_MaxHistorySamples) {
        stats.memoryHistory.erase(stats.memoryHistory.begin());
    }
    stats.memoryHistory.push_back(stats.memoryUsage / (1024.0f * 1024.0f));
}
