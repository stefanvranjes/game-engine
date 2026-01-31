# Scripting Profiler UI - Architecture & Design

## System Architecture

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       Game Engine                                │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Application Loop                         │ │
│  │                                                              │ │
│  │  Init() → Update() → Render() → RenderEditorUI() → ...    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                      │
│    ┌───────────────────────┼──────────────────────┐              │
│    │                       │                      │              │
│    ▼                       ▼                      ▼              │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  PhysicsSys │  │ ScriptingProfUI  │  │  ScriptDebugger  │   │
│  └─────────────┘  │                  │  │      UI          │   │
│                   │ • Init()          │  └──────────────────┘   │
│                   │ • Update()        │                          │
│                   │ • Render()        │                          │
│                   │ • Export()        │                          │
│                   └────────┬──────────┘                          │
│                            │                                      │
│    ┌───────────────────────┼──────────────────────┐              │
│    │                       │                      │              │
│    ▼                       ▼                      ▼              │
│ ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│ │  LuaJIT     │  │  AngelScript     │  │    WASM          │    │
│ │  Profiler   │  │    Profiler      │  │    Profiler      │    │
│ └─────────────┘  └──────────────────┘  └──────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Script Execution (Game Loop)
    │
    ▼
┌─────────────────────────────────┐
│  SampleLanguageMetrics()        │  ← Called from Update()
│  CollectFrameData()             │    every 100ms (configurable)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  LanguageStats (per-language)   │
│  - ExecutionTime                │
│  - MemoryUsage                  │
│  - CallCount                    │
│  - FunctionStats[]              │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  History Buffers                │  ← Maintained for charts
│  - executionTimeHistory[]       │    (max 1000 samples)
│  - memoryHistory[]              │
└──────────────┬──────────────────┘
               │
          ┌────┴────┐
          │          │
          ▼          ▼
      ┌─────────┐ ┌──────────┐
      │ ImGui   │ │ Export   │
      │Render() │ │ JSON/CSV │
      └─────────┘ └──────────┘
```

### Class Structure

```
ScriptingProfilerUI (Main Class)
│
├─ Public Methods
│  ├─ Init() / Shutdown()
│  ├─ Update(deltaTime)
│  ├─ Render()
│  ├─ Show() / Hide() / Toggle()
│  ├─ SetLanguageProfilingEnabled()
│  ├─ ClearData()
│  ├─ ExportToJSON() / ExportToCSV()
│  ├─ SetMaxHistorySamples()
│  └─ SetPaused()
│
├─ Private Data
│  ├─ m_LanguageStats (map)         ← Per-language metrics
│  ├─ m_AvailableLanguages (vector) ← List of supported languages
│  ├─ m_IsOpen (bool)               ← Window visibility
│  ├─ m_Paused (bool)               ← Data collection state
│  └─ m_TimeSinceLastUpdate (float) ← Refresh timer
│
├─ Private UI Methods
│  ├─ RenderMainWindow()
│  ├─ RenderToolbar()
│  ├─ RenderOverviewPanel()
│  ├─ RenderLanguageDetailsPanel()
│  ├─ RenderPerformanceChartsPanel()
│  ├─ RenderMemoryStatsPanel()
│  ├─ RenderCallGraphPanel()
│  ├─ RenderSettingsPanel()
│  └─ RenderLanguageTabs()
│
├─ Private Language Methods
│  ├─ RenderLuaJitStats()
│  ├─ RenderAngelScriptStats()
│  ├─ RenderWasmStats()
│  ├─ RenderWrenStats()
│  └─ RenderGDScriptStats()
│
└─ Private Data Collection
   ├─ CollectFrameData()
   ├─ SampleLanguageMetrics()
   └─ UpdateHistoryData()

LanguageStats (Data Structure)
│
├─ language (string)
├─ totalExecutionTime (double)
├─ totalCallCount (uint64_t)
├─ memoryUsage (size_t)
├─ isAvailable (bool)
├─ profilingEnabled (bool)
├─ functionStats (vector<FunctionStats>)
├─ executionTimeHistory (vector<double>)
├─ memoryHistory (vector<double>)
└─ lastSampleTime (chrono::time_point)

FunctionStats (Data Structure)
│
├─ name (string)
├─ callCount (uint64_t)
├─ totalTime (double)
├─ minTime (double)
├─ maxTime (double)
├─ avgTime (double)
└─ samples (size_t)
```

## UI Component Hierarchy

```
ScriptingProfilerUI Window
│
├─ Toolbar
│  ├─ [Pause/Resume Button]
│  ├─ [Clear Data Button]
│  ├─ [Export JSON Button]
│  ├─ [Export CSV Button]
│  ├─ [Auto Refresh Checkbox]
│  └─ [Refresh Rate Slider]
│
├─ Separator
│
└─ Tab Bar
   │
   ├─ Tab: Overview
   │  └─ Language Summary Table
   │     ├─ Language (Column)
   │     ├─ Status (Column)
   │     ├─ Exec Time (Column)
   │     ├─ Calls (Column)
   │     └─ Memory (Column)
   │
   ├─ Tab: Language Details
   │  └─ Language Tab Bar
   │     ├─ Sub-Tab: LuaJIT
   │     │  ├─ Execution Time Display
   │     │  ├─ Call Count Display
   │     │  ├─ JIT Coverage Progress Bar
   │     │  ├─ Active Traces Display
   │     │  └─ Compiled Functions Display
   │     │
   │     ├─ Sub-Tab: AngelScript
   │     │  ├─ Execution Time Display
   │     │  ├─ Call Count Display
   │     │  └─ Memory Usage Display
   │     │
   │     ├─ Sub-Tab: WASM
   │     │  └─ Module Metrics Display
   │     │
   │     ├─ Sub-Tab: Wren
   │     │  └─ VM Statistics Display
   │     │
   │     └─ Sub-Tab: GDScript
   │        └─ Script Metrics Display
   │
   ├─ Tab: Performance Charts
   │  ├─ [Checkbox] Show Execution Time Chart
   │  ├─ Execution Time Graph (ImGui::PlotLines)
   │  ├─ [Checkbox] Show Memory Chart
   │  └─ Memory Usage Graph (ImGui::PlotLines)
   │
   ├─ Tab: Memory Stats
   │  └─ Memory Usage Table
   │     ├─ Language (Column)
   │     ├─ Memory MB (Column)
   │     └─ Profiling Enabled (Column)
   │
   ├─ Tab: Call Graph
   │  └─ Call Graph Visualization Placeholder
   │
   └─ Tab: Settings
      ├─ [Checkbox] Auto Refresh
      ├─ Refresh Rate Slider
      ├─ Max History Samples Slider
      ├─ Language Profiling Checkboxes
      │  ├─ [Checkbox] LuaJIT
      │  ├─ [Checkbox] AngelScript
      │  ├─ [Checkbox] WASM
      │  ├─ [Checkbox] Wren
      │  └─ [Checkbox] GDScript
      ├─ Export Filepath Input
      ├─ [Button] Export JSON
      └─ [Button] Export CSV
```

## Data Structures

### LanguageStats
```cpp
struct LanguageStats {
    std::string language;                              // Language name
    double totalExecutionTime = 0.0;                   // Total ms
    uint64_t totalCallCount = 0;                       // Total calls
    size_t memoryUsage = 0;                            // Total bytes
    bool isAvailable = false;                          // Available?
    bool profilingEnabled = false;                     // Enabled?
    std::vector<FunctionStats> functionStats;          // Per-function data
    std::vector<double> executionTimeHistory;          // Historical
    std::vector<double> memoryHistory;                 // Historical
    std::chrono::high_resolution_clock::time_point 
        lastSampleTime;                                // Last update time
};
```

### FunctionStats
```cpp
struct FunctionStats {
    std::string name;              // Function name
    uint64_t callCount = 0;        // Call count
    double totalTime = 0.0;        // Total ms
    double minTime = 0.0;          // Min ms
    double maxTime = 0.0;          // Max ms
    double avgTime = 0.0;          // Average ms
    size_t samples = 0;            // Number of samples
};
```

## Integration Points

### Application.h
```cpp
#include "ScriptingProfilerUI.h"

class Application {
    std::unique_ptr<ScriptingProfilerUI> m_ScriptingProfilerUI;
};
```

### Application::Init()
```cpp
m_ScriptingProfilerUI = std::make_unique<ScriptingProfilerUI>();
m_ScriptingProfilerUI->Init();
```

### Application::RenderEditorUI()
```cpp
// In Tools menu
if (ImGui::MenuItem("Scripting Profiler", "Ctrl+Shift+P")) {
    m_ScriptingProfilerUI->Toggle();
}

// Rendering
if (m_ScriptingProfilerUI) {
    m_ScriptingProfilerUI->Update(m_LastFrameTime);
    m_ScriptingProfilerUI->Render();
}
```

## Data Flow Sequence

### On Application Update
```
1. Application::Update(deltaTime)
2. m_ScriptingProfilerUI->Update(deltaTime)
3. m_TimeSinceLastUpdate += deltaTime
4. IF m_TimeSinceLastUpdate >= m_RefreshRate THEN
5.   CollectFrameData()
6.   FOR each language in m_AvailableLanguages DO
7.     SampleLanguageMetrics(language)
8.   FOR each language DO
9.     UpdateHistoryData(language)
10. m_TimeSinceLastUpdate = 0
```

### On User Interaction
```
User Opens Profiler (Ctrl+Shift+P)
  │
  ▼
Application::RenderEditorUI()
  │
  ├─ ImGui::Begin("Scripting Profiler")
  │
  ├─ RenderMainWindow()
  │  ├─ RenderToolbar()
  │  ├─ Handle Toolbar Button Clicks
  │  └─ RenderTabContent()
  │
  └─ ImGui::End()
```

### On Export
```
User Clicks "Export JSON" Button
  │
  ▼
ExportToJSON(filepath)
  │
  ├─ Create JSON object
  ├─ FOR each LanguageStats DO
  │  ├─ Add language data
  │  └─ Add function statistics
  ├─ Write to file
  └─ Close file
```

## Performance Profile

### Memory Usage
```
Base Overhead:          ~5-10 MB
Per Language (1000):    ~1-2 MB
Total (5 languages):    ~10-20 MB
```

### CPU Usage
```
Update (per frame):     < 1 ms
Render (per frame):     < 2 ms
Data Collection:        ~100-200 μs (periodic)
Export:                 < 500 ms (on-demand)
```

### Network/IO
```
JSON Export Size:       ~100-500 KB
CSV Export Size:        ~50-300 KB
File Writing:           < 100 ms
```

## Threading Model

```
Main Thread
  │
  ├─ Update()
  │  └─ Collect metrics from script systems (thread-safe reads)
  │
  └─ Render()
     └─ ImGui rendering (single-threaded)
```

**Note**: All script systems must provide thread-safe read access to profiling data.

## Extension Points

### Adding a New Language

1. **Add to supported list**:
```cpp
m_AvailableLanguages.push_back("NewLang");
```

2. **Create tab rendering function**:
```cpp
void RenderNewLangStats() {
    // Render language-specific metrics
}
```

3. **Add to language tabs**:
```cpp
if (ImGui::BeginTabItem("NewLang")) {
    RenderNewLangStats();
    ImGui::EndTabItem();
}
```

4. **Integrate data collection**:
```cpp
void SampleLanguageMetrics(const std::string& language) {
    if (language == "NewLang") {
        // Get metrics from system
        // Update LanguageStats
    }
}
```

### Custom Export Formats

```cpp
void ExportToCustomFormat(const std::string& filepath) {
    // Iterate through m_LanguageStats
    // Format and write data
}
```

## Color Scheme

| Element | Color | RGBA |
|---------|-------|------|
| Execution Time Chart | Green | (0.2, 0.8, 0.2, 0.8) |
| Memory Chart | Blue | (0.2, 0.6, 1.0, 0.8) |
| Call Count Chart | Yellow | (1.0, 0.8, 0.2, 0.8) |
| Header Background | Dark Gray | (0.3, 0.3, 0.3, 0.7) |
| Row Background | Darker Gray | (0.15, 0.15, 0.15, 0.5) |

## Refresh Rate Configuration

| Setting | Min | Default | Max | Unit |
|---------|-----|---------|-----|------|
| Refresh Rate | 0.01 | 0.1 | 1.0 | seconds |
| Max History | 100 | 1000 | 10000 | samples |

## Summary

The Scripting Profiler UI is designed as a modular, extensible system that:

1. **Integrates seamlessly** with the Application loop
2. **Collects data** from multiple script systems
3. **Visualizes** performance metrics in real-time
4. **Exports** data for external analysis
5. **Extends easily** for new languages/metrics

Architecture emphasizes:
- Separation of concerns (UI, data, collection)
- Minimal overhead (configurable refresh rate)
- Extensibility (pluggable languages)
- User-friendliness (intuitive UI)

---

**Architecture Version**: 1.0  
**Last Updated**: 2026-01-31
