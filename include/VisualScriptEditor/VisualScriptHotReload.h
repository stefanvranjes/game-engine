#pragma once

#include "VisualGraph.h"
#include "../FileWatcher.h"
#include <memory>
#include <string>
#include <functional>
#include <map>
#include <thread>

class VisualScriptHotReload {
public:
    VisualScriptHotReload();
    ~VisualScriptHotReload();

    // Initialize hot-reload system
    void Initialize(const std::string& scriptOutputDirectory);
    void Shutdown();

    // Compile visual graph to C++ code and load it
    bool CompileAndLoad(const std::shared_ptr<VisualGraph>& graph);

    // Reload already compiled script
    bool ReloadScript(const std::string& scriptName);

    // Watch a graph file for changes
    void WatchGraphFile(const std::string& filePath, 
                       std::function<void(const std::shared_ptr<VisualGraph>&)> onReload);
    void UnwatchGraphFile(const std::string& filePath);

    // Get generated header file for a graph
    std::string GetGeneratedHeaderPath(const std::string& scriptName) const;
    std::string GetGeneratedSourcePath(const std::string& scriptName) const;

    // Script execution
    using ScriptFunction = std::function<void()>;
    void RegisterScriptFunction(const std::string& scriptName, const std::string& functionName, ScriptFunction func);
    bool ExecuteScriptFunction(const std::string& scriptName, const std::string& functionName);

    // Get compilation errors
    const std::string& GetLastCompilationError() const { return m_LastCompilationError; }
    bool HasCompilationError() const { return !m_LastCompilationError.empty(); }

    // Settings
    void SetAutoCompile(bool enabled) { m_AutoCompile = enabled; }
    bool GetAutoCompile() const { return m_AutoCompile; }

    void SetAutoReload(bool enabled) { m_AutoReload = enabled; }
    bool GetAutoReload() const { return m_AutoReload; }

    // Callbacks
    using CompileCallback = std::function<void(bool success, const std::string& message)>;
    void SetOnCompileComplete(CompileCallback callback) { m_OnCompileComplete = callback; }

    using ReloadCallback = std::function<void(bool success)>;
    void SetOnReloadComplete(ReloadCallback callback) { m_OnReloadComplete = callback; }

    // Get statistics
    struct CompilationStats {
        size_t totalGraphsCompiled = 0;
        size_t totalScriptsLoaded = 0;
        float lastCompileTime = 0.0f;
        int compilationErrorCount = 0;
    };
    const CompilationStats& GetStats() const { return m_Stats; }
    void ResetStats() { m_Stats = {}; }

private:
    struct LoadedScript {
        std::string name;
        std::string headerPath;
        std::string sourcePath;
        std::string generatedCode;
        bool isLoaded = false;
    };

    bool InternalCompile(const std::shared_ptr<VisualGraph>& graph, LoadedScript& outScript);
    bool InternalLoad(const LoadedScript& script);
    void FileWatchThread();
    std::string CompileViaCMake(const std::string& scriptName);

    std::string m_OutputDirectory;
    std::map<std::string, LoadedScript> m_LoadedScripts;
    std::map<std::string, std::function<void()>> m_ScriptFunctions;

    std::unique_ptr<FileWatcher> m_FileWatcher;
    std::map<std::string, std::function<void(const std::shared_ptr<VisualGraph>&)>> m_GraphFileWatchers;

    std::string m_LastCompilationError;
    CompilationStats m_Stats;

    bool m_AutoCompile = true;
    bool m_AutoReload = true;

    CompileCallback m_OnCompileComplete;
    ReloadCallback m_OnReloadComplete;

    std::thread m_WatchThread;
    bool m_IsRunning = false;
};
