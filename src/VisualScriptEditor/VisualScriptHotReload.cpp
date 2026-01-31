#include "../include/VisualScriptEditor/VisualScriptHotReload.h"
#include "../include/VisualScriptEditor/CodeGenerator.h"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

VisualScriptHotReload::VisualScriptHotReload() : m_IsRunning(false) {
}

VisualScriptHotReload::~VisualScriptHotReload() {
    Shutdown();
}

void VisualScriptHotReload::Initialize(const std::string& scriptOutputDirectory) {
    m_OutputDirectory = scriptOutputDirectory;
    m_IsRunning = true;
    
    // Create output directory if it doesn't exist
    if (!fs::exists(m_OutputDirectory)) {
        fs::create_directories(m_OutputDirectory);
    }

    m_FileWatcher = std::make_unique<FileWatcher>();
}

void VisualScriptHotReload::Shutdown() {
    m_IsRunning = false;
    if (m_WatchThread.joinable()) {
        m_WatchThread.join();
    }
}

bool VisualScriptHotReload::CompileAndLoad(const std::shared_ptr<VisualGraph>& graph) {
    if (!graph) {
        m_LastCompilationError = "Graph is null";
        return false;
    }

    LoadedScript script;
    script.name = graph->GetName();
    script.headerPath = GetGeneratedHeaderPath(script.name);
    script.sourcePath = GetGeneratedSourcePath(script.name);

    auto start = std::chrono::high_resolution_clock::now();

    // Generate code
    auto generator = CodeGeneratorFactory::CreateGenerator(graph->GetType());
    if (!generator) {
        m_LastCompilationError = "Failed to create code generator for graph type";
        if (m_OnCompileComplete) {
            m_OnCompileComplete(false, m_LastCompilationError);
        }
        return false;
    }

    // Generate header file
    try {
        std::string headerContent = generator->GenerateHeader(graph);
        std::ofstream headerFile(script.headerPath);
        headerFile << headerContent;
        headerFile.close();
    } catch (const std::exception& e) {
        m_LastCompilationError = "Failed to write header file: " + std::string(e.what());
        if (m_OnCompileComplete) {
            m_OnCompileComplete(false, m_LastCompilationError);
        }
        return false;
    }

    // Generate source file
    try {
        script.generatedCode = generator->GenerateCode(graph);
        std::ofstream sourceFile(script.sourcePath);
        sourceFile << script.generatedCode;
        sourceFile.close();
    } catch (const std::exception& e) {
        m_LastCompilationError = "Failed to write source file: " + std::string(e.what());
        if (m_OnCompileComplete) {
            m_OnCompileComplete(false, m_LastCompilationError);
        }
        return false;
    }

    // Compile via CMake (in a real implementation)
    std::string compileError = CompileViaCMake(script.name);
    if (!compileError.empty()) {
        m_LastCompilationError = compileError;
        ++m_Stats.compilationErrorCount;
        if (m_OnCompileComplete) {
            m_OnCompileComplete(false, m_LastCompilationError);
        }
        return false;
    }

    // Load the compiled script
    if (!InternalLoad(script)) {
        m_LastCompilationError = "Failed to load compiled script";
        if (m_OnCompileComplete) {
            m_OnCompileComplete(false, m_LastCompilationError);
        }
        return false;
    }

    m_LoadedScripts[script.name] = script;
    m_LastCompilationError = "";
    ++m_Stats.totalScriptsLoaded;

    auto end = std::chrono::high_resolution_clock::now();
    m_Stats.lastCompileTime = std::chrono::duration<float>(end - start).count();

    if (m_OnCompileComplete) {
        m_OnCompileComplete(true, "Compilation successful");
    }

    return true;
}

bool VisualScriptHotReload::ReloadScript(const std::string& scriptName) {
    auto it = m_LoadedScripts.find(scriptName);
    if (it == m_LoadedScripts.end()) {
        m_LastCompilationError = "Script not found: " + scriptName;
        return false;
    }

    return InternalLoad(it->second);
}

void VisualScriptHotReload::WatchGraphFile(const std::string& filePath,
                                          std::function<void(const std::shared_ptr<VisualGraph>&)> onReload) {
    m_GraphFileWatchers[filePath] = onReload;
    if (m_FileWatcher) {
        m_FileWatcher->Watch(filePath, [this, filePath](const std::string&) {
            // File changed, reload it
            if (m_AutoReload) {
                // In a real implementation, deserialize the graph and call the callback
            }
        });
    }
}

void VisualScriptHotReload::UnwatchGraphFile(const std::string& filePath) {
    m_GraphFileWatchers.erase(filePath);
}

std::string VisualScriptHotReload::GetGeneratedHeaderPath(const std::string& scriptName) const {
    return m_OutputDirectory + "/" + scriptName + ".h";
}

std::string VisualScriptHotReload::GetGeneratedSourcePath(const std::string& scriptName) const {
    return m_OutputDirectory + "/" + scriptName + ".cpp";
}

void VisualScriptHotReload::RegisterScriptFunction(const std::string& scriptName,
                                                   const std::string& functionName,
                                                   ScriptFunction func) {
    m_ScriptFunctions[scriptName + "::" + functionName] = func;
}

bool VisualScriptHotReload::ExecuteScriptFunction(const std::string& scriptName,
                                                  const std::string& functionName) {
    auto key = scriptName + "::" + functionName;
    auto it = m_ScriptFunctions.find(key);
    if (it != m_ScriptFunctions.end()) {
        try {
            it->second();
            return true;
        } catch (const std::exception& e) {
            m_LastCompilationError = "Error executing function: " + std::string(e.what());
            return false;
        }
    }
    return false;
}

bool VisualScriptHotReload::InternalCompile(const std::shared_ptr<VisualGraph>& graph, LoadedScript& outScript) {
    // Implementation depends on build system
    // For now, this is handled in CompileAndLoad
    return true;
}

bool VisualScriptHotReload::InternalLoad(const LoadedScript& script) {
    // In a real implementation with hot-reload, this would:
    // 1. Load the compiled shared library/DLL
    // 2. Extract function pointers
    // 3. Replace old function pointers with new ones
    // For now, we just mark it as loaded
    
    const_cast<LoadedScript&>(script).isLoaded = true;
    return true;
}

void VisualScriptHotReload::FileWatchThread() {
    while (m_IsRunning) {
        // Process file watch events
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

std::string VisualScriptHotReload::CompileViaCMake(const std::string& scriptName) {
    // In a real implementation, this would:
    // 1. Create a temporary CMakeLists.txt for the script
    // 2. Run cmake to build it
    // 3. Return any compilation errors
    
    // For now, we'll just return empty string (success)
    ++m_Stats.totalGraphsCompiled;
    return "";
}
