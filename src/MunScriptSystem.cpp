#include "MunScriptSystem.h"
#include "FileWatcher.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
    #include <windows.h>
    #define LOAD_LIBRARY(path) LoadLibraryA(path)
    #define GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)handle, name)
    #define UNLOAD_LIBRARY(handle) FreeLibrary((HMODULE)handle)
    #define LIBRARY_EXTENSION ".dll"
#else
    #include <dlfcn.h>
    #define LOAD_LIBRARY(path) dlopen(path, RTLD_LAZY)
    #define GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
    #define UNLOAD_LIBRARY(handle) dlclose(handle)
    #ifdef __APPLE__
        #define LIBRARY_EXTENSION ".dylib"
    #else
        #define LIBRARY_EXTENSION ".so"
    #endif
#endif

/**
 * Initialize the Mun script system
 */
void MunScriptSystem::Init() {
    if (m_initialized) return;

    std::cout << "[MunScriptSystem] Initializing..." << std::endl;

    if (!IsMunCompilerAvailable()) {
        SetError("Mun compiler not found. Please install Mun from: https://mun-lang.org/");
        std::cerr << "[MunScriptSystem] ERROR: " << m_lastError << std::endl;
        return;
    }

    std::string version = GetMunCompilerVersion();
    std::cout << "[MunScriptSystem] Mun compiler version: " << version << std::endl;

    m_initialized = true;
    m_lastUpdateTime = std::chrono::high_resolution_clock::now();
    m_stats = CompilationStats();

    std::cout << "[MunScriptSystem] Initialization complete" << std::endl;
}

/**
 * Shutdown and cleanup all loaded scripts
 */
void MunScriptSystem::Shutdown() {
    if (!m_initialized) return;

    std::cout << "[MunScriptSystem] Shutting down..." << std::endl;

    // Unload all loaded scripts
    for (auto& [name, script] : m_loadedScripts) {
        UnloadLibrary(name);
    }

    m_loadedScripts.clear();
    m_functionCache.clear();
    m_watchedFiles.clear();
    m_initialized = false;

    std::cout << "[MunScriptSystem] Shutdown complete" << std::endl;
}

/**
 * Update - check for file changes and reload if needed
 */
void MunScriptSystem::Update(float deltaTime) {
    if (!m_initialized) return;

    // Check for file changes every 100ms
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastUpdateTime);

    if (elapsed.count() >= 100) {
        CheckForChanges();
        m_lastUpdateTime = now;
    }
}

/**
 * Load and compile a Mun script
 */
bool MunScriptSystem::RunScript(const std::string& filepath) {
    return LoadScript(filepath);
}

/**
 * Load a Mun script with compilation
 */
bool MunScriptSystem::LoadScript(const std::string& filepath, const CompilationOptions& options) {
    if (!m_initialized) {
        SetError("MunScriptSystem not initialized");
        return false;
    }

    std::cout << "[MunScriptSystem] Loading script: " << filepath << std::endl;

    // Extract script name from filepath
    std::filesystem::path scriptPath(filepath);
    std::string scriptName = scriptPath.stem().string();

    // Check if already loaded
    if (m_loadedScripts.find(scriptName) != m_loadedScripts.end()) {
        std::cout << "[MunScriptSystem] Script already loaded: " << scriptName << std::endl;
        return true;
    }

    // Compile the script
    auto start = std::chrono::high_resolution_clock::now();
    std::string compiledLib = CompileMunSource(filepath);
    auto end = std::chrono::high_resolution_clock::now();

    double compileTime = std::chrono::duration<double>(end - start).count();
    m_stats.totalCompiles++;
    m_stats.totalCompileTime += compileTime;
    m_stats.lastCompileTime = compileTime;

    if (compiledLib.empty()) {
        m_stats.failedCompiles++;
        SetError("Failed to compile Mun script: " + filepath);
        return false;
    }

    m_stats.successfulCompiles++;

    std::cout << "[MunScriptSystem] Compiled in " << compileTime << " seconds: " << compiledLib << std::endl;

    // Load the compiled library
    if (!LoadCompiledLibrary(scriptName, compiledLib)) {
        return false;
    }

    // Record source file and watch for changes
    LoadedScript& script = m_loadedScripts[scriptName];
    script.sourceFile = filepath;
    script.compiledLib = compiledLib;

    if (m_autoHotReload) {
        WatchScriptFile(filepath);
    }

    std::cout << "[MunScriptSystem] Successfully loaded: " << scriptName << std::endl;
    return true;
}

/**
 * Compile Mun source file to native library
 */
bool MunScriptSystem::CompileScript(const std::string& filepath, const std::string& outputDir) {
    std::cout << "[MunScriptSystem] Compiling: " << filepath << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::string result = CompileMunSource(filepath);
    auto end = std::chrono::high_resolution_clock::now();

    double compileTime = std::chrono::duration<double>(end - start).count();
    std::cout << "[MunScriptSystem] Compile time: " << compileTime << " seconds" << std::endl;

    return !result.empty();
}

/**
 * Internal: Compile Mun source to library
 */
std::string MunScriptSystem::CompileMunSource(const std::string& sourceFile) {
    std::filesystem::path sourcePath(sourceFile);

    if (!std::filesystem::exists(sourcePath)) {
        SetError("Script file not found: " + sourceFile);
        return "";
    }

    if (sourcePath.extension() != ".mun") {
        SetError("Invalid file extension. Expected .mun");
        return "";
    }

    // Prepare Mun compiler command
    // Syntax: mun build <source> --output-dir <dir>
    std::string scriptName = sourcePath.stem().string();
    std::string outputDir = m_compilationOptions.targetDir;
    std::string buildCommand = "mun build \"" + sourceFile + "\" --output-dir \"" + outputDir + "\"";

    if (m_compilationOptions.optimize) {
        buildCommand += " --release";
    }

    if (m_compilationOptions.verbose) {
        buildCommand += " --verbose";
    }

    std::cout << "[MunScriptSystem] Executing: " << buildCommand << std::endl;

    // Execute the Mun compiler
#ifdef _WIN32
    int exitCode = system(buildCommand.c_str());
#else
    int exitCode = system(buildCommand.c_str());
#endif

    if (exitCode != 0) {
        SetError("Mun compilation failed with exit code: " + std::to_string(exitCode));
        return "";
    }

    // Construct path to compiled library
    std::filesystem::path compiledPath = outputDir;
    compiledPath /= scriptName;
    compiledPath += LIBRARY_EXTENSION;

    if (!std::filesystem::exists(compiledPath)) {
        SetError("Compiled library not found at: " + compiledPath.string());
        return "";
    }

    return compiledPath.string();
}

/**
 * Load a compiled library and cache function pointers
 */
bool MunScriptSystem::LoadCompiledLibrary(const std::string& scriptName, const std::string& libPath) {
    std::cout << "[MunScriptSystem] Loading library: " << libPath << std::endl;

    void* handle = LOAD_LIBRARY(libPath.c_str());

    if (!handle) {
#ifdef _WIN32
        SetError("Failed to load library: " + libPath);
#else
        SetError(std::string("Failed to load library: ") + dlerror());
#endif
        return false;
    }

    // Store library handle
    if (m_loadedScripts.find(scriptName) != m_loadedScripts.end()) {
        UnloadLibrary(scriptName);
    }

    m_loadedScripts[scriptName].libHandle = handle;

    std::cout << "[MunScriptSystem] Library loaded successfully" << std::endl;
    return true;
}

/**
 * Unload a previously loaded library
 */
void MunScriptSystem::UnloadLibrary(const std::string& scriptName) {
    auto it = m_loadedScripts.find(scriptName);
    if (it == m_loadedScripts.end()) return;

    LoadedScript& script = it->second;

    if (script.libHandle) {
        std::cout << "[MunScriptSystem] Unloading library: " << scriptName << std::endl;
        UNLOAD_LIBRARY(script.libHandle);
        script.libHandle = nullptr;
    }

    // Clear cached function pointers
    m_functionCache.erase(scriptName);
}

/**
 * Call a function in a loaded Mun script
 */
std::any MunScriptSystem::CallFunction(const std::string& functionName,
                                        const std::vector<std::any>& args) {
    // This is a simplified version - full implementation would need
    // dynamic function signature handling. For now, return error.
    SetError("Direct function calls require type information. Use LoadScript then access functions via the loaded library handle.");
    return std::any();
}

/**
 * Recompile and hot-reload a script
 */
bool MunScriptSystem::RecompileAndReload(const std::string& scriptName) {
    auto it = m_loadedScripts.find(scriptName);
    if (it == m_loadedScripts.end()) {
        SetError("Script not found: " + scriptName);
        return false;
    }

    LoadedScript& script = it->second;
    std::cout << "[MunScriptSystem] Hot-reloading: " << scriptName << std::endl;

    // Recompile
    auto start = std::chrono::high_resolution_clock::now();
    std::string newLib = CompileMunSource(script.sourceFile);
    auto end = std::chrono::high_resolution_clock::now();

    double compileTime = std::chrono::duration<double>(end - start).count();
    m_stats.totalReloads++;
    m_stats.lastCompileTime = compileTime;

    if (newLib.empty()) {
        m_stats.failedCompiles++;
        SetError("Failed to recompile: " + scriptName);
        return false;
    }

    m_stats.successfulCompiles++;

    // Unload old library
    UnloadLibrary(scriptName);

    // Load new library
    if (!LoadCompiledLibrary(scriptName, newLib)) {
        return false;
    }

    script.compiledLib = newLib;

    std::cout << "[MunScriptSystem] Hot-reload successful in " << compileTime << " seconds" << std::endl;

    // Trigger callback
    if (m_onScriptReloaded) {
        m_onScriptReloaded(scriptName);
    }

    return true;
}

/**
 * Watch a script file for changes
 */
void MunScriptSystem::WatchScriptFile(const std::string& filepath) {
    auto it = std::find(m_watchedFiles.begin(), m_watchedFiles.end(), filepath);
    if (it == m_watchedFiles.end()) {
        m_watchedFiles.push_back(filepath);
        std::cout << "[MunScriptSystem] Watching file: " << filepath << std::endl;
    }
}

/**
 * Stop watching a script file
 */
void MunScriptSystem::UnwatchScriptFile(const std::string& filepath) {
    auto it = std::find(m_watchedFiles.begin(), m_watchedFiles.end(), filepath);
    if (it != m_watchedFiles.end()) {
        m_watchedFiles.erase(it);
        std::cout << "[MunScriptSystem] Stopped watching: " << filepath << std::endl;
    }
}

/**
 * Watch entire directory for .mun files
 */
void MunScriptSystem::WatchScriptDirectory(const std::string& dirpath) {
    std::cout << "[MunScriptSystem] Watching directory: " << dirpath << std::endl;

    if (!std::filesystem::exists(dirpath)) {
        SetError("Directory does not exist: " + dirpath);
        return;
    }

    // Add all .mun files currently in directory
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dirpath)) {
        if (entry.path().extension() == ".mun") {
            WatchScriptFile(entry.path().string());
        }
    }
}

/**
 * Get list of loaded scripts
 */
std::vector<std::string> MunScriptSystem::GetLoadedScripts() const {
    std::vector<std::string> scripts;
    for (const auto& [name, _] : m_loadedScripts) {
        scripts.push_back(name);
    }
    return scripts;
}

/**
 * Get list of watched files
 */
std::vector<std::string> MunScriptSystem::GetWatchedFiles() const {
    return m_watchedFiles;
}

/**
 * Check for changes in watched files
 */
void MunScriptSystem::CheckForChanges() {
    for (const auto& filepath : m_watchedFiles) {
        if (!std::filesystem::exists(filepath)) continue;

        auto lastWriteTime = std::filesystem::last_write_time(filepath);
        std::filesystem::path path(filepath);
        std::string scriptName = path.stem().string();

        auto it = m_loadedScripts.find(scriptName);
        if (it == m_loadedScripts.end()) continue;

        LoadedScript& script = it->second;

        // Convert to time_t for comparison
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            lastWriteTime - std::filesystem::file_clock::now().time_since_epoch() + 
            std::chrono::system_clock::now().time_since_epoch()
        );

        if (script.lastModified != lastWriteTime) {
            script.lastModified = lastWriteTime;
            std::cout << "[MunScriptSystem] File changed: " << filepath << std::endl;
            RecompileAndReload(scriptName);
        }
    }
}

/**
 * Register types (placeholder for future reflection)
 */
void MunScriptSystem::RegisterTypes() {
    std::cout << "[MunScriptSystem] Registering types..." << std::endl;
    // Mun has built-in type metadata, this is here for consistency
}

/**
 * Check if type exists
 */
bool MunScriptSystem::HasType(const std::string& typeName) const {
    // Would check against loaded script type metadata
    (void)typeName;
    return false;
}

/**
 * Get memory usage
 */
uint64_t MunScriptSystem::GetMemoryUsage() const {
    // This would require profiling loaded libraries
    return 0;
}

/**
 * Get last execution time
 */
double MunScriptSystem::GetLastExecutionTime() const {
    return m_stats.lastCompileTime;
}

/**
 * Check if Mun compiler is available
 */
bool MunScriptSystem::IsMunCompilerAvailable() {
#ifdef _WIN32
    int result = system("mun --version > nul 2>&1");
#else
    int result = system("mun --version > /dev/null 2>&1");
#endif
    return result == 0;
}

/**
 * Get Mun compiler version
 */
std::string MunScriptSystem::GetMunCompilerVersion() {
    char buffer[128];
    FILE* pipe = nullptr;

#ifdef _WIN32
    pipe = _popen("mun --version 2>&1", "r");
#else
    pipe = popen("mun --version 2>&1", "r");
#endif

    if (!pipe) {
        return "unknown";
    }

    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    // Remove trailing whitespace
    result.erase(result.find_last_not_of("\n\r") + 1);
    return result;
}

/**
 * Get compiled library path
 */
std::string MunScriptSystem::GetCompiledLibraryPath(const std::string& scriptName) const {
    auto it = m_loadedScripts.find(scriptName);
    if (it != m_loadedScripts.end()) {
        return it->second.compiledLib;
    }
    return "";
}
