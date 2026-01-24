#include "KotlinScriptSystem.h"
#include <iostream>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <cstdlib>

namespace fs = std::filesystem;

void KotlinScriptSystem::Init() {
    if (initialized) {
        std::cout << "KotlinScriptSystem already initialized" << std::endl;
        return;
    }

    // Initialize Kotlin runtime
    kotlinRuntime = std::make_unique<KotlinRuntime>();
    
    KotlinRuntime::JVMConfig config;
    config.maxHeapSize = 512;      // 512MB max heap
    config.initialHeapSize = 256;  // 256MB initial
    config.enableAssertions = true;
    config.verboseOutput = false;

    if (!kotlinRuntime->Initialize(config)) {
        lastError = "Failed to initialize JVM";
        std::cerr << "KotlinScriptSystem: " << lastError << std::endl;
        return;
    }

    // Register native types that can be accessed from Kotlin
    RegisterTypes();

    initialized = true;
    std::cout << "KotlinScriptSystem Initialized (JVM enabled)" << std::endl;
}

void KotlinScriptSystem::Shutdown() {
    if (!initialized) return;

    // Clear active objects
    for (auto& obj : activeObjects) {
        kotlinRuntime->DeleteInstance(obj.second);
    }
    activeObjects.clear();

    // Shutdown runtime
    if (kotlinRuntime) {
        kotlinRuntime->Shutdown();
        kotlinRuntime.reset();
    }

    loadedScripts.clear();
    initialized = false;
    std::cout << "KotlinScriptSystem Shutdown" << std::endl;
}

void KotlinScriptSystem::Update(float deltaTime) {
    if (!initialized) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Request garbage collection every 100 frames
    static int frameCount = 0;
    if (++frameCount % 100 == 0) {
        kotlinRuntime->RequestGarbageCollection();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    lastExecutionTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

bool KotlinScriptSystem::RunScript(const std::string& filepath) {
    if (!initialized) {
        lastError = "KotlinScriptSystem not initialized";
        return false;
    }

    // Check if file exists
    if (!fs::exists(filepath)) {
        lastError = "Script file not found: " + filepath;
        std::cerr << "KotlinScriptSystem: " << lastError << std::endl;
        return false;
    }

    std::string ext = fs::path(filepath).extension().string();

    // If it's a .kt source file, compile it first
    if (ext == ".kt") {
        if (!CompileAndLoad(filepath)) {
            lastError = "Failed to compile Kotlin source: " + filepath;
            std::cerr << "KotlinScriptSystem: " << lastError << std::endl;
            return false;
        }
    }
    // If it's a compiled class file or JAR
    else if (ext == ".class" || ext == ".jar") {
        // Add to classpath
        std::string dir = fs::path(filepath).parent_path().string();
        if (!dir.empty()) {
            AddClassPath(dir);
        }

        // For .class files, extract class name from path
        if (ext == ".class") {
            std::string className = fs::path(filepath).stem().string();
            if (!LoadClass(className)) {
                lastError = "Failed to load Kotlin class: " + filepath;
                return false;
            }
        }
    } else {
        lastError = "Unsupported Kotlin file type: " + ext;
        return false;
    }

    loadedScripts.push_back(filepath);
    return true;
}

bool KotlinScriptSystem::ExecuteString(const std::string& source) {
    if (!initialized) {
        lastError = "KotlinScriptSystem not initialized";
        return false;
    }

    // Kotlin source code execution is typically done via compilation
    // For now, we support inline execution of pre-compiled code only
    // Runtime Kotlin eval would require a separate tool
    lastError = "Direct source execution not supported; use RunScript with .kt files";
    std::cerr << "KotlinScriptSystem: " << lastError << std::endl;
    return false;
}

void KotlinScriptSystem::RegisterTypes() {
    if (!initialized) return;

    // Register common C++ types that Kotlin might need
    // This would register callbacks for string conversion, math functions, etc.
    // Implementation depends on what native functions you want to expose
}

bool KotlinScriptSystem::HasType(const std::string& typeName) const {
    if (!initialized) return false;

    // Check if a class exists in the JVM
    return kotlinRuntime->IsClassLoaded(typeName);
}

std::any KotlinScriptSystem::CallFunction(const std::string& functionName,
                                         const std::vector<std::any>& args) {
    if (!initialized) {
        lastError = "KotlinScriptSystem not initialized";
        return std::any();
    }

    // Parse function name to extract class and method
    // Expected format: "ClassName.methodName" or "package.ClassName.methodName"
    size_t lastDot = functionName.rfind('.');
    if (lastDot == std::string::npos) {
        lastError = "Invalid function format; use 'ClassName.methodName'";
        return std::any();
    }

    std::string className = functionName.substr(0, lastDot);
    std::string methodName = functionName.substr(lastDot + 1);

    return CallStaticFunction(className, methodName, args);
}

void KotlinScriptSystem::ReloadScript(const std::string& filepath) {
    if (!initialized) return;

    // Unload the script
    auto it = std::find(loadedScripts.begin(), loadedScripts.end(), filepath);
    if (it != loadedScripts.end()) {
        loadedScripts.erase(it);

        // If it's a class, unload it
        std::string className = fs::path(filepath).stem().string();
        UnloadClass(className);
    }

    // Reload the script
    RunScript(filepath);
}

uint64_t KotlinScriptSystem::GetMemoryUsage() const {
    if (!initialized) return 0;
    return kotlinRuntime->GetHeapUsage();
}

bool KotlinScriptSystem::HasErrors() const {
    if (!initialized) return !lastError.empty();
    return kotlinRuntime->HasException() || !lastError.empty();
}

std::string KotlinScriptSystem::GetLastError() const {
    std::string error = lastError;
    if (kotlinRuntime && kotlinRuntime->HasException()) {
        if (!error.empty()) error += "; ";
        error += kotlinRuntime->GetLastException();
    }
    return error;
}

void KotlinScriptSystem::AddClassPath(const std::string& path) {
    if (!initialized) return;
    kotlinRuntime->AddClassPath(path);
}

bool KotlinScriptSystem::LoadClass(const std::string& className) {
    if (!initialized) return false;
    return kotlinRuntime->LoadClass(className);
}

void KotlinScriptSystem::UnloadClass(const std::string& className) {
    if (!initialized) return;
    kotlinRuntime->UnloadClass(className);
}

std::any KotlinScriptSystem::CallStaticFunction(const std::string& className,
                                               const std::string& methodName,
                                               const std::vector<std::any>& args) {
    if (!initialized) {
        lastError = "KotlinScriptSystem not initialized";
        return std::any();
    }

    try {
        auto startTime = std::chrono::high_resolution_clock::now();

        std::any result = kotlinRuntime->CallStaticMethod(className, methodName, args);

        auto endTime = std::chrono::high_resolution_clock::now();
        lastExecutionTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

        return result;
    } catch (const std::exception& e) {
        lastError = std::string(e.what());
        return std::any();
    }
}

KotlinRuntime::KotlinObject KotlinScriptSystem::CreateInstance(const std::string& className,
                                                              const std::vector<std::any>& constructorArgs) {
    if (!initialized) {
        return KotlinRuntime::KotlinObject{"", nullptr, false};
    }

    auto obj = kotlinRuntime->CreateInstance(className, constructorArgs);
    if (obj.isValid) {
        activeObjects[className] = obj;
    }
    return obj;
}

std::any KotlinScriptSystem::CallMethod(const KotlinRuntime::KotlinObject& obj,
                                       const std::string& methodName,
                                       const std::vector<std::any>& args) {
    if (!initialized) {
        lastError = "KotlinScriptSystem not initialized";
        return std::any();
    }

    try {
        auto startTime = std::chrono::high_resolution_clock::now();

        std::any result = kotlinRuntime->CallMethod(obj, methodName, args);

        auto endTime = std::chrono::high_resolution_clock::now();
        lastExecutionTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

        return result;
    } catch (const std::exception& e) {
        lastError = std::string(e.what());
        return std::any();
    }
}

void KotlinScriptSystem::CallSuspendFunction(const std::string& className,
                                            const std::string& methodName,
                                            const std::vector<std::any>& args,
                                            std::function<void(std::any)> callback) {
    if (!initialized) {
        if (callback) callback(std::any());
        return;
    }

    // For coroutine support, we'd need to integrate with Kotlin's suspend framework
    // This is a placeholder implementation
    std::cerr << "KotlinScriptSystem: Suspend function calls require additional coroutine integration" << std::endl;
}

bool KotlinScriptSystem::CompileKotlinFile(const std::string& sourceFile, const std::string& outputDir) {
    if (!fs::exists(sourceFile)) {
        lastError = "Source file not found: " + sourceFile;
        return false;
    }

    // Create output directory if needed
    fs::create_directories(outputDir);

    // Build kotlinc command
    std::string command = "kotlinc \"" + sourceFile + "\" -d \"" + outputDir + "\"";

    int result = std::system(command.c_str());
    if (result != 0) {
        lastError = "kotlinc compilation failed with code: " + std::to_string(result);
        return false;
    }

    return true;
}

int KotlinScriptSystem::CompileKotlinDirectory(const std::string& sourceDir, const std::string& outputDir) {
    if (!fs::is_directory(sourceDir)) {
        lastError = "Source directory not found: " + sourceDir;
        return 0;
    }

    fs::create_directories(outputDir);

    int compiledCount = 0;
    for (const auto& entry : fs::recursive_directory_iterator(sourceDir)) {
        if (entry.path().extension() == ".kt") {
            if (CompileKotlinFile(entry.path().string(), outputDir)) {
                compiledCount++;
            }
        }
    }

    return compiledCount;
}

std::vector<std::string> KotlinScriptSystem::GetLoadedClasses() const {
    return std::vector<std::string>();  // Would need to track in runtime
}

bool KotlinScriptSystem::IsClassLoaded(const std::string& className) const {
    if (!initialized) return false;
    return kotlinRuntime->IsClassLoaded(className);
}

void KotlinScriptSystem::RequestGarbageCollection() {
    if (!initialized) return;
    kotlinRuntime->RequestGarbageCollection();
}

bool KotlinScriptSystem::CompileAndLoad(const std::string& sourceFile) {
    // Create temporary output directory
    std::string outputDir = fs::path(sourceFile).parent_path().string() + "/kotlin_build";
    
    if (!CompileKotlinFile(sourceFile, outputDir)) {
        return false;
    }

    // Add compiled output to classpath
    AddClassPath(outputDir);

    // Extract class name from source filename
    std::string className = fs::path(sourceFile).stem().string();
    return LoadClass(className);
}

std::string KotlinScriptSystem::GetCompiledOutputPath(const std::string& sourceFile) {
    std::string parent = fs::path(sourceFile).parent_path().string();
    std::string stem = fs::path(sourceFile).stem().string();
    return parent + "/kotlin_build/" + stem + ".class";
}
