#pragma once

#include "MunScriptSystem.h"
#include <memory>
#include <iostream>

/**
 * @file MunScriptIntegrationExample.h
 * @brief Example integration of MunScriptSystem into your game application
 * 
 * This file shows how to incorporate Mun language support into your
 * game engine's Application class for compiled hot-reload scripting.
 */

// Example: Adding to your Application class
class ApplicationWithMun {
private:
    std::unique_ptr<MunScriptSystem> m_munScriptSystem;
    bool m_useCompiledScripting = true;

public:
    /**
     * Initialize Mun scripting system
     * Called during Application::Init()
     */
    bool InitializeMunScripting() {
        std::cout << "[Application] Initializing Mun scripting system..." << std::endl;

        m_munScriptSystem = std::make_unique<MunScriptSystem>();
        m_munScriptSystem->Init();

        if (m_munScriptSystem->HasErrors()) {
            std::cerr << "[Application] Mun init failed: "
                      << m_munScriptSystem->GetLastError() << std::endl;
            m_useCompiledScripting = false;
            return false;
        }

        // Set up compilation options for development
        MunScriptSystem::CompilationOptions options;
        options.optimize = false;      // Faster compilation during dev
        options.verbose = false;       // Reduce console spam
        options.emitMetadata = true;   // Enable runtime reflection

        m_munScriptSystem->SetCompilationOptions(options);

        // Set hot-reload callback
        m_munScriptSystem->SetOnScriptReloaded([this](const std::string& scriptName) {
            OnScriptReloaded(scriptName);
        });

        // Load all gameplay scripts
        return LoadGameplayScripts();
    }

    /**
     * Load all Mun scripts for gameplay
     */
    bool LoadGameplayScripts() {
        if (!m_useCompiledScripting) return false;

        std::cout << "[Application] Loading Mun gameplay scripts..." << std::endl;

        // List of scripts to load
        const std::vector<std::string> scripts = {
            "scripts/gameplay.mun",
            "scripts/ai.mun",
            "scripts/physics.mun",
            "scripts/combat.mun",
        };

        bool allLoaded = true;
        for (const auto& script : scripts) {
            if (!m_munScriptSystem->LoadScript(script)) {
                std::cerr << "[Application] Failed to load: " << script << std::endl;
                std::cerr << "[Application] Error: "
                          << m_munScriptSystem->GetLastError() << std::endl;
                allLoaded = false;
            } else {
                std::cout << "[Application] Loaded: " << script << std::endl;
            }
        }

        // Start watching scripts directory for changes
        m_munScriptSystem->WatchScriptDirectory("scripts/");

        return allLoaded;
    }

    /**
     * Called when a Mun script is hot-reloaded
     */
    void OnScriptReloaded(const std::string& scriptName) {
        std::cout << "[Application] Script reloaded: " << scriptName << std::endl;

        // Optionally: Trigger game state updates, reinitialize systems, etc.
        // Example: When AI script reloads, reset all AI entities
        if (scriptName == "ai") {
            std::cout << "[Application] Re-initializing AI systems..." << std::endl;
            // ResetAISystems();
        }

        // Example: When gameplay script reloads, refresh balancing
        if (scriptName == "gameplay") {
            std::cout << "[Application] Refreshing gameplay parameters..." << std::endl;
            // RefreshGameplayParameters();
        }
    }

    /**
     * Called during main game loop for updates
     */
    void UpdateMunScripting(float deltaTime) {
        if (!m_useCompiledScripting) return;

        // This checks for file changes and recompiles/reloads if needed
        m_munScriptSystem->Update(deltaTime);

        // Display hot-reload status in debug mode
        if (false) {  // Set to true for development
            PrintMunDebugInfo();
        }
    }

    /**
     * Shutdown Mun scripting system
     * Called during Application::Shutdown()
     */
    void ShutdownMunScripting() {
        std::cout << "[Application] Shutting down Mun scripting system..." << std::endl;

        if (m_munScriptSystem) {
            m_munScriptSystem->Shutdown();
            m_munScriptSystem.reset();
        }
    }

    /**
     * Manually trigger recompilation of a script
     * Useful for forcing reload without waiting for file changes
     */
    bool RecompileScript(const std::string& scriptName) {
        if (!m_useCompiledScripting) return false;

        std::cout << "[Application] Recompiling: " << scriptName << std::endl;
        return m_munScriptSystem->RecompileAndReload(scriptName);
    }

    /**
     * Get statistics about Mun compilation
     */
    void PrintMunDebugInfo() {
        if (!m_munScriptSystem) return;

        const auto& stats = m_munScriptSystem->GetCompilationStats();

        std::cout << "\n=== Mun System Statistics ===" << std::endl;
        std::cout << "Loaded Scripts: " << m_munScriptSystem->GetLoadedScripts().size() << std::endl;

        for (const auto& script : m_munScriptSystem->GetLoadedScripts()) {
            std::cout << "  - " << script << std::endl;
        }

        std::cout << "\nCompilation Stats:" << std::endl;
        std::cout << "  Total Compiles: " << stats.totalCompiles << std::endl;
        std::cout << "  Successful: " << stats.successfulCompiles << std::endl;
        std::cout << "  Failed: " << stats.failedCompiles << std::endl;
        std::cout << "  Hot-Reloads: " << stats.totalReloads << std::endl;
        std::cout << "  Total Time: " << stats.totalCompileTime << "s" << std::endl;
        std::cout << "  Last Compile: " << stats.lastCompileTime << "s" << std::endl;

        std::cout << "\nWatched Files: " << m_munScriptSystem->GetWatchedFiles().size() << std::endl;
        for (const auto& file : m_munScriptSystem->GetWatchedFiles()) {
            std::cout << "  - " << file << std::endl;
        }

        std::cout << "============================\n" << std::endl;
    }

    /**
     * Check if Mun scripting is enabled
     */
    bool IsCompiledScriptingEnabled() const {
        return m_useCompiledScripting;
    }

    /**
     * Get reference to Mun system (for advanced usage)
     */
    MunScriptSystem* GetMunScriptSystem() {
        return m_munScriptSystem.get();
    }
};

/**
 * INTEGRATION CHECKLIST
 * 
 * To integrate Mun into your existing Application class:
 * 
 * 1. Add member variable:
 *    std::unique_ptr<MunScriptSystem> m_munScriptSystem;
 * 
 * 2. Call in Application::Init():
 *    InitializeMunScripting();
 * 
 * 3. Call in Application::Update():
 *    UpdateMunScripting(deltaTime);
 * 
 * 4. Call in Application::Shutdown():
 *    ShutdownMunScripting();
 * 
 * 5. Add ImGui editor panel (optional):
 *    void ShowMunEditorPanel() {
 *        ImGui::Begin("Mun Scripts");
 *        
 *        for (const auto& script : m_munScriptSystem->GetLoadedScripts()) {
 *            ImGui::BulletText("%s", script.c_str());
 *        }
 *        
 *        if (ImGui::Button("Force Reload All")) {
 *            for (const auto& script : m_munScriptSystem->GetLoadedScripts()) {
 *                RecompileScript(script);
 *            }
 *        }
 *        
 *        ImGui::End();
 *    }
 * 
 * 6. Build with Mun system included:
 *    # In CMakeLists.txt
 *    add_executable(GameEngine
 *        ...
 *        src/MunScriptSystem.cpp
 *    )
 * 
 * 7. Verify Mun compiler is installed:
 *    mun --version
 * 
 * 8. Create scripts in scripts/ directory:
 *    scripts/gameplay.mun
 *    scripts/ai.mun
 *    etc.
 * 
 * 9. Test hot-reload:
 *    - Edit scripts/gameplay.mun
 *    - Save file
 *    - Watch console for "Script reloaded" message
 *    - Verify game behavior updated
 */

/**
 * WORKFLOW EXAMPLE
 * 
 * 1. Start engine:
 *    ./GameEngine
 * 
 * 2. Engine initializes:
 *    [Application] Initializing Mun scripting system...
 *    [MunScriptSystem] Initializing...
 *    [MunScriptSystem] Mun compiler version: mun 0.4.0
 *    [Application] Loading Mun gameplay scripts...
 *    [MunScriptSystem] Loading script: scripts/gameplay.mun
 *    [MunScriptSystem] Compiled in 0.45 seconds: mun-target/gameplay.dll
 *    [MunScriptSystem] Successfully loaded: gameplay
 *    [MunScriptSystem] Watching file: scripts/gameplay.mun
 * 
 * 3. Edit gameplay.mun in your editor:
 *    pub fn new_calculation() -> f32 { ... }
 * 
 * 4. Save file:
 *    [MunScriptSystem] File changed: scripts/gameplay.mun
 *    [MunScriptSystem] Hot-reloading: gameplay
 *    [MunScriptSystem] Compiled in 0.23 seconds: mun-target/gameplay.dll
 *    [MunScriptSystem] Hot-reload successful in 0.23 seconds
 *    [Application] Script reloaded: gameplay
 *    [Application] Refreshing gameplay parameters...
 * 
 * 5. Game immediately uses updated code!
 */
