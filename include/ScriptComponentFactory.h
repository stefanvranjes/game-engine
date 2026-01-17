#pragma once

#include "ScriptComponent.h"
#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <map>

/**
 * @class ScriptComponentFactory
 * @brief Factory for creating script components with automatic language detection
 * 
 * Provides convenient methods to create ScriptComponent instances,
 * automatically detecting the scripting language from file extension
 * and managing their lifecycle.
 * 
 * Usage:
 * ```cpp
 * auto player = ScriptComponentFactory::CreateScriptComponent(
 *     "scripts/player.lua",
 *     gameObject
 * );
 * 
 * // Or explicitly specify language
 * auto npc = ScriptComponentFactory::CreateScriptComponent(
 *     "scripts/npc.ts",
 *     gameObject,
 *     ScriptLanguage::TypeScript
 * );
 * 
 * // Multi-script component
 * auto controller = ScriptComponentFactory::CreateMultiLanguageComponent(gameObject);
 * controller->AddScript("scripts/physics.rust");
 * controller->AddScript("scripts/input.lua");
 * controller->AddScript("scripts/ui.js");
 * ```
 */
class ScriptComponentFactory {
public:
    /**
     * Create a script component from a file
     * @param scriptPath Path to script file
     * @param gameObject GameObject to attach component to
     * @return Shared pointer to created ScriptComponent
     */
    static std::shared_ptr<ScriptComponent> CreateScriptComponent(
        const std::string& scriptPath,
        std::shared_ptr<class GameObject> gameObject
    );

    /**
     * Create a script component with explicit language
     * @param scriptPath Path to script file
     * @param gameObject GameObject to attach component to
     * @param language Script language to use
     * @return Shared pointer to created ScriptComponent
     */
    static std::shared_ptr<ScriptComponent> CreateScriptComponent(
        const std::string& scriptPath,
        std::shared_ptr<class GameObject> gameObject,
        ScriptLanguage language
    );

    /**
     * Create a multi-language script component
     * Allows attaching scripts in different languages to a single component
     * @param gameObject GameObject to attach component to
     * @return Shared pointer to created ScriptComponent
     */
    static std::shared_ptr<ScriptComponent> CreateMultiLanguageComponent(
        std::shared_ptr<class GameObject> gameObject
    );

    /**
     * Create a script component from source code
     * @param source Source code string
     * @param language Script language
     * @param gameObject GameObject to attach component to
     * @return Shared pointer to created ScriptComponent
     */
    static std::shared_ptr<ScriptComponent> CreateScriptComponentFromString(
        const std::string& source,
        ScriptLanguage language,
        std::shared_ptr<class GameObject> gameObject
    );

    /**
     * Clone a script component
     * @param source Script component to clone
     * @param newGameObject GameObject for cloned component
     * @return Shared pointer to cloned ScriptComponent
     */
    static std::shared_ptr<ScriptComponent> CloneScriptComponent(
        const std::shared_ptr<ScriptComponent>& source,
        std::shared_ptr<class GameObject> newGameObject
    );

    /**
     * Detect script language from file
     * @param scriptPath Path to script file
     * @return Detected language, Custom if unknown
     */
    static ScriptLanguage DetectLanguage(const std::string& scriptPath);

    /**
     * Check if a script language is supported by the factory
     * @param language Script language to check
     * @return true if supported, false otherwise
     */
    static bool IsLanguageSupported(ScriptLanguage language);

    /**
     * Get all supported file extensions
     * @return Map of extension to language
     */
    static std::map<std::string, ScriptLanguage> GetSupportedExtensions();

private:
    ScriptComponentFactory() = delete;
    ~ScriptComponentFactory() = delete;
};

/**
 * @class MultiLanguageScriptComponent
 * @brief Specialized script component supporting multiple languages
 * 
 * Allows a single GameObject to have scripts in different languages
 * that all work together as a cohesive unit.
 */
class MultiLanguageScriptComponent : public ScriptComponent {
public:
    explicit MultiLanguageScriptComponent(std::shared_ptr<class GameObject> gameObject);

    /**
     * Add a script in any language
     * @param scriptPath Path to script file
     * @return true if successful, false on error
     */
    bool AddScript(const std::string& scriptPath);

    /**
     * Add a script from source
     * @param source Source code
     * @param language Script language
     * @return true if successful, false on error
     */
    bool AddScript(const std::string& source, ScriptLanguage language);

    /**
     * Remove a script
     * @param scriptPath Path to previously added script
     * @return true if successful, false on error
     */
    bool RemoveScript(const std::string& scriptPath);

    /**
     * Get all attached scripts
     * @return Vector of script paths
     */
    const std::vector<std::string>& GetAttachedScripts() const { return m_AttachedScripts; }

    /**
     * Call a function across all languages
     * First language that has the function will execute it
     * @param functionName Function to call
     * @param args Arguments
     * @return Return value from function
     */
    std::any CallFunction(const std::string& functionName, const std::vector<std::any>& args);

    /**
     * Call a function in a specific language
     * @param language Script language
     * @param functionName Function to call
     * @param args Arguments
     * @return Return value from function
     */
    std::any CallFunction(ScriptLanguage language, const std::string& functionName,
                          const std::vector<std::any>& args);

    // Component interface
    void Init() override;
    void Update(float deltaTime) override;
    void OnDestroy() override;
    void OnCollisionEnter(std::shared_ptr<class GameObject> other) override;
    void OnCollisionExit(std::shared_ptr<class GameObject> other) override;

private:
    std::vector<std::string> m_AttachedScripts;
    std::map<std::string, ScriptLanguage> m_ScriptLanguages;
};
