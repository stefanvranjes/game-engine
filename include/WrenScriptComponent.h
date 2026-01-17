#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>

// Forward declarations
class GameObject;
struct WrenVM;
struct WrenHandle;

/**
 * @class WrenScriptComponent
 * @brief Script component that executes Wren scripts on a GameObject
 * 
 * Provides lifecycle management (Init, Update, Destroy) for Wren scripts
 * attached to GameObjects. Supports multiple scripts per component and
 * automatic function binding.
 * 
 * Usage:
 * ```cpp
 * auto scriptComp = std::make_shared<WrenScriptComponent>(gameObject);
 * scriptComp->LoadScript("player_behavior.wren");
 * gameObject->SetScriptComponent(scriptComp);
 * ```
 * 
 * In Wren:
 * ```wren
 * class PlayerBehavior {
 *     init() {
 *         System.print("Player initialized")
 *     }
 *     
 *     update(dt) {
 *         System.print("Player updating: %(dt)")
 *     }
 *     
 *     destroy() {
 *         System.print("Player destroyed")
 *     }
 *     
 *     onCollision(other) {
 *         System.print("Hit: %(other)")
 *     }
 * }
 * 
 * var _player = null
 * 
 * construct init() {
 *     _player = PlayerBehavior.new()
 *     _player.init()
 * }
 * 
 * construct update(dt) {
 *     _player.update(dt)
 * }
 * 
 * construct destroy() {
 *     _player.destroy()
 * }
 * ```
 */
class WrenScriptComponent {
public:
    using UpdateCallback = std::function<void(float)>;
    using EventCallback = std::function<void(const std::string&)>;

    /**
     * Create a Wren script component
     * @param owner Weak pointer to owning GameObject
     */
    explicit WrenScriptComponent(std::weak_ptr<GameObject> owner);
    
    ~WrenScriptComponent();

    /**
     * Load a Wren script file
     * @param filepath Path to the .wren script file
     * @return true if successful, false on error
     */
    bool LoadScript(const std::string& filepath);

    /**
     * Load multiple scripts (combines them in execution order)
     * @param filepaths Vector of script file paths
     * @return true if all scripts loaded successfully
     */
    bool LoadScripts(const std::vector<std::string>& filepaths);

    /**
     * Load Wren code from a string
     * @param source Wren source code
     * @param scriptName Optional name for the script (for debugging)
     * @return true if successful, false on error
     */
    bool LoadCode(const std::string& source, const std::string& scriptName = "inline");

    /**
     * Initialize the script (calls init() if defined)
     * @return true if successful, false on error
     */
    bool Init();

    /**
     * Update the script (calls update(dt) if defined)
     * @param deltaTime Time elapsed since last frame in seconds
     * @return true if successful, false on error
     */
    bool Update(float deltaTime);

    /**
     * Shutdown and cleanup the script (calls destroy() if defined)
     * @return true if successful, false on error
     */
    bool Destroy();

    /**
     * Reload all loaded scripts
     * @return true if successful, false on error
     */
    bool Reload();

    /**
     * Get the owning GameObject (if still valid)
     * @return Shared pointer to GameObject or nullptr
     */
    std::shared_ptr<GameObject> GetOwner() const;

    /**
     * Get loaded script paths
     * @return Vector of loaded script file paths
     */
    const std::vector<std::string>& GetLoadedScripts() const { return m_LoadedScripts; }

    /**
     * Check if a function exists in the script
     * @param functionName Function name (e.g., "init", "update")
     * @return true if function exists, false otherwise
     */
    bool HasFunction(const std::string& functionName) const;

    /**
     * Invoke a custom event in the script
     * @param eventName Event name (e.g., "onCollision", "onTrigger")
     * @param args Event arguments as a Wren-compatible string
     * @return true if successful, false on error
     */
    bool InvokeEvent(const std::string& eventName, const std::string& args = "");

    /**
     * Set a variable in the script's context
     * @param varName Variable name
     * @param value Value as Wren code
     */
    void SetVariable(const std::string& varName, const std::string& value);

    /**
     * Get a variable from the script's context
     * @param varName Variable name
     * @return String representation of the variable value
     */
    std::string GetVariable(const std::string& varName) const;

    /**
     * Enable/disable script updates
     * @param enabled true to enable updates, false to disable
     */
    void SetUpdateEnabled(bool enabled) { m_UpdateEnabled = enabled; }

    /**
     * Check if script updates are enabled
     * @return true if updates are enabled
     */
    bool IsUpdateEnabled() const { return m_UpdateEnabled; }

    /**
     * Get the Wren VM pointer (for advanced usage)
     * @return WrenVM pointer or nullptr
     */
    WrenVM* GetVM() const;

    /**
     * Get the script module name
     * @return Module name derived from first loaded script
     */
    const std::string& GetModuleName() const { return m_ModuleName; }

    /**
     * Subscribe to script events (like error callbacks)
     * @param eventName Event name
     * @param callback Function to call when event occurs
     */
    void OnEvent(const std::string& eventName, EventCallback callback);

private:
    std::weak_ptr<GameObject> m_Owner;
    std::vector<std::string> m_LoadedScripts;
    std::string m_ModuleName;
    std::vector<WrenHandle*> m_FunctionHandles;
    bool m_UpdateEnabled = true;
    bool m_Initialized = false;
    
    std::map<std::string, EventCallback> m_EventCallbacks;

    /**
     * Execute a Wren function without arguments
     * @param functionName Function name
     * @return true if successful
     */
    bool ExecuteFunction(const std::string& functionName);

    /**
     * Execute a Wren function with float argument (for delta time)
     * @param functionName Function name
     * @param argument Float value to pass
     * @return true if successful
     */
    bool ExecuteFunctionWithFloat(const std::string& functionName, float argument);

    /**
     * Cache function handles for frequently called functions
     */
    void CacheFunctionHandles();

    /**
     * Release cached function handles
     */
    void ReleaseFunctionHandles();
};

/**
 * Helper class for easier Wren script management
 */
class WrenScriptFactory {
public:
    /**
     * Create a Wren script component for a GameObject
     * @param gameObject Shared pointer to the GameObject
     * @param scriptPath Path to the Wren script file
     * @return Shared pointer to the created component
     */
    static std::shared_ptr<WrenScriptComponent> CreateComponent(
        std::shared_ptr<GameObject> gameObject,
        const std::string& scriptPath);

    /**
     * Create and attach a Wren script component
     * @param gameObject Shared pointer to the GameObject
     * @param scriptPaths Vector of script paths to load
     * @return Shared pointer to the created component
     */
    static std::shared_ptr<WrenScriptComponent> CreateComponent(
        std::shared_ptr<GameObject> gameObject,
        const std::vector<std::string>& scriptPaths);

    /**
     * Load a script into an existing component
     * @param component Shared pointer to the component
     * @param scriptPath Path to the script file
     * @return true if successful
     */
    static bool AttachScript(
        std::shared_ptr<WrenScriptComponent> component,
        const std::string& scriptPath);
};
