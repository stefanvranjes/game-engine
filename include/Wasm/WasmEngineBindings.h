#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <glm/glm.hpp>
#include "Wasm/WasmInstance.h"

// Forward declarations
class GameObject;
class Transform;
class AudioSource;
class ParticleEmitter;

/**
 * @class WasmEngineBindings
 * @brief Provides C++ engine functionality accessible from WASM modules
 * 
 * Implements a bridge between WebAssembly and the game engine, allowing
 * WASM scripts to interact with GameObjects, physics, audio, rendering, etc.
 */
class WasmEngineBindings {
public:
    static WasmEngineBindings& GetInstance() {
        static WasmEngineBindings instance;
        return instance;
    }

    /**
     * Register all engine bindings with a WASM instance
     * Should be called when a WASM module is loaded
     */
    void RegisterBindings(std::shared_ptr<WasmInstance> instance);

    /**
     * Get GameObject by name (for binding scripts to objects)
     */
    std::shared_ptr<GameObject> GetGameObject(const std::string& name);

    /**
     * Create binding group for specific subsystem
     */
    void RegisterPhysicsBindings(std::shared_ptr<WasmInstance> instance);
    void RegisterAudioBindings(std::shared_ptr<WasmInstance> instance);
    void RegisterRenderingBindings(std::shared_ptr<WasmInstance> instance);
    void RegisterUIBindings(std::shared_ptr<WasmInstance> instance);
    void RegisterInputBindings(std::shared_ptr<WasmInstance> instance);
    void RegisterDebugBindings(std::shared_ptr<WasmInstance> instance);

    /**
     * Log message from WASM (for debug output)
     */
    void LogFromWasm(const std::string& message);

    /**
     * Generic binding registration for custom functions
     */
    using WasmBinding = std::function<WasmValue(std::shared_ptr<WasmInstance>, const std::vector<WasmValue>&)>;
    void RegisterBinding(std::shared_ptr<WasmInstance> instance, 
                        const std::string& name, WasmBinding binding);

private:
    WasmEngineBindings() = default;
    
    // Prevent copying
    WasmEngineBindings(const WasmEngineBindings&) = delete;
    WasmEngineBindings& operator=(const WasmEngineBindings&) = delete;

    std::unordered_map<std::string, WasmBinding> m_Bindings;

    // Physics bindings
    WasmValue BindCreateRigidBody(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindApplyForce(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindSetVelocity(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindCastRay(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);

    // Audio bindings
    WasmValue BindPlaySound(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindStopSound(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindSetVolume(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);

    // Rendering bindings
    WasmValue BindSetMaterial(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindSetColor(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);

    // Transform bindings
    WasmValue BindSetPosition(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindGetPosition(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindSetRotation(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindGetRotation(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindSetScale(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);

    // Debug bindings
    WasmValue BindDebugLog(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
    WasmValue BindDebugDraw(std::shared_ptr<WasmInstance> instance, const std::vector<WasmValue>& args);
};

