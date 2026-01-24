#include "Wasm/WasmEngineBindings.h"
#include "Wasm/WasmInstance.h"
#include "Wasm/WasmModule.h"
#include "GameObject.h"
#include "Transform.h"
#include <iostream>

void WasmEngineBindings::RegisterBindings(std::shared_ptr<WasmInstance> instance) {
    if (!instance) {
        return;
    }

    RegisterPhysicsBindings(instance);
    RegisterAudioBindings(instance);
    RegisterRenderingBindings(instance);
    RegisterUIBindings(instance);
    RegisterInputBindings(instance);
    RegisterDebugBindings(instance);
}

std::shared_ptr<GameObject> WasmEngineBindings::GetGameObject(const std::string& name) {
    // This would interface with the scene manager to find GameObjects
    // Implementation depends on your GameObject registry structure
    return nullptr;
}

void WasmEngineBindings::RegisterPhysicsBindings(std::shared_ptr<WasmInstance> instance) {
    // Register physics functions
    instance->RegisterHostCallback("physics_apply_force", 
        [this](const std::vector<WasmValue>& args) {
            // Would handle force application
            return WasmValue::I32(1);
        }
    );
}

void WasmEngineBindings::RegisterAudioBindings(std::shared_ptr<WasmInstance> instance) {
    // Register audio functions
    instance->RegisterHostCallback("audio_play", 
        [this](const std::vector<WasmValue>& args) {
            // Would handle sound playback
            return WasmValue::I32(1);
        }
    );
}

void WasmEngineBindings::RegisterRenderingBindings(std::shared_ptr<WasmInstance> instance) {
    // Register rendering functions
    instance->RegisterHostCallback("render_set_color",
        [this](const std::vector<WasmValue>& args) {
            // Would handle rendering operations
            return WasmValue::I32(1);
        }
    );
}

void WasmEngineBindings::RegisterUIBindings(std::shared_ptr<WasmInstance> instance) {
    // Register UI functions
    instance->RegisterHostCallback("ui_log",
        [this](const std::vector<WasmValue>& args) {
            // Would handle UI operations
            return WasmValue::I32(1);
        }
    );
}

void WasmEngineBindings::RegisterInputBindings(std::shared_ptr<WasmInstance> instance) {
    // Register input functions
    instance->RegisterHostCallback("input_is_key_pressed",
        [this](const std::vector<WasmValue>& args) {
            // Would check input state
            return WasmValue::I32(0);
        }
    );
}

void WasmEngineBindings::RegisterDebugBindings(std::shared_ptr<WasmInstance> instance) {
    // Register debug functions
    instance->RegisterHostCallback("debug_log",
        [this](const std::vector<WasmValue>& args) {
            // Would output debug information
            return WasmValue::I32(1);
        }
    );
}

void WasmEngineBindings::LogFromWasm(const std::string& message) {
    std::cout << "[WASM] " << message << std::endl;
}

void WasmEngineBindings::RegisterBinding(std::shared_ptr<WasmInstance> instance,
                                         const std::string& name, WasmBinding binding) {
    if (!instance) {
        return;
    }

    instance->RegisterHostCallback(name,
        [binding, instance](const std::vector<WasmValue>& args) {
            return binding(instance, args);
        }
    );
}

WasmValue WasmEngineBindings::BindCreateRigidBody(std::shared_ptr<WasmInstance> instance,
                                                   const std::vector<WasmValue>& args) {
    // Implementation would create a rigid body
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindApplyForce(std::shared_ptr<WasmInstance> instance,
                                              const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindSetVelocity(std::shared_ptr<WasmInstance> instance,
                                               const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindCastRay(std::shared_ptr<WasmInstance> instance,
                                           const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindPlaySound(std::shared_ptr<WasmInstance> instance,
                                             const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindStopSound(std::shared_ptr<WasmInstance> instance,
                                             const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindSetVolume(std::shared_ptr<WasmInstance> instance,
                                             const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindSetMaterial(std::shared_ptr<WasmInstance> instance,
                                               const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindSetColor(std::shared_ptr<WasmInstance> instance,
                                            const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindSetPosition(std::shared_ptr<WasmInstance> instance,
                                               const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindGetPosition(std::shared_ptr<WasmInstance> instance,
                                               const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindSetRotation(std::shared_ptr<WasmInstance> instance,
                                               const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindGetRotation(std::shared_ptr<WasmInstance> instance,
                                               const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindSetScale(std::shared_ptr<WasmInstance> instance,
                                            const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindDebugLog(std::shared_ptr<WasmInstance> instance,
                                            const std::vector<WasmValue>& args) {
    // Read string from WASM memory and log it
    if (!args.empty() && args[0].type == WasmValueType::I32) {
        try {
            uint32_t offset = std::any_cast<int32_t>(args[0].value);
            std::string message = instance->ReadString(offset);
            LogFromWasm(message);
            return WasmValue::I32(1);
        } catch (...) {
            return WasmValue::I32(0);
        }
    }
    return WasmValue::I32(0);
}

WasmValue WasmEngineBindings::BindDebugDraw(std::shared_ptr<WasmInstance> instance,
                                             const std::vector<WasmValue>& args) {
    return WasmValue::I32(0);
}

