#include "WrenScriptSystem.h"
#include "GameObject.h"
#include "Transform.h"
#include "Math/Vec3.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

// Wren handles are already available via wren.h included in the header

WrenScriptSystem::WrenScriptSystem() : m_VM(nullptr) {
    m_PrintHandler = [](const std::string& msg) {
        std::cout << "[Wren] " << msg << std::endl;
    };
    m_ErrorHandler = [](const std::string& msg) {
        std::cerr << "[Wren Error] " << msg << std::endl;
    };
}

WrenScriptSystem::~WrenScriptSystem() {
    Shutdown();
}

void WrenScriptSystem::Init() {
    if (m_VM) return; // Already initialized
    
    m_VM = wrenNewVM();
    if (!m_VM) {
        std::cerr << "Failed to create Wren VM" << std::endl;
        return;
    }
    
    // Register all built-in engine bindings
    RegisterGameObjectBindings();
    RegisterTransformBindings();
    RegisterVec3Bindings();
    RegisterPhysicsBindings();
    RegisterAudioBindings();
    RegisterParticleBindings();
    RegisterTimeBindings();
    RegisterInputBindings();
    RegisterUtilityBindings();
    
    std::cout << "WrenScriptSystem Initialized" << std::endl;
}

void WrenScriptSystem::Shutdown() {
    if (m_VM) {
        wrenFreeVM(m_VM);
        m_VM = nullptr;
    }
    m_LoadedScripts.clear();
}

void WrenScriptSystem::Update(float deltaTime) {
    // Call Update on all registered script instances if needed
    // This would iterate through active scripts and call their update() method
    wrenCollectGarbage(m_VM);
}

bool WrenScriptSystem::RunScript(const std::string& filepath) {
    if (!m_VM) {
        m_ErrorHandler("Wren VM not initialized");
        return false;
    }
    
    // Read script file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        m_ErrorHandler("Failed to open script: " + filepath);
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    file.close();
    
    // Get module name from filepath
    std::string moduleName = std::filesystem::path(filepath).stem().string();
    
    // Execute script
    WrenInterpretResult result = wrenInterpret(m_VM, moduleName.c_str(), source.c_str());
    
    bool success = (result == WREN_RESULT_SUCCESS);
    if (success) {
        m_LoadedScripts.push_back(filepath);
        m_PrintHandler("Script loaded: " + filepath);
    } else {
        m_ErrorHandler("Failed to load script: " + filepath);
    }
    
    return success;
}

bool WrenScriptSystem::ExecuteString(const std::string& source) {
    if (!m_VM) {
        m_ErrorHandler("Wren VM not initialized");
        return false;
    }
    
    WrenInterpretResult result = wrenInterpret(m_VM, "main", source.c_str());
    return result == WREN_RESULT_SUCCESS;
}

bool WrenScriptSystem::CallFunction(const std::string& functionName, const std::string& args) {
    if (!m_VM) {
        m_ErrorHandler("Wren VM not initialized");
        return false;
    }
    
    // Build the call signature
    std::string signature = functionName + "()";
    
    WrenHandle* handle = wrenMakeCallHandle(m_VM, signature.c_str());
    if (!handle) {
        m_ErrorHandler("Function not found: " + functionName);
        return false;
    }
    
    WrenInterpretResult result = wrenCall(m_VM, handle);
    wrenReleaseHandle(m_VM, handle);
    
    return result == WREN_RESULT_SUCCESS;
}

bool WrenScriptSystem::CallMethod(const std::string& instanceName, 
                                  const std::string& methodName, 
                                  const std::string& args) {
    if (!m_VM) {
        m_ErrorHandler("Wren VM not initialized");
        return false;
    }
    
    // Build method call signature
    std::string signature = methodName + "()";
    
    WrenHandle* handle = wrenMakeCallHandle(m_VM, signature.c_str());
    if (!handle) {
        m_ErrorHandler("Method not found: " + methodName);
        return false;
    }
    
    // TODO: Push instance to stack and call
    WrenInterpretResult result = wrenCall(m_VM, handle);
    wrenReleaseHandle(m_VM, handle);
    
    return result == WREN_RESULT_SUCCESS;
}

void WrenScriptSystem::SetGlobalVariable(const std::string& varName, const std::string& value) {
    if (!m_VM) return;
    
    std::string code = varName + " = " + value;
    ExecuteString(code);
}

std::string WrenScriptSystem::GetGlobalVariable(const std::string& varName) {
    if (!m_VM) return "";
    
    // Retrieve global variable - implementation depends on Wren's actual API
    // This is a placeholder
    return "";
}

void WrenScriptSystem::RegisterNativeMethod(const std::string& className,
                                           const std::string& methodName,
                                           int numParams,
                                           WrenForeignFunction function) {
    if (!m_VM) return;
    
    // Implementation would register the method with Wren's foreign function mechanism
    // This is a simplified placeholder
}

void WrenScriptSystem::RegisterStaticMethod(const std::string& moduleName,
                                           const std::string& className,
                                           const std::string& methodName,
                                           int numParams,
                                           WrenForeignFunction function) {
    if (!m_VM) return;
    
    // Register static method with Wren
    // This is a simplified placeholder
}

void WrenScriptSystem::ReloadAll() {
    if (!m_VM) return;
    
    auto scripts = m_LoadedScripts;
    m_LoadedScripts.clear();
    
    for (const auto& scriptPath : scripts) {
        RunScript(scriptPath);
    }
}

bool WrenScriptSystem::HasFunction(const std::string& functionName) {
    if (!m_VM) return false;
    
    std::string code = functionName + " != null && Fiber.isValid";
    // This would need actual Wren API integration to check existence
    return true;
}

bool WrenScriptSystem::HasVariable(const std::string& variableName) {
    if (!m_VM) return false;
    
    // Check if variable exists in Wren
    return true;
}

// ============================================================================
// Built-in Bindings Implementation
// ============================================================================

void WrenScriptSystem::RegisterGameObjectBindings() {
    if (!m_VM) return;
    
    // Define Wren class for GameObject binding
    std::string gameObjectClass = R"(
        foreign class GameObject {
            construct new(name) { }
            foreign name
            foreign transform
            foreign active
            foreign layer
            foreign setActive(active)
            foreign destroy()
            foreign addComponent(componentType)
            foreign getComponent(componentType)
            foreign tag
            foreign setTag(tag)
        }
    )";
    
    ExecuteString(gameObjectClass);
}

void WrenScriptSystem::RegisterTransformBindings() {
    if (!m_VM) return;
    
    std::string transformClass = R"(
        foreign class Transform {
            foreign position
            foreign rotation
            foreign scale
            foreign setPosition(x, y, z)
            foreign setRotation(x, y, z, w)
            foreign setScale(x, y, z)
            foreign parent
            foreign children
            foreign worldPosition
            foreign localPosition
            foreign forward
            foreign right
            foreign up
            foreign translate(x, y, z)
            foreign rotate(x, y, z)
        }
    )";
    
    ExecuteString(transformClass);
}

void WrenScriptSystem::RegisterVec3Bindings() {
    if (!m_VM) return;
    
    std::string vec3Class = R"(
        foreign class Vec3 {
            construct new(x, y, z) { }
            foreign x
            foreign y
            foreign z
            foreign magnitude
            foreign normalized
            foreign dot(other)
            foreign cross(other)
            foreign distance(other)
            foreign toString
        }
    )";
    
    ExecuteString(vec3Class);
}

void WrenScriptSystem::RegisterPhysicsBindings() {
    if (!m_VM) return;
    
    std::string physicsClass = R"(
        foreign class RigidBody {
            foreign mass
            foreign velocity
            foreign angularVelocity
            foreign setVelocity(x, y, z)
            foreign applyForce(x, y, z)
            foreign applyTorque(x, y, z)
            foreign applyImpulse(x, y, z)
            foreign isKinematic
            foreign setKinematic(kinematic)
            foreign linearDamping
            foreign angularDamping
            foreign useGravity
            foreign setUseGravity(useGravity)
        }
        
        foreign class Collider {
            foreign isTrigger
            foreign setTrigger(isTrigger)
            foreign material
            foreign shape
            foreign onCollisionEnter
            foreign onCollisionStay
            foreign onCollisionExit
        }
    )";
    
    ExecuteString(physicsClass);
}

void WrenScriptSystem::RegisterAudioBindings() {
    if (!m_VM) return;
    
    std::string audioClass = R"(
        foreign class AudioSource {
            foreign clip
            foreign volume
            foreign pitch
            foreign loop
            foreign spatialBlend
            foreign dopplerLevel
            foreign play()
            foreign pause()
            foreign stop()
            foreign playOneShotAtPoint(clip, x, y, z, volume)
            foreign isPlaying
        }
    )";
    
    ExecuteString(audioClass);
}

void WrenScriptSystem::RegisterParticleBindings() {
    if (!m_VM) return;
    
    std::string particleClass = R"(
        foreign class ParticleSystem {
            foreign emission
            foreign emissionRate
            foreign lifetime
            foreign speed
            foreign size
            foreign simulationSpace
            foreign play()
            foreign stop()
            foreign pause()
            foreign emit(count)
        }
    )";
    
    ExecuteString(particleClass);
}

void WrenScriptSystem::RegisterTimeBindings() {
    if (!m_VM) return;
    
    std::string timeClass = R"(
        class Time {
            static deltaTime { __deltaTime }
            static time { __time }
            static timeScale { __timeScale }
            static frameCount { __frameCount }
            static fixedDeltaTime { __fixedDeltaTime }
            static setTimeScale(scale) {
                __timeScale = scale
            }
        }
    )";
    
    ExecuteString(timeClass);
}

void WrenScriptSystem::RegisterInputBindings() {
    if (!m_VM) return;
    
    std::string inputClass = R"(
        class Input {
            static getKey(keyCode) { false }
            static getKeyDown(keyCode) { false }
            static getKeyUp(keyCode) { false }
            static getMouseButton(button) { false }
            static getMouseButtonDown(button) { false }
            static getMouseButtonUp(button) { false }
            static getAxis(axisName) { 0.0 }
            static mousePosition { Vec3.new(0, 0, 0) }
            static isInputActive { true }
        }
    )";
    
    ExecuteString(inputClass);
}

void WrenScriptSystem::RegisterUtilityBindings() {
    if (!m_VM) return;
    
    std::string utilityClass = R"(
        class Debug {
            static log(message) {
                System.print("[DEBUG] %(message)")
            }
            static warn(message) {
                System.print("[WARN] %(message)")
            }
            static error(message) {
                System.print("[ERROR] %(message)")
            }
            static drawLine(start, end, color) { }
            static drawRay(origin, direction, color) { }
            static drawBox(center, size, color) { }
        }
        
        class Mathf {
            static pi { 3.14159265359 }
            static e { 2.71828182846 }
            static infinity { 1.0 / 0.0 }
            static neg_infinity { -1.0 / 0.0 }
            
            static abs(x) { x > 0 ? x : -x }
            static max(a, b) { a > b ? a : b }
            static min(a, b) { a < b ? a : b }
            static clamp(value, min, max) {
                if (value < min) return min
                if (value > max) return max
                return value
            }
            static lerp(a, b, t) { a + (b - a) * t }
            static smoothstep(min, max, x) {
                var t = clamp((x - min) / (max - min), 0, 1)
                return t * t * (3 - 2 * t)
            }
        }
    )";
    
    ExecuteString(utilityClass);
}

void WrenScriptSystem::HandleError(const std::string& error) {
    WrenScriptSystem::GetInstance().m_ErrorHandler(error);
}
