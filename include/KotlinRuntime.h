#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include <any>
#include <jni.h>
#include <stdint.h>

/**
 * @class KotlinRuntime
 * @brief JVM runtime wrapper for Kotlin script execution
 * 
 * Manages:
 * - JVM lifecycle (creation, configuration, shutdown)
 * - Kotlin class loading and bytecode execution
 * - Method invocation via reflection
 * - Exception handling and translation
 * - Memory management and garbage collection control
 * 
 * Kotlin is ideal for:
 * - Type-safe game logic with null safety
 * - Full OOP design patterns (inheritance, composition, interfaces)
 * - Coroutines for async/concurrent behavior
 * - Data classes for game state representation
 * - Extension functions for clean APIs
 * - Seamless Java/Kotlin library interoperability
 * 
 * Usage Example:
 * ```cpp
 * auto kotlinRuntime = std::make_unique<KotlinRuntime>();
 * kotlinRuntime->Initialize();
 * 
 * // Load compiled Kotlin classes
 * kotlinRuntime->AddClassPath("scripts/bin");
 * kotlinRuntime->LoadClass("gameplay.PlayerController");
 * 
 * // Call Kotlin function
 * std::vector<std::any> args = {playerObj, deltaTime};
 * auto result = kotlinRuntime->CallFunction("gameplay.PlayerController", "update", args);
 * 
 * kotlinRuntime->Shutdown();
 * ```
 */
class KotlinRuntime {
public:
    /**
     * Represents a loaded Kotlin class with cached method references
     */
    struct KotlinClass {
        std::string className;
        jclass javaClass;
        std::map<std::string, jmethodID> methods;
        std::map<std::string, jfieldID> fields;
        bool isLoaded;
    };

    /**
     * Represents an instance of a Kotlin class
     */
    struct KotlinObject {
        std::string className;
        jobject instance;
        bool isValid;
    };

    /**
     * JVM configuration options
     */
    struct JVMConfig {
        int maxHeapSize;           // Maximum heap size in MB (default 512)
        int initialHeapSize;       // Initial heap size in MB (default 256)
        std::vector<std::string> classPaths;  // Additional classpaths
        bool enableAssertions;     // Enable Java assertions
        bool verboseOutput;        // Verbose output for debugging
        int threadStackSize;       // Thread stack size in KB
    };

    KotlinRuntime();
    ~KotlinRuntime();

    /**
     * Initialize JVM and Kotlin runtime (call once at engine startup)
     * @param config JVM configuration options
     * @return true if initialization successful
     */
    bool Initialize(const JVMConfig& config = JVMConfig());

    /**
     * Shutdown JVM and Kotlin runtime (call at engine shutdown)
     */
    void Shutdown();

    /**
     * Check if runtime is initialized
     */
    bool IsInitialized() const { return initialized; }

    // Class and classpath management
    /**
     * Add a classpath for loading Kotlin classes
     * @param path Directory or JAR file path
     * @return true if added successfully
     */
    bool AddClassPath(const std::string& path);

    /**
     * Load a Kotlin class by name
     * @param className Fully qualified class name (e.g., "gameplay.PlayerController")
     * @return true if loaded successfully, false if already loaded or error
     */
    bool LoadClass(const std::string& className);

    /**
     * Check if a class is loaded
     * @param className Fully qualified class name
     * @return true if class is loaded
     */
    bool IsClassLoaded(const std::string& className) const;

    /**
     * Unload a Kotlin class
     * @param className Fully qualified class name
     */
    void UnloadClass(const std::string& className);

    // Object creation and management
    /**
     * Create an instance of a Kotlin class
     * @param className Fully qualified class name
     * @param constructorArgs Arguments for constructor
     * @return KotlinObject with instance, or null if failed
     */
    KotlinObject CreateInstance(const std::string& className, 
                                 const std::vector<std::any>& constructorArgs = {});

    /**
     * Delete a Kotlin object instance
     * @param obj The object to delete
     */
    void DeleteInstance(KotlinObject& obj);

    // Method invocation
    /**
     * Call a static method on a Kotlin class
     * @param className Fully qualified class name
     * @param methodName Name of the method
     * @param args Arguments to pass (or empty for no args)
     * @return Return value from method
     */
    std::any CallStaticMethod(const std::string& className, 
                              const std::string& methodName,
                              const std::vector<std::any>& args = {});

    /**
     * Call an instance method on a Kotlin object
     * @param obj The Kotlin object instance
     * @param methodName Name of the method
     * @param args Arguments to pass (or empty for no args)
     * @return Return value from method
     */
    std::any CallMethod(const KotlinObject& obj, 
                        const std::string& methodName,
                        const std::vector<std::any>& args = {});

    /**
     * Call a suspend (async) method and get result via callback
     * Kotlin coroutines are executed on the JVM's thread pool
     * @param obj The Kotlin object instance
     * @param methodName Name of the suspend method
     * @param args Arguments to pass
     * @param callback Called when coroutine completes
     */
    void CallSuspendMethod(const KotlinObject& obj,
                           const std::string& methodName,
                           const std::vector<std::any>& args,
                           std::function<void(std::any)> callback);

    // Property access
    /**
     * Get a field value from a Kotlin object
     * @param obj The Kotlin object instance
     * @param fieldName Name of the field
     * @return Field value
     */
    std::any GetField(const KotlinObject& obj, const std::string& fieldName);

    /**
     * Set a field value on a Kotlin object
     * @param obj The Kotlin object instance
     * @param fieldName Name of the field
     * @param value New field value
     * @return true if successful
     */
    bool SetField(KotlinObject& obj, const std::string& fieldName, const std::any& value);

    /**
     * Get a static field from a Kotlin class
     * @param className Fully qualified class name
     * @param fieldName Name of the field
     * @return Field value
     */
    std::any GetStaticField(const std::string& className, const std::string& fieldName);

    /**
     * Set a static field on a Kotlin class
     * @param className Fully qualified class name
     * @param fieldName Name of the field
     * @param value New field value
     * @return true if successful
     */
    bool SetStaticField(const std::string& className, const std::string& fieldName, const std::any& value);

    // Array support for Kotlin collections
    /**
     * Create a Kotlin IntArray
     * @param data Vector of integers
     * @return JNI int array object
     */
    jobject CreateIntArray(const std::vector<int>& data);

    /**
     * Create a Kotlin FloatArray
     * @param data Vector of floats
     * @return JNI float array object
     */
    jobject CreateFloatArray(const std::vector<float>& data);

    /**
     * Create a Kotlin ObjectArray (for generic objects)
     * @param size Array size
     * @param className Element class name
     * @return JNI object array
     */
    jobject CreateObjectArray(size_t size, const std::string& className);

    // Type conversion utilities
    /**
     * Convert C++ std::any to JVM value
     * Supports: int, float, double, bool, std::string, jobject
     * @param value The C++ value
     * @return JNI jobject or primitive value
     */
    jobject AnyToJavaObject(const std::any& value);

    /**
     * Convert JVM value to C++ std::any
     * @param value The Java object or primitive
     * @param expectedType Expected C++ type
     * @return Converted value
     */
    std::any JavaObjectToAny(jobject value, const std::type_info& expectedType);

    // Exception handling
    /**
     * Check if a JVM exception occurred
     * @return true if exception is pending
     */
    bool HasException() const;

    /**
     * Get last exception message
     * @return Exception message or empty string
     */
    std::string GetLastException();

    /**
     * Clear pending exception
     */
    void ClearException();

    // GC control
    /**
     * Request garbage collection
     */
    void RequestGarbageCollection();

    /**
     * Get JVM memory statistics
     * @return Heap used in bytes
     */
    uint64_t GetHeapUsage() const;

    /**
     * Get the JNI environment (advanced usage)
     * @return JNIEnv pointer for direct JNI calls
     */
    JNIEnv* GetJNIEnv() { return jniEnv; }

    /**
     * Get the JVM instance (advanced usage)
     * @return JavaVM pointer
     */
    JavaVM* GetJVM() { return jvm; }

private:
    JavaVM* jvm;
    JNIEnv* jniEnv;
    bool initialized;
    
    std::vector<std::string> classPaths;
    std::map<std::string, KotlinClass> loadedClasses;
    std::vector<jobject> managedObjects;
    std::string lastError;

    // Helper methods
    jclass FindClass(const std::string& className);
    std::string ConvertClassName(const std::string& className) const;
    void CacheMethodsAndFields(KotlinClass& kotlinClass);
};
