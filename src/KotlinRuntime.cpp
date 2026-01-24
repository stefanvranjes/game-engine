#include "KotlinRuntime.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>

KotlinRuntime::KotlinRuntime()
    : jvm(nullptr), jniEnv(nullptr), initialized(false) {}

KotlinRuntime::~KotlinRuntime() {
    if (initialized) {
        Shutdown();
    }
}

bool KotlinRuntime::Initialize(const JVMConfig& config) {
    if (initialized) {
        return true;
    }

    // Build JVM options
    std::vector<JavaVMOption> options;
    std::vector<std::string> optionStrings;

    // Heap size options
    optionStrings.push_back("-Xmx" + std::to_string(config.maxHeapSize) + "m");
    optionStrings.push_back("-Xms" + std::to_string(config.initialHeapSize) + "m");

    // Assertions
    if (config.enableAssertions) {
        optionStrings.push_back("-ea");
    }

    // Stack size
    if (config.threadStackSize > 0) {
        optionStrings.push_back("-Xss" + std::to_string(config.threadStackSize) + "k");
    }

    // Class paths
    std::string classPathArg = "-Djava.class.path=.";
    for (const auto& path : config.classPaths) {
        classPathArg += ":" + path;
    }
    optionStrings.push_back(classPathArg);

    // Verbose options
    if (config.verboseOutput) {
        optionStrings.push_back("-verbose:class");
    }

    // Convert strings to JavaVMOption
    for (auto& optStr : optionStrings) {
        JavaVMOption opt;
        opt.optionString = const_cast<char*>(optStr.c_str());
        opt.extraInfo = nullptr;
        options.push_back(opt);
    }

    // Create JVM
    JavaVMInitArgs vm_args;
    vm_args.version = JNI_VERSION_10;
    vm_args.nOptions = options.size();
    vm_args.options = options.data();
    vm_args.ignoreUnrecognized = false;

    jint result = JNI_CreateJavaVM(&jvm, (void**)&jniEnv, &vm_args);
    if (result != JNI_OK) {
        lastError = "Failed to create JVM: " + std::to_string(result);
        std::cerr << "KotlinRuntime: " << lastError << std::endl;
        return false;
    }

    classPaths = config.classPaths;
    initialized = true;

    std::cout << "KotlinRuntime: JVM initialized successfully" << std::endl;
    return true;
}

void KotlinRuntime::Shutdown() {
    if (!initialized) return;

    // Clean up managed objects
    for (auto& obj : managedObjects) {
        if (obj != nullptr) {
            jniEnv->DeleteGlobalRef(obj);
        }
    }
    managedObjects.clear();
    loadedClasses.clear();

    // Destroy JVM
    if (jvm) {
        jvm->DestroyJavaVM();
        jvm = nullptr;
    }

    jniEnv = nullptr;
    initialized = false;
    std::cout << "KotlinRuntime: JVM shutdown" << std::endl;
}

bool KotlinRuntime::AddClassPath(const std::string& path) {
    if (!initialized) return false;

    // Add to classpath list
    auto it = std::find(classPaths.begin(), classPaths.end(), path);
    if (it == classPaths.end()) {
        classPaths.push_back(path);
    }

    return true;
}

bool KotlinRuntime::LoadClass(const std::string& className) {
    if (!initialized) return false;

    // Check if already loaded
    if (loadedClasses.find(className) != loadedClasses.end()) {
        return true;
    }

    // Convert class name format (e.g., "gameplay.PlayerController" -> "gameplay/PlayerController")
    std::string javaClassName = ConvertClassName(className);

    jclass cls = FindClass(javaClassName);
    if (cls == nullptr) {
        lastError = "Failed to load class: " + className;
        ClearException();
        return false;
    }

    KotlinClass kotlinClass;
    kotlinClass.className = className;
    kotlinClass.javaClass = (jclass)jniEnv->NewGlobalRef(cls);
    kotlinClass.isLoaded = true;

    // Cache methods and fields
    CacheMethodsAndFields(kotlinClass);

    loadedClasses[className] = kotlinClass;

    std::cout << "KotlinRuntime: Loaded class " << className << std::endl;
    return true;
}

bool KotlinRuntime::IsClassLoaded(const std::string& className) const {
    return loadedClasses.find(className) != loadedClasses.end();
}

void KotlinRuntime::UnloadClass(const std::string& className) {
    auto it = loadedClasses.find(className);
    if (it != loadedClasses.end()) {
        if (it->second.javaClass != nullptr) {
            jniEnv->DeleteGlobalRef(it->second.javaClass);
        }
        loadedClasses.erase(it);
    }
}

KotlinRuntime::KotlinObject KotlinRuntime::CreateInstance(const std::string& className,
                                                          const std::vector<std::any>& constructorArgs) {
    KotlinObject obj{"", nullptr, false};

    if (!initialized) return obj;

    // Load class if not already loaded
    if (!IsClassLoaded(className)) {
        if (!LoadClass(className)) {
            return obj;
        }
    }

    auto classIt = loadedClasses.find(className);
    if (classIt == loadedClasses.end()) {
        return obj;
    }

    jclass cls = classIt->second.javaClass;

    // Find constructor
    jmethodID constructor = jniEnv->GetMethodID(cls, "<init>", "()V");
    if (constructor == nullptr) {
        ClearException();
        // Try finding other constructors based on args
        // For now, use no-arg constructor
        constructor = jniEnv->GetMethodID(cls, "<init>", "()V");
        if (constructor == nullptr) {
            lastError = "No suitable constructor found for " + className;
            ClearException();
            return obj;
        }
    }

    // Create instance
    jobject instance = jniEnv->NewObject(cls, constructor);
    if (instance == nullptr) {
        lastError = "Failed to create instance of " + className;
        ClearException();
        return obj;
    }

    // Convert to global reference for long-term storage
    jobject globalRef = jniEnv->NewGlobalRef(instance);
    jniEnv->DeleteLocalRef(instance);

    obj.className = className;
    obj.instance = globalRef;
    obj.isValid = true;

    managedObjects.push_back(globalRef);
    return obj;
}

void KotlinRuntime::DeleteInstance(KotlinObject& obj) {
    if (obj.instance != nullptr && initialized) {
        jniEnv->DeleteGlobalRef(obj.instance);
        obj.instance = nullptr;
        obj.isValid = false;

        auto it = std::find(managedObjects.begin(), managedObjects.end(), obj.instance);
        if (it != managedObjects.end()) {
            managedObjects.erase(it);
        }
    }
}

std::any KotlinRuntime::CallStaticMethod(const std::string& className,
                                        const std::string& methodName,
                                        const std::vector<std::any>& args) {
    std::any result;

    if (!initialized) {
        lastError = "KotlinRuntime not initialized";
        return result;
    }

    // Load class if needed
    if (!IsClassLoaded(className)) {
        if (!LoadClass(className)) {
            return result;
        }
    }

    auto classIt = loadedClasses.find(className);
    if (classIt == loadedClasses.end()) {
        lastError = "Class not found: " + className;
        return result;
    }

    jclass cls = classIt->second.javaClass;

    // Find method - for simplicity, assume no arguments and int return
    // In production, would need signature parsing based on argument types
    jmethodID method = jniEnv->GetStaticMethodID(cls, methodName.c_str(), "()I");
    if (method == nullptr) {
        // Try other signatures
        method = jniEnv->GetStaticMethodID(cls, methodName.c_str(), "()V");
        if (method == nullptr) {
            method = jniEnv->GetStaticMethodID(cls, methodName.c_str(), "()Ljava/lang/Object;");
            ClearException();
        } else {
            ClearException();
        }
    } else {
        ClearException();
    }

    if (method == nullptr) {
        lastError = "Method not found: " + className + "." + methodName;
        return result;
    }

    // Call method
    jint intResult = jniEnv->CallStaticIntMethod(cls, method);
    if (HasException()) {
        lastError = "Exception during method call: " + GetLastException();
        return result;
    }

    return std::any(intResult);
}

std::any KotlinRuntime::CallMethod(const KotlinObject& obj,
                                  const std::string& methodName,
                                  const std::vector<std::any>& args) {
    std::any result;

    if (!initialized || !obj.isValid) {
        lastError = "Invalid Kotlin object or runtime not initialized";
        return result;
    }

    auto classIt = loadedClasses.find(obj.className);
    if (classIt == loadedClasses.end()) {
        lastError = "Class not found: " + obj.className;
        return result;
    }

    jclass cls = classIt->second.javaClass;

    // Find method
    jmethodID method = jniEnv->GetMethodID(cls, methodName.c_str(), "()I");
    if (method == nullptr) {
        method = jniEnv->GetMethodID(cls, methodName.c_str(), "()V");
        ClearException();
    } else {
        ClearException();
    }

    if (method == nullptr) {
        lastError = "Method not found: " + obj.className + "." + methodName;
        return result;
    }

    // Call method
    jint intResult = jniEnv->CallIntMethod(obj.instance, method);
    if (HasException()) {
        lastError = "Exception during method call: " + GetLastException();
        return result;
    }

    return std::any(intResult);
}

void KotlinRuntime::CallSuspendMethod(const KotlinObject& obj,
                                     const std::string& methodName,
                                     const std::vector<std::any>& args,
                                     std::function<void(std::any)> callback) {
    // Placeholder for suspend function support
    // Would require integration with Kotlin coroutine framework
    if (callback) {
        callback(std::any());
    }
}

std::any KotlinRuntime::GetField(const KotlinObject& obj, const std::string& fieldName) {
    std::any result;

    if (!initialized || !obj.isValid) {
        return result;
    }

    auto classIt = loadedClasses.find(obj.className);
    if (classIt == loadedClasses.end()) {
        return result;
    }

    jclass cls = classIt->second.javaClass;
    jfieldID field = jniEnv->GetFieldID(cls, fieldName.c_str(), "I");
    if (field == nullptr) {
        ClearException();
        return result;
    }

    jint value = jniEnv->GetIntField(obj.instance, field);
    return std::any(value);
}

bool KotlinRuntime::SetField(KotlinObject& obj, const std::string& fieldName, const std::any& value) {
    if (!initialized || !obj.isValid) {
        return false;
    }

    auto classIt = loadedClasses.find(obj.className);
    if (classIt == loadedClasses.end()) {
        return false;
    }

    jclass cls = classIt->second.javaClass;
    jfieldID field = jniEnv->GetFieldID(cls, fieldName.c_str(), "I");
    if (field == nullptr) {
        ClearException();
        return false;
    }

    try {
        int intVal = std::any_cast<int>(value);
        jniEnv->SetIntField(obj.instance, field, intVal);
        return true;
    } catch (...) {
        return false;
    }
}

std::any KotlinRuntime::GetStaticField(const std::string& className, const std::string& fieldName) {
    std::any result;

    if (!IsClassLoaded(className)) {
        return result;
    }

    auto classIt = loadedClasses.find(className);
    if (classIt == loadedClasses.end()) {
        return result;
    }

    jclass cls = classIt->second.javaClass;
    jfieldID field = jniEnv->GetStaticFieldID(cls, fieldName.c_str(), "I");
    if (field == nullptr) {
        ClearException();
        return result;
    }

    jint value = jniEnv->GetStaticIntField(cls, field);
    return std::any(value);
}

bool KotlinRuntime::SetStaticField(const std::string& className, const std::string& fieldName, const std::any& value) {
    if (!IsClassLoaded(className)) {
        return false;
    }

    auto classIt = loadedClasses.find(className);
    if (classIt == loadedClasses.end()) {
        return false;
    }

    jclass cls = classIt->second.javaClass;
    jfieldID field = jniEnv->GetStaticFieldID(cls, fieldName.c_str(), "I");
    if (field == nullptr) {
        ClearException();
        return false;
    }

    try {
        int intVal = std::any_cast<int>(value);
        jniEnv->SetStaticIntField(cls, field, intVal);
        return true;
    } catch (...) {
        return false;
    }
}

jobject KotlinRuntime::CreateIntArray(const std::vector<int>& data) {
    if (!initialized) return nullptr;

    jintArray arr = jniEnv->NewIntArray(data.size());
    if (arr == nullptr) return nullptr;

    jniEnv->SetIntArrayRegion(arr, 0, data.size(), data.data());
    return arr;
}

jobject KotlinRuntime::CreateFloatArray(const std::vector<float>& data) {
    if (!initialized) return nullptr;

    jfloatArray arr = jniEnv->NewFloatArray(data.size());
    if (arr == nullptr) return nullptr;

    jniEnv->SetFloatArrayRegion(arr, 0, data.size(), data.data());
    return arr;
}

jobject KotlinRuntime::CreateObjectArray(size_t size, const std::string& className) {
    if (!initialized) return nullptr;

    std::string javaClassName = ConvertClassName(className);
    jclass cls = FindClass(javaClassName);
    if (cls == nullptr) {
        ClearException();
        return nullptr;
    }

    jobjectArray arr = jniEnv->NewObjectArray(size, cls, nullptr);
    return arr;
}

jobject KotlinRuntime::AnyToJavaObject(const std::any& value) {
    if (!value.has_value()) return nullptr;

    try {
        if (value.type() == typeid(int)) {
            int val = std::any_cast<int>(value);
            jclass intClass = jniEnv->FindClass("java/lang/Integer");
            jmethodID constructor = jniEnv->GetMethodID(intClass, "<init>", "(I)V");
            return jniEnv->NewObject(intClass, constructor, val);
        } else if (value.type() == typeid(float)) {
            float val = std::any_cast<float>(value);
            jclass floatClass = jniEnv->FindClass("java/lang/Float");
            jmethodID constructor = jniEnv->GetMethodID(floatClass, "<init>", "(F)V");
            return jniEnv->NewObject(floatClass, constructor, val);
        } else if (value.type() == typeid(double)) {
            double val = std::any_cast<double>(value);
            jclass doubleClass = jniEnv->FindClass("java/lang/Double");
            jmethodID constructor = jniEnv->GetMethodID(doubleClass, "<init>", "(D)V");
            return jniEnv->NewObject(doubleClass, constructor, val);
        } else if (value.type() == typeid(bool)) {
            bool val = std::any_cast<bool>(value);
            jclass boolClass = jniEnv->FindClass("java/lang/Boolean");
            jmethodID constructor = jniEnv->GetMethodID(boolClass, "<init>", "(Z)V");
            return jniEnv->NewObject(boolClass, constructor, val);
        } else if (value.type() == typeid(std::string)) {
            const std::string& str = std::any_cast<const std::string&>(value);
            return (jobject)jniEnv->NewStringUTF(str.c_str());
        }
    } catch (...) {
    }

    return nullptr;
}

std::any KotlinRuntime::JavaObjectToAny(jobject value, const std::type_info& expectedType) {
    std::any result;

    if (value == nullptr) return result;

    if (expectedType == typeid(int)) {
        jclass intClass = jniEnv->FindClass("java/lang/Integer");
        jmethodID intValue = jniEnv->GetMethodID(intClass, "intValue", "()I");
        jint val = jniEnv->CallIntMethod(value, intValue);
        result = std::any(val);
    } else if (expectedType == typeid(float)) {
        jclass floatClass = jniEnv->FindClass("java/lang/Float");
        jmethodID floatValue = jniEnv->GetMethodID(floatClass, "floatValue", "()F");
        jfloat val = jniEnv->CallFloatMethod(value, floatValue);
        result = std::any(val);
    } else if (expectedType == typeid(std::string)) {
        const char* str = jniEnv->GetStringUTFChars((jstring)value, nullptr);
        result = std::any(std::string(str));
        jniEnv->ReleaseStringUTFChars((jstring)value, str);
    }

    return result;
}

bool KotlinRuntime::HasException() const {
    if (!initialized) return false;
    return jniEnv->ExceptionCheck() == JNI_TRUE;
}

std::string KotlinRuntime::GetLastException() {
    if (!HasException()) return "";

    jthrowable exception = jniEnv->ExceptionOccurred();
    if (exception == nullptr) return "";

    jclass exClass = jniEnv->GetObjectClass(exception);
    jmethodID toString = jniEnv->GetMethodID(exClass, "toString", "()Ljava/lang/String;");
    jstring str = (jstring)jniEnv->CallObjectMethod(exception, toString);

    const char* exStr = jniEnv->GetStringUTFChars(str, nullptr);
    std::string result(exStr);
    jniEnv->ReleaseStringUTFChars(str, exStr);
    jniEnv->DeleteLocalRef(exception);

    return result;
}

void KotlinRuntime::ClearException() {
    if (jniEnv && jniEnv->ExceptionCheck()) {
        jniEnv->ExceptionClear();
    }
}

void KotlinRuntime::RequestGarbageCollection() {
    if (!initialized) return;

    jclass runtimeClass = jniEnv->FindClass("java/lang/Runtime");
    jmethodID getRuntime = jniEnv->GetStaticMethodID(runtimeClass, "getRuntime", "()Ljava/lang/Runtime;");
    jobject runtime = jniEnv->CallStaticObjectMethod(runtimeClass, getRuntime);

    jmethodID gc = jniEnv->GetMethodID(runtimeClass, "gc", "()V");
    jniEnv->CallVoidMethod(runtime, gc);

    jniEnv->DeleteLocalRef(runtime);
    jniEnv->DeleteLocalRef(runtimeClass);
}

uint64_t KotlinRuntime::GetHeapUsage() const {
    if (!initialized) return 0;

    // This is a simplified version; actual implementation would use MemoryMXBean
    // For now, return a placeholder
    return 0;
}

jclass KotlinRuntime::FindClass(const std::string& className) {
    if (!initialized) return nullptr;

    jclass cls = jniEnv->FindClass(className.c_str());
    return cls;
}

std::string KotlinRuntime::ConvertClassName(const std::string& className) const {
    // Convert "package.ClassName" to "package/ClassName"
    std::string result = className;
    std::replace(result.begin(), result.end(), '.', '/');
    return result;
}

void KotlinRuntime::CacheMethodsAndFields(KotlinClass& kotlinClass) {
    if (kotlinClass.javaClass == nullptr) return;

    // Get all methods
    jmethodID methods = jniEnv->GetMethodID(kotlinClass.javaClass, "getClass", "()Ljava/lang/Class;");
    // More detailed caching would go here
}
