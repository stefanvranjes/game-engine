#include "ThreadAffinity.h"
#include <iostream>

// Platform-specific includes
#ifdef _WIN32
    #include <windows.h>
#elif defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
#elif defined(__APPLE__)
    #include <pthread.h>
    #include <mach/thread_policy.h>
    #include <mach/thread_act.h>
#endif

bool ThreadAffinity::SetAffinity(std::thread& thread, size_t coreIndex) {
#ifdef _WIN32
    // Windows implementation
    DWORD_PTR mask = 1ULL << coreIndex;
    HANDLE handle = static_cast<HANDLE>(thread.native_handle());
    DWORD_PTR result = SetThreadAffinityMask(handle, mask);
    
    if (result == 0) {
        std::cerr << "Failed to set thread affinity to core " << coreIndex << std::endl;
        return false;
    }
    
    std::cout << "Set thread affinity to core " << coreIndex << std::endl;
    return true;
    
#elif defined(__linux__)
    // Linux implementation
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreIndex, &cpuset);
    
    pthread_t handle = thread.native_handle();
    int result = pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset);
    
    if (result != 0) {
        std::cerr << "Failed to set thread affinity to core " << coreIndex << std::endl;
        return false;
    }
    
    std::cout << "Set thread affinity to core " << coreIndex << std::endl;
    return true;
    
#elif defined(__APPLE__)
    // macOS implementation (thread affinity tags)
    thread_affinity_policy_data_t policy = { static_cast<integer_t>(coreIndex) };
    mach_port_t handle = pthread_mach_thread_np(thread.native_handle());
    
    kern_return_t result = thread_policy_set(
        handle,
        THREAD_AFFINITY_POLICY,
        (thread_policy_t)&policy,
        THREAD_AFFINITY_POLICY_COUNT
    );
    
    if (result != KERN_SUCCESS) {
        std::cerr << "Failed to set thread affinity to core " << coreIndex << std::endl;
        return false;
    }
    
    std::cout << "Set thread affinity tag to " << coreIndex << std::endl;
    return true;
    
#else
    std::cerr << "Thread affinity not supported on this platform" << std::endl;
    return false;
#endif
}

bool ThreadAffinity::SetCurrentThreadAffinity(size_t coreIndex) {
#ifdef _WIN32
    DWORD_PTR mask = 1ULL << coreIndex;
    DWORD_PTR result = SetThreadAffinityMask(GetCurrentThread(), mask);
    return result != 0;
    
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreIndex, &cpuset);
    
    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    return result == 0;
    
#elif defined(__APPLE__)
    thread_affinity_policy_data_t policy = { static_cast<integer_t>(coreIndex) };
    
    kern_return_t result = thread_policy_set(
        mach_thread_self(),
        THREAD_AFFINITY_POLICY,
        (thread_policy_t)&policy,
        THREAD_AFFINITY_POLICY_COUNT
    );
    
    return result == KERN_SUCCESS;
    
#else
    return false;
#endif
}

size_t ThreadAffinity::GetCoreCount() {
    return std::thread::hardware_concurrency();
}

bool ThreadAffinity::IsSupported() {
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

bool ThreadAffinity::SetPriority(std::thread& thread, int priority) {
#ifdef _WIN32
    // Windows priority levels
    int winPriority;
    switch (priority) {
        case 0: winPriority = THREAD_PRIORITY_BELOW_NORMAL; break;
        case 1: winPriority = THREAD_PRIORITY_NORMAL; break;
        case 2: winPriority = THREAD_PRIORITY_ABOVE_NORMAL; break;
        default: winPriority = THREAD_PRIORITY_NORMAL; break;
    }
    
    HANDLE handle = static_cast<HANDLE>(thread.native_handle());
    BOOL result = SetThreadPriority(handle, winPriority);
    return result != 0;
    
#elif defined(__linux__)
    // Linux scheduling priority
    struct sched_param param;
    param.sched_priority = priority;
    
    pthread_t handle = thread.native_handle();
    int result = pthread_setschedparam(handle, SCHED_OTHER, &param);
    return result == 0;
    
#elif defined(__APPLE__)
    // macOS thread priority
    struct sched_param param;
    param.sched_priority = priority;
    
    pthread_t handle = thread.native_handle();
    int result = pthread_setschedparam(handle, SCHED_OTHER, &param);
    return result == 0;
    
#else
    return false;
#endif
}
