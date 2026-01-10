#pragma once

#include <thread>
#include <cstddef>

/**
 * @brief Platform-specific thread affinity utilities
 */
class ThreadAffinity {
public:
    /**
     * @brief Set thread affinity to a specific CPU core
     * @param thread Thread to set affinity for
     * @param coreIndex CPU core index (0-based)
     * @return True if successful, false otherwise
     */
    static bool SetAffinity(std::thread& thread, size_t coreIndex);
    
    /**
     * @brief Set current thread affinity to a specific CPU core
     * @param coreIndex CPU core index (0-based)
     * @return True if successful, false otherwise
     */
    static bool SetCurrentThreadAffinity(size_t coreIndex);
    
    /**
     * @brief Get number of available CPU cores
     */
    static size_t GetCoreCount();
    
    /**
     * @brief Check if thread affinity is supported on this platform
     */
    static bool IsSupported();
    
    /**
     * @brief Set thread priority
     * @param thread Thread to set priority for
     * @param priority Priority level (0 = lowest, 2 = highest)
     * @return True if successful, false otherwise
     */
    static bool SetPriority(std::thread& thread, int priority);
};
