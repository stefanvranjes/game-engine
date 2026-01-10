#pragma once

#include <cstdint>

#ifdef HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

/**
 * @brief GPU profiling utility using NVIDIA NVTX markers
 * 
 * Provides RAII-style profiling ranges and event markers for use with
 * NVIDIA Nsight Systems and Nsight Compute profiling tools.
 * 
 * When HAS_NVTX is not defined, all profiling calls compile to no-ops.
 */
class GpuProfiler {
public:
    /**
     * @brief RAII-style profiling range
     * 
     * Automatically pushes an NVTX range on construction and pops it on destruction.
     * Use GPU_PROFILE_SCOPE macro for convenience.
     */
    class ScopedRange {
    public:
        ScopedRange(const char* name, uint32_t color = 0xFF00FF00);
        ~ScopedRange();
        
    private:
        // Non-copyable
        ScopedRange(const ScopedRange&) = delete;
        ScopedRange& operator=(const ScopedRange&) = delete;
    };
    
    /**
     * @brief Mark a single profiling event
     * @param name Event name to display in profiler
     */
    static void Mark(const char* name);
    
    /**
     * @brief Profiling categories for organization
     */
    enum Category {
        PHYSICS_SIMULATION,
        SOFT_BODY_UPDATE,
        COLLISION_DETECTION,
        LOD_MANAGEMENT,
        TEAR_SYSTEM
    };
    
    /**
     * @brief Set the current profiling category
     * @param cat Category to use for subsequent markers
     */
    static void SetCategory(Category cat);
    
    // Predefined colors for different operation types
    static constexpr uint32_t COLOR_PHYSICS = 0xFF00FF00;      // Green
    static constexpr uint32_t COLOR_SOFT_BODY = 0xFF0000FF;    // Blue
    static constexpr uint32_t COLOR_COLLISION = 0xFFFFFF00;    // Yellow
    static constexpr uint32_t COLOR_LOD = 0xFFFF8800;          // Orange
    static constexpr uint32_t COLOR_TEAR = 0xFFFF0000;         // Red
};

// Convenience macros for profiling
#ifdef HAS_NVTX
    #define GPU_PROFILE_SCOPE(name) GpuProfiler::ScopedRange _gpu_prof_##__LINE__(name)
    #define GPU_PROFILE_SCOPE_COLOR(name, color) GpuProfiler::ScopedRange _gpu_prof_##__LINE__(name, color)
    #define GPU_PROFILE_MARK(name) GpuProfiler::Mark(name)
#else
    #define GPU_PROFILE_SCOPE(name)
    #define GPU_PROFILE_SCOPE_COLOR(name, color)
    #define GPU_PROFILE_MARK(name)
#endif
