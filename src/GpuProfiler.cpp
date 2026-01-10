#include "GpuProfiler.h"

#ifdef HAS_NVTX

GpuProfiler::ScopedRange::ScopedRange(const char* name, uint32_t color) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxRangePushEx(&eventAttrib);
}

GpuProfiler::ScopedRange::~ScopedRange() {
    nvtxRangePop();
}

void GpuProfiler::Mark(const char* name) {
    nvtxMarkA(name);
}

void GpuProfiler::SetCategory(Category cat) {
    // Categories can be used to organize markers in the profiler
    // For now, this is a placeholder for future category-based filtering
    (void)cat; // Suppress unused parameter warning
}

#else

// No-op implementations when NVTX is not available
GpuProfiler::ScopedRange::ScopedRange(const char*, uint32_t) {}
GpuProfiler::ScopedRange::~ScopedRange() {}
void GpuProfiler::Mark(const char*) {}
void GpuProfiler::SetCategory(Category) {}

#endif
