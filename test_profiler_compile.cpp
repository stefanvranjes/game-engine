#include "Profiler.h"
#include "TelemetryServer.h"

int main() {
    // Test instantiation
    auto& profiler = Profiler::Instance();
    auto& gpu_profiler = GPUProfiler::Instance();
    auto& monitor = PerformanceMonitor::Instance();
    auto& remote = RemoteProfiler::Instance();
    
    // Test basic operations
    profiler.BeginFrame();
    profiler.EndFrame();
    
    // Test scoped profiling
    {
        SCOPED_PROFILE("TestScope");
    }
    
    // Get some stats
    double avg_time = profiler.GetAverageFrameTime();
    
    // Test server
    TelemetryServer server(8080);
    
    return 0;
}
