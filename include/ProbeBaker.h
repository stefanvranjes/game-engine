#pragma once

#include <vector>
#include <atomic>
#include <memory>
#include <glm/glm.hpp>
#include "ProbeGrid.h"

class GameObject;
class Light;

/**
 * @class ProbeBaker
 * @brief Handles offline baking of light probes using raytracing
 * 
 * Samples indirect lighting at probe positions and encodes the results
 * into spherical harmonics for efficient runtime evaluation.
 */
class ProbeBaker {
public:
    struct BakeSettings {
        int samplesPerProbe = 512;      // Number of rays per probe
        float maxRayDistance = 100.0f;  // Maximum ray distance
        int numBounces = 2;             // Number of indirect bounces
        bool useGPU = true;             // Use GPU raytracing (if available)
        bool multiGPU = false;          // Use multiple GPUs (experimental)
        bool showProgress = true;       // Show progress bar
        int probesPerFrame = 64;        // Probes to bake per frame in Async/Progressive mode
        bool backgroundThread = true;   // Run CPU baking in background thread
        
        BakeSettings() = default;
    };
    
    enum class BakingState {
        Idle,
        Baking,
        Paused,
        Completed
    };

    struct RayHit {
        bool hit;
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec3 albedo;
        float distance;
        GameObject* object;
        
        RayHit() : hit(false), distance(FLT_MAX), object(nullptr) {}
    };

    ProbeBaker();
    ~ProbeBaker();

    // Initialization
    void InitializeGPUResources();
    void CleanupGPUResources();

    // Baking (Blocking)
    void BakeProbes(ProbeGrid* grid, const std::vector<GameObject*>& scene,
                    const std::vector<Light>& lights, const BakeSettings& settings);

    // Progressive Baking (Async)
    void StartBakingAsync(ProbeGrid* grid, const std::vector<GameObject*>& scene,
                          const std::vector<Light>& lights, const BakeSettings& settings);
    void UpdateBaking(); // Call this every frame
    void StopBaking();
    BakingState GetBakingState() const { return m_State; }
    int GetCurrentProbeIndex() const { return m_CurrentProbeIndex; }
    int GetTotalProbes() const { return m_TotalProbes; }
    void BakeProbe(int probeIndex, ProbeGrid* grid, const std::vector<GameObject*>& scene,
                   const std::vector<Light>& lights);

    // Progress tracking
    float GetProgress() const { return m_Progress.load(); }
    bool IsBaking() const { return m_IsBaking.load(); }
    void Cancel() { m_IsBaking = false; }

private:
    // Raytracing helpers
    struct SceneSnapshot {
        struct MeshData {
            std::vector<glm::vec3> vertices;
            std::vector<glm::vec3> normals;
            std::vector<uint32_t> indices;
            glm::vec3 albedo;
            glm::mat4 worldMatrix;
            glm::mat4 invWorldMatrix;
        };
        std::vector<MeshData> meshes;
        std::vector<Light> lights;
    };
    SceneSnapshot CaptureScene(const std::vector<GameObject*>& scene, const std::vector<Light>& lights);
    void BakeProbe(int probeIndex, ProbeGrid* grid, const SceneSnapshot& sceneSnapshot);
    bool Raytrace(const glm::vec3& origin, const glm::vec3& direction, const SceneSnapshot& scene, RayHit& hit) const;
    glm::vec3 ComputeDirectLighting(const RayHit& hit, const SceneSnapshot& scene) const;
    bool RayTriangleIntersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir,
                              const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                              float& t, float& u, float& v) const;

    // Lighting computation
    glm::vec3 TraceIndirectBounce(const RayHit& hit, const std::vector<GameObject*>& scene,
                                  const std::vector<Light>& lights, int bounceDepth);
    glm::vec3 GetSkyColor(const glm::vec3& direction) const;

    // Sampling
    glm::vec3 GenerateHemisphereSample(int index, int totalSamples, const glm::vec3& normal) const;
    glm::vec3 GenerateUniformHemisphereSample(float u, float v) const;

    // Spherical harmonics encoding
    void EncodeToSphericalHarmonics(const std::vector<glm::vec3>& samples,
                                    const std::vector<glm::vec3>& directions,
                                    float shCoeffs[27]);
    float EvaluateSHBasis(int l, int m, const glm::vec3& direction) const;

    // Settings
    BakeSettings m_Settings;

    // Internal state
    std::atomic<float> m_Progress;
    std::atomic<bool> m_IsBaking;
    BakingState m_State = BakingState::Idle;
    std::thread m_BackgroundThread; // Main CPU background thread
    int m_CurrentProbeIndex = 0;
    int m_TotalProbes = 0;
    
    // Cached pointers for async update
    ProbeGrid* m_CurrentGrid = nullptr;
    const std::vector<GameObject*>* m_CurrentScene = nullptr;
    const std::vector<Light>* m_CurrentLights = nullptr;
    
    // Raytracing helpers (simple gradient)
    glm::vec3 m_SkyColorTop;
    glm::vec3 m_SkyColorHorizon;

    // GPU Resources
    struct GPUBakeResources {
        unsigned int probePositionSSBO;
        unsigned int sceneVertexSSBO;
        unsigned int sceneIndexSSBO;
        unsigned int sceneNormalSSBO;
        unsigned int sceneMaterialSSBO;
        unsigned int probeDataSSBO;
        unsigned int bvhSSBO;
        unsigned int primIndexSSBO;
        
        class Shader* bakeShader;
    };
    
    GPUBakeResources m_GPUResources;
    bool m_GPUInitialized = false;

    // GPU Helper methods
    void BakeProbesGPUChunk(ProbeGrid* grid, int startProbe, int count);
    void BakeProbesGPU(ProbeGrid* grid, const std::vector<GameObject*>& scene, const std::vector<Light>& lights);
    void UploadSceneToGPU(const std::vector<GameObject*>& scene);
    void UploadProbesToGPU(ProbeGrid* grid);
    void AllocateOutputBuffer(int numProbes);
    void BuildBVHAndUpload(const std::vector<GameObject*>& scene);
    void DownloadProbeData(ProbeGrid* grid);
    void DownloadProbeDataChunk(ProbeGrid* grid, int startProbe, int count);
    
    // BVH Structures for internal build
    struct BVHNode {
        glm::vec3 aabbMin;
        float leftChild; // Encoded as float for layout comp w/ vec4
        glm::vec3 aabbMax;
        float rightChild; // Encoded as float
        int leftIdx;
        int rightIdx;
        int firstPrim;
        int primCount;
    };

    struct GPUBVHNode {
        glm::vec4 aabbMin; // w = leftChild
        glm::vec4 aabbMax; // w = rightChild/firstPrim
        glm::ivec4 data;   // x=left, y=right, z=first, w=count
    };

    // Hardware Ray Tracing Resources (RTX)
    struct RTXResources {
        unsigned int tlas;
        std::vector<unsigned int> blas; // One per unique mesh
        std::vector<uint64_t> blasHandles;
        bool initialized = false;
        
        class Shader* rtShader;
    };
    RTXResources m_RTX;

    // RTX Methods
    void BakeProbesRTX(ProbeGrid* grid, const std::vector<GameObject*>& scene, const std::vector<Light>& lights);
    bool InitializeRTX();
    void BuildBLAS(const std::vector<GameObject*>& scene);
    void BuildTLAS(const std::vector<GameObject*>& scene);
    void CleanupRTX();

    // Multi-GPU Worker Support
    struct GPUWorker {
        std::thread thread;
        std::atomic<bool> finished;
        std::vector<float> results;
        int startProbe;
        int count;
        int gpuIndex;
        
        // Context handle would go here (platform specific)
        
        // Simplified: worker just runs CPU fallback for now or mocks the isolation
        // Real impl requires wglCreateContext
    };
    std::vector<std::unique_ptr<GPUWorker>> m_Workers;
    void StartMultiGPU(ProbeGrid* grid, const std::vector<GameObject*>& scene, const std::vector<Light>& lights);
    void UpdateMultiGPU();
};
