#include "DDGIVolume.h"
#include "GameObject.h"
#include "Mesh.h"
#include "Material.h"
#include <vector>
#include <iostream>
#include <random>
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Helper to calculate next power of 2 or alignment
int GetAlignedSize(int size, int align) {
    return (size + align - 1) / align * align;
}

DDGIVolume::DDGIVolume() {}

DDGIVolume::~DDGIVolume() {
    Cleanup();
}

void DDGIVolume::Initialize(const Settings& settings) {
    if (m_Initialized) Cleanup();
    
    m_Settings = settings;
    
    // Initialize Origin to align with StartPosition initially
    // Calculate integer origin from float start pos?
    // Actually, StartPosition is usually user defined absolute.
    // For infinite scrolling, we usually define center relative to camera.
    // If StartPosition is explicit, we assume Streamer takes over control later.
    // Let's assume GridOrigin = 0 mapping to StartPosition.
    m_GridOrigin = glm::ivec3(0);
    m_ResetIndices = glm::ivec3(-1);
    
    CreateTextures();
    CreateBuffers();
    LoadShaders();
    
    m_Initialized = true;
    std::cout << "[DDGI] Initialized Volume: " << settings.gridDimensions.x << "x" << settings.gridDimensions.y << "x" << settings.gridDimensions.z << std::endl;
}

void DDGIVolume::MoveTo(const glm::vec3& position) {
    if (!m_Initialized) return;
    
    // Calculate target origin (bottom-left corner of the box centered on position)
    // Box Size
    glm::vec3 boxSize = glm::vec3(m_Settings.gridDimensions) * m_Settings.probeSpacing;
    glm::vec3 halfBox = boxSize * 0.5f;
    glm::vec3 targetMin = position - halfBox;
    
    // Quantize to Probe Spacing
    glm::ivec3 targetOrigin;
    targetOrigin.x = static_cast<int>(floor(targetMin.x / m_Settings.probeSpacing.x));
    targetOrigin.y = static_cast<int>(floor(targetMin.y / m_Settings.probeSpacing.y));
    targetOrigin.z = static_cast<int>(floor(targetMin.z / m_Settings.probeSpacing.z));
    
    // Delta
    glm::ivec3 delta = targetOrigin - m_GridOrigin;
    
    m_ResetIndices = glm::ivec3(-1);
    
    // If moved, identify wrapped planes
    // Note: We only support shifts of +/- 1 usually per frame, or we must reset multiple.
    // If delta > dim, we reset everything (panic).
    // If delta.x > 0: The "Leading Edge" in +X is now (targetOrigin.x + dim.x - 1).
    // Physical index is that % dim.x.
    
    glm::ivec3 dim = m_Settings.gridDimensions;
    
    if (delta.x != 0) {
        if (abs(delta.x) >= dim.x) {
             // Moved too far, reset all? Or just let it smear?
             // Reset whole volume usually better.
             m_ResetIndices.x = -2; // Special code for 'Reset All' in shader? 
             // Or we just accept artifacts for 1 frame.
        } else {
             // Which plane is new?
             // If +X: The rightmost plane relative to new origin.
             // World Index: newOrigin.x + dim.x - 1
             if (delta.x > 0) m_ResetIndices.x = (targetOrigin.x + dim.x - 1) % dim.x;
             // If -X: The leftmost plane relative to new origin.
             // World Index: newOrigin.x
             else m_ResetIndices.x = (targetOrigin.x) % dim.x; 
             
             // Wrap negative
             if (m_ResetIndices.x < 0) m_ResetIndices.x += dim.x;
        }
    }
    
    if (delta.y != 0) {
        if (abs(delta.y) >= dim.y) {} // ..
        else {
             if (delta.y > 0) m_ResetIndices.y = (targetOrigin.y + dim.y - 1) % dim.y;
             else m_ResetIndices.y = (targetOrigin.y) % dim.y;
             if (m_ResetIndices.y < 0) m_ResetIndices.y += dim.y;
        }
    }
    
    if (delta.z != 0) {
        if (abs(delta.z) >= dim.z) {} // ..
        else {
             if (delta.z > 0) m_ResetIndices.z = (targetOrigin.z + dim.z - 1) % dim.z;
             else m_ResetIndices.z = (targetOrigin.z) % dim.z;
             if (m_ResetIndices.z < 0) m_ResetIndices.z += dim.z;
        }
    }
    
    m_GridOrigin = targetOrigin;
}

void DDGIVolume::Cleanup() {
    if (m_IrradianceTexture) glDeleteTextures(1, &m_IrradianceTexture);
    if (m_DistanceTexture) glDeleteTextures(1, &m_DistanceTexture);
    if (m_ProbeDataSSBO) glDeleteBuffers(1, &m_ProbeDataSSBO);
    if (m_RayHitSSBO) glDeleteBuffers(1, &m_RayHitSSBO);
    if (m_LightSSBO) glDeleteBuffers(1, &m_LightSSBO);
    
    glDeleteBuffers(1, &m_GPU.sceneVertexSSBO);
    glDeleteBuffers(1, &m_GPU.sceneIndexSSBO);
    glDeleteBuffers(1, &m_GPU.sceneNormalSSBO);
    glDeleteBuffers(1, &m_GPU.sceneMaterialSSBO);
    glDeleteBuffers(1, &m_GPU.bvhSSBO);
    glDeleteBuffers(1, &m_GPU.primIndexSSBO);
    
    if (m_RaytraceShader) delete m_RaytraceShader;
    if (m_UpdateShader) delete m_UpdateShader;
    if (m_BorderShader) delete m_BorderShader;
    
    m_Initialized = false;
}

void DDGIVolume::CreateTextures() {
    int numProbes = m_Settings.gridDimensions.x * m_Settings.gridDimensions.y * m_Settings.gridDimensions.z;
    int totalProbes = numProbes;
    int atlasCols = (int)ceil(sqrt((double)totalProbes));
    int atlasRows = (int)ceil((double)totalProbes / atlasCols);
    
    int irrBlockSize = m_Settings.irradianceRes + 2; 
    int distBlockSize = m_Settings.distanceRes + 2;
    int irrWidth = atlasCols * irrBlockSize;
    int irrHeight = atlasRows * irrBlockSize;
    int distWidth = atlasCols * distBlockSize;
    int distHeight = atlasRows * distBlockSize;
    
    glGenTextures(1, &m_IrradianceTexture);
    glBindTexture(GL_TEXTURE_2D, m_IrradianceTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB10_A2, irrWidth, irrHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glGenTextures(1, &m_DistanceTexture);
    glBindTexture(GL_TEXTURE_2D, m_DistanceTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, distWidth, distHeight, 0, GL_RG, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void DDGIVolume::CreateBuffers() {
    int numProbes = m_Settings.gridDimensions.x * m_Settings.gridDimensions.y * m_Settings.gridDimensions.z;
    size_t hitBufferSize = numProbes * m_Settings.raysPerProbe * sizeof(glm::vec4);
    glGenBuffers(1, &m_RayHitSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_RayHitSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, hitBufferSize, nullptr, GL_DYNAMIC_DRAW);
    
    glGenBuffers(1, &m_LightSSBO);

    glGenBuffers(1, &m_GPU.sceneVertexSSBO);
    glGenBuffers(1, &m_GPU.sceneIndexSSBO);
    glGenBuffers(1, &m_GPU.sceneNormalSSBO);
    glGenBuffers(1, &m_GPU.sceneMaterialSSBO);
    glGenBuffers(1, &m_GPU.bvhSSBO);
    glGenBuffers(1, &m_GPU.primIndexSSBO);
}

void DDGIVolume::LoadShaders() {
    m_RaytraceShader = new Shader("shaders/ddgi_raytrace.comp");
    m_UpdateShader = new Shader("shaders/ddgi_update.comp");
    m_BorderShader = new Shader("shaders/ddgi_border.comp");
}

void DDGIVolume::UploadScene(const std::vector<GameObject*>& scene) {
    // Reusing simplified logic
    struct GPUVertex { glm::vec4 pos; };
    struct GPUNormal { glm::vec4 norm; };
    struct GPUMat { glm::vec4 alb; };
    std::vector<GPUVertex> vertices;
    std::vector<GPUNormal> normals;
    std::vector<GPUMat> materials;
    std::vector<uint32_t> indices;
    uint32_t baseIdx = 0;
    
    for(auto* obj : scene) {
         if (!obj->IsActive()) continue;
         Mesh* mesh = obj->GetComponent<Mesh>();
         if (!mesh) continue;
         Material* mat = obj->GetComponent<Material>();
         glm::vec3 c = mat ? mat->GetDiffuse() : glm::vec3(0.8f);
         
         glm::mat4 M = obj->GetTransform().GetWorldMatrix();
         glm::mat3 N = glm::transpose(glm::inverse(glm::mat3(M)));
         auto rawV = mesh->GetVertices();
         auto rawI = mesh->GetIndices();
         
         for(auto& v : rawV) {
             vertices.push_back({M * glm::vec4(v.position, 1.0f)});
             normals.push_back({glm::vec4(glm::normalize(N * v.normal), 0.0f)});
             materials.push_back({glm::vec4(c, 0.0f)});
         }
         for(auto i : rawI) indices.push_back(baseIdx + i);
         baseIdx += rawV.size();
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_GPU.sceneVertexSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, vertices.size()*sizeof(GPUVertex), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_GPU.sceneNormalSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, normals.size()*sizeof(GPUNormal), normals.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_GPU.sceneIndexSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, indices.size()*sizeof(uint32_t), indices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_GPU.sceneMaterialSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, materials.size()*sizeof(GPUMat), materials.data(), GL_STATIC_DRAW);
}

// === Light Upload Implementation ===

struct alignas(16) GPULight {
    glm::vec4 position;  // w = type (0=Point, 1=Dir, 2=Spot)
    glm::vec4 direction; // w = range
    glm::vec4 color;     // w = intensity
    glm::vec4 params;    // x=spotAngle, y=spotSoftness
};

void DDGIVolume::UploadLights(const std::vector<Light>& lights) {
    if (m_LightSSBO == 0) return;

    std::vector<GPULight> gpuLights;
    gpuLights.reserve(lights.size());
    
    for(const auto& l : lights) {
        GPULight gl;
        // Position & Type
        float typeVal = 0.0f;
        if (l.type == LightType::Directional) typeVal = 1.0f;
        else if (l.type == LightType::Spot) typeVal = 2.0f;
        else if (l.type == LightType::Point) typeVal = 0.0f;
        
        gl.position = glm::vec4(l.position.x, l.position.y, l.position.z, typeVal);
        
        // Direction & Range
        gl.direction = glm::vec4(l.direction.x, l.direction.y, l.direction.z, l.range);
        
        // Color & Intensity
        gl.color = glm::vec4(l.color.x, l.color.y, l.color.z, l.intensity);
        
        // Params
        float cutoff = glm::cos(glm::radians(l.cutOff));
        float outer = glm::cos(glm::radians(l.outerCutOff));
        gl.params = glm::vec4(cutoff, outer, 0.0f, 0.0f);
        
        gpuLights.push_back(gl);
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_LightSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gpuLights.size() * sizeof(GPULight), gpuLights.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    if(m_RaytraceShader) {
        m_RaytraceShader->Use();
        m_RaytraceShader->SetInt("numLights", (int)lights.size());
    }
}

void DDGIVolume::Update(const std::vector<GameObject*>& scene, const std::vector<class Light>& lights, float deltaTime) {
    if (!m_Initialized) return;
    
    static std::mt19937 rng(1337);
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float randAngle = dist(rng) * 3.14159f * 2.0f;
    glm::vec3 axis = glm::normalize(glm::vec3(dist(rng), dist(rng), dist(rng)));
    m_RandomRotation = glm::rotate(glm::mat4(1.0f), randAngle, axis);
    
    UpdateIrradianceTextures(); // If using 3D textures, update them? No, this function is just a hook. 
    // Actually DDGIVolume doesn't own ProbeGrid's textures.
    
    UploadLights(lights);

    DispatchRaytrace(scene);
    DispatchUpdate();
    DispatchBorderFix();
    
    // Reset indices after one frame of clearing
    m_ResetIndices = glm::ivec3(-1);
}

void DDGIVolume::DispatchRaytrace(const std::vector<GameObject*>& scene) {
    m_RaytraceShader->Use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_RayHitSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_GPU.sceneVertexSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_GPU.sceneIndexSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_GPU.sceneNormalSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_GPU.sceneMaterialSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_LightSSBO);
    
    m_RaytraceShader->SetMat4("randomVar", m_RandomRotation);
    m_RaytraceShader->SetVec3("gridStart", m_Settings.startPosition); // Deprecated if toroidal?
    // With Toroidal, 'gridStart' is dynamic. 
    // We pass probeSpacing and gridOrigin.
    // The shader will compute world pos.
    m_RaytraceShader->SetVec3("probeSpacing", m_Settings.probeSpacing);
    m_RaytraceShader->SetIVec3("gridOrigin", m_GridOrigin);
    m_RaytraceShader->SetIVec3("gridDim", m_Settings.gridDimensions);
    m_RaytraceShader->SetInt("raysPerProbe", m_Settings.raysPerProbe);
    m_RaytraceShader->SetFloat("maxDist", m_Settings.maxRayDistance);
    
    int numProbes = m_Settings.gridDimensions.x * m_Settings.gridDimensions.y * m_Settings.gridDimensions.z;
    int totalRays = numProbes * m_Settings.raysPerProbe;
    int groupSize = 64;
    glDispatchCompute((totalRays + groupSize - 1) / groupSize, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void DDGIVolume::DispatchUpdate() {
    m_UpdateShader->Use();
    glBindImageTexture(0, m_IrradianceTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGB10_A2);
    glBindImageTexture(1, m_DistanceTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG16F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_RayHitSSBO);
    
    m_UpdateShader->SetVec3("probeSpacing", m_Settings.probeSpacing);
    m_UpdateShader->SetIVec3("gridOrigin", m_GridOrigin);
    m_UpdateShader->SetIVec3("gridDim", m_Settings.gridDimensions);
    m_UpdateShader->SetInt("raysPerProbe", m_Settings.raysPerProbe);
    m_UpdateShader->SetFloat("hysteresis", m_Settings.hysteresis);
    m_UpdateShader->SetIVec3("resetIndices", m_ResetIndices);
    
    int numProbes = m_Settings.gridDimensions.x * m_Settings.gridDimensions.y * m_Settings.gridDimensions.z;
    glDispatchCompute(numProbes, 1, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
}

void DDGIVolume::DispatchBorderFix() {
    m_BorderShader->Use();
    glBindImageTexture(0, m_IrradianceTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGB10_A2);
    glBindImageTexture(1, m_DistanceTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG16F);
    m_BorderShader->SetIVec3("gridDim", m_Settings.gridDimensions);
    int numProbes = m_Settings.gridDimensions.x * m_Settings.gridDimensions.y * m_Settings.gridDimensions.z;
    glDispatchCompute(numProbes, 1, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
}

void DDGIVolume::BindTextures(int irrUnit, int distUnit, int dataUnit) {
    if(!m_Initialized) return;
    glActiveTexture(GL_TEXTURE0 + irrUnit);
    glBindTexture(GL_TEXTURE_2D, m_IrradianceTexture);
    glActiveTexture(GL_TEXTURE0 + distUnit);
    glBindTexture(GL_TEXTURE_2D, m_DistanceTexture);
}
