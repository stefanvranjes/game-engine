#pragma once

#include "Light.h"
#include "GameObject.h"
#include "GBuffer.h"
#include "PostProcessing.h"
#include "CubemapShadow.h"
#include "SSAO.h"
#include "SSR.h"
#include "TAA.h"
#include "LightProbe.h"
#include "ReflectionProbe.h"
#include "ParticleSystem.h"
#include "Skybox.h"
#include "CascadedShadowMap.h"
#include "ShadowMap.h"
#include "MaterialLibrary.h"
#include "TextureManager.h"
#include "VolumetricFog.h"
#include "SceneSerializer.h"
#include "Prefab.h"
#include "PlanarReflection.h"
#include "Water.h"
#include "WaterSprayEmitter.h"
#include <memory>
#include <vector>
#include <string>

class Camera;
class Material;

class Renderer {
public:
    static const int MAX_LIGHTS = 32;

    Renderer();
    ~Renderer();

    bool Init();
    void Update(float deltaTime);
    // Impostor System
    std::shared_ptr<Texture> GenerateImpostorTexture(std::shared_ptr<GameObject> obj, int resolution = 512);
    void SetupImpostor(std::shared_ptr<GameObject> obj, float minDistance);
    
    // Shader Management
    void UpdateShaders(); // Hot-Reload
    void Shutdown();
    
    void SetCamera(Camera* camera) { m_Camera = camera; }
    bool CheckCollision(const AABB& bounds);

    // Scene serialization (JSON/Binary formats)
    void SaveScene(const std::string& filename, SceneSerializer::SerializationFormat format = SceneSerializer::SerializationFormat::JSON);
    void LoadScene(const std::string& filename);

    // Prefab management
    std::shared_ptr<Prefab> CreatePrefab(const std::string& prefabName, std::shared_ptr<GameObject> sourceObject = nullptr);
    std::shared_ptr<Prefab> GetPrefab(const std::string& prefabName);
    std::shared_ptr<GameObject> InstantiatePrefab(const std::string& prefabName, const Vec3& position = Vec3(0), const std::string& instanceName = "");
    PrefabManager* GetPrefabManager() { return m_PrefabManager.get(); }
    
    // Scene manipulation for editor
    std::shared_ptr<GameObject> GetRoot() { return m_Root; }
    std::vector<Light>& GetLights() { return m_Lights; }

    TextureManager* GetTextureManager() { return m_TextureManager.get(); }
    MaterialLibrary* GetMaterialLibrary() { return m_MaterialLibrary.get(); }
    ParticleSystem* GetParticleSystem() { return m_ParticleSystem.get(); }
    PostProcessing* GetPostProcessing() { return m_PostProcessing.get(); }
    
    // Debug
    void SetShowCascades(bool show) { m_ShowCascades = show; }
    bool GetShowCascades() const { return m_ShowCascades; }
    
    // Shadow fade
    void SetShadowFadeStart(float distance) { m_ShadowFadeStart = distance; }
    void SetShadowFadeEnd(float distance) { m_ShadowFadeEnd = distance; }
    float GetShadowFadeStart() const { return m_ShadowFadeStart; }
    float GetShadowFadeEnd() const { return m_ShadowFadeEnd; }
    
    // SSAO
    SSAO* GetSSAO() { return m_SSAO.get(); }
    void SetSSAOEnabled(bool enabled) { m_SSAOEnabled = enabled; }
    bool GetSSAOEnabled() const { return m_SSAOEnabled; }
    
    // SSR
    SSR* GetSSR() { return m_SSR.get(); }
    void SetSSREnabled(bool enabled) { m_SSREnabled = enabled; }
    bool GetSSREnabled() const { return m_SSREnabled; }
    
    // TAA
    TAA* GetTAA() { return m_TAA.get(); }
    void SetTAAEnabled(bool enabled) { m_TAAEnabled = enabled; }
    bool GetTAAEnabled() const { return m_TAAEnabled; }
    
    // Volumetric Fog
    VolumetricFog* GetVolumetricFog() { return m_VolumetricFog.get(); }
    void SetVolumetricFogEnabled(bool enabled) { m_VolumetricFogEnabled = enabled; }
    bool GetVolumetricFogEnabled() const { return m_VolumetricFogEnabled; }
    
    // Batched Rendering
    void SetBatchedRenderingEnabled(bool enabled) { m_BatchedRenderingEnabled = enabled; }
    bool GetBatchedRenderingEnabled() const { return m_BatchedRenderingEnabled; }
    
    // Planar Reflections
    PlanarReflection* GetPlanarReflection() { return m_PlanarReflection.get(); }
    void SetPlanarReflectionEnabled(bool enabled) { m_PlanarReflectionEnabled = enabled; }
    bool GetPlanarReflectionEnabled() const { return m_PlanarReflectionEnabled; }
    
    void AddCube(const Transform& transform);
    void AddPyramid(const Transform& transform);
    void AddLODTestObject(const Transform& transform);
    void RemoveObject(size_t index);
    
    void AddLight(const Light& light) { m_Lights.push_back(light); }
    void RemoveLight(size_t index) { 
        if (index < m_Lights.size()) m_Lights.erase(m_Lights.begin() + index); 
    }

    // Light Probes
    void AddLightProbe(const Vec3& position, float radius);
    void BakeLightProbes();
    std::vector<std::unique_ptr<LightProbe>>& GetLightProbes() { return m_LightProbes; }
    
    // Reflection Probes
    void AddReflectionProbe(const Vec3& position, float radius, unsigned int resolution = 256);
    void CaptureReflectionProbes();
    std::vector<std::unique_ptr<ReflectionProbe>>& GetReflectionProbes() { return m_ReflectionProbes; }

private:
    void SetupScene();
    void RenderQuad(); // For fullscreen quad in lighting pass
    void RenderCube(); // For cubemap rendering
    
    // Batched Rendering
    struct RenderItem {
        GameObject* object;
        std::shared_ptr<Mesh> mesh;
        Mat4 worldMatrix;
        float distance; // Squared distance to camera
    };
    void CollectRenderItems(GameObject* obj, std::map<Material*, std::vector<RenderItem>>& batches, Frustum* frustum, bool forceRender);
    void RenderBatched(const std::map<Material*, std::vector<RenderItem>>& batches, Shader* shader, const Mat4& view, const Mat4& projection);
    
    std::vector<RenderItem> m_TransparentItems;
    void SortItems(std::vector<RenderItem>& items, bool frontToBack);

    void RenderTransparentItems(Shader* shader, const Mat4& view, const Mat4& projection);
    
    // Decals
    void RenderDecals(const Mat4& view, const Mat4& projection);
    void CollectDecals(GameObject* obj, std::vector<GameObject*>& decalObjects);
    unsigned int m_UnitCubeVAO = 0;
    unsigned int m_UnitCubeVBO = 0;
    void RenderUnitCube(); // Helper for drawing decal volume

    // IBL
    void InitIBL();
    unsigned int m_EnvCubemap;
    unsigned int m_IrradianceMap;
    unsigned int m_PrefilterMap;
    unsigned int m_BRDFLUT;
    
    std::unique_ptr<Shader> m_EquirectangularToCubemapShader;
    std::unique_ptr<Shader> m_IrradianceShader;
    std::unique_ptr<Shader> m_PrefilterShader;
    std::unique_ptr<Shader> m_BRDFShader;

    std::unique_ptr<Shader> m_Shader;
    std::unique_ptr<Shader> m_DepthShader;
    std::unique_ptr<Shader> m_GeometryShader;
    std::unique_ptr<Shader> m_LightingShader;
    std::unique_ptr<Shader> m_DecalShader; // Decal Shader
    std::unique_ptr<Shader> m_WaterShader; // Water Shader
    unsigned int m_RefractionTexture; // Texture for refraction
    std::shared_ptr<Texture> m_Texture; // Default texture
    Camera* m_Camera;
    std::unique_ptr<Skybox> m_Skybox;
    std::unique_ptr<CascadedShadowMap> m_CSM;
    std::unique_ptr<Shader> m_PointShadowShader;
    std::vector<std::unique_ptr<CubemapShadow>> m_PointShadows;
    std::vector<std::unique_ptr<ShadowMap>> m_SpotShadows;
    std::unique_ptr<TextureManager> m_TextureManager;
    std::unique_ptr<MaterialLibrary> m_MaterialLibrary;
    std::unique_ptr<GBuffer> m_GBuffer;
    std::unique_ptr<PostProcessing> m_PostProcessing;
    std::unique_ptr<SSAO> m_SSAO;
    bool m_SSAOEnabled;
    std::unique_ptr<SSR> m_SSR;
    bool m_SSREnabled;
    std::unique_ptr<TAA> m_TAA;
    bool m_TAAEnabled;
    std::unique_ptr<VolumetricFog> m_VolumetricFog;
    bool m_VolumetricFogEnabled;
    std::unique_ptr<ParticleSystem> m_ParticleSystem;
    bool m_BatchedRenderingEnabled;
    
    // Planar Reflections
    std::unique_ptr<PlanarReflection> m_PlanarReflection;
    bool m_PlanarReflectionEnabled = true;
    void RenderReflectionPass();
    
    // Water Spray Particles
    std::unordered_map<GameObject*, std::unique_ptr<class WaterSprayEmitter>> m_WaterSprayEmitters;
    void UpdateWaterSprayEmitters(float deltaTime);
    
    unsigned int m_QuadVAO, m_QuadVBO;
    unsigned int m_InstanceVBO; // VBO for instance data (model matrices)
    
    std::shared_ptr<GameObject> m_Root;
    std::vector<Light> m_Lights;
    std::vector<std::unique_ptr<LightProbe>> m_LightProbes;
    std::vector<std::unique_ptr<ReflectionProbe>> m_ReflectionProbes;
    
    // Scene serialization and prefabs
    std::unique_ptr<SceneSerializer> m_SceneSerializer;
    std::unique_ptr<PrefabManager> m_PrefabManager;
    
    void BakeProbe(LightProbe* probe);
    void CaptureProbe(ReflectionProbe* probe);
    void RenderSceneForward(Shader* shader);
    void RenderWater(const Mat4& view, const Mat4& projection);

    // Debug
    bool m_ShowCascades;
    
    // Shadow fade
    float m_ShadowFadeStart;
    float m_ShadowFadeEnd;

    // CSM Helpers
    std::vector<float> m_CascadeSplits;
    std::vector<Mat4> GetLightSpaceMatrices();
    std::vector<Vec4> GetFrustumCornersWorldSpace(const Mat4& proj, const Mat4& view);
    Mat4 GetLightSpaceMatrix(const float nearPlane, const float farPlane);
    Mat4 GetSpotLightMatrix(const Light& light);
    
    void UpdateSprites(std::shared_ptr<GameObject> node, float deltaTime);
    
    // Test sprite for transition demo
    std::shared_ptr<class Sprite> m_TestSprite;
    float m_TransitionTestTimer;
    
    // Test animated model for blend demo
    std::shared_ptr<GameObject> m_TestAnimatedModel;
    float m_AnimBlendTestTimer;
};
