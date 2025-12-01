#pragma once

#include "Shader.h"
#include "Texture.h"
#include "Mesh.h"
#include "Transform.h"
#include "Math/AABB.h"
#include "Math/Vec4.h"
#include "Material.h"
#include "Light.h"
#include "Skybox.h"
#include "CascadedShadowMap.h"
#include "ShadowMap.h"
#include "TextureManager.h"
#include "GameObject.h"
#include "GBuffer.h"
#include "PostProcessing.h"
#include "CubemapShadow.h"
#include "SSAO.h"
#include "LightProbe.h"
#include <memory>
#include <vector>
#include <string>

class Camera;
struct Material;

class Renderer {
public:
    static const int MAX_LIGHTS = 32;

    Renderer();
    ~Renderer();

    bool Init();
    void Render();
    void Shutdown();
    
    void SetCamera(Camera* camera) { m_Camera = camera; }
    bool CheckCollision(const AABB& bounds);

    void SaveScene(const std::string& filename);
    void LoadScene(const std::string& filename);

    // Scene manipulation for editor
    std::shared_ptr<GameObject> GetRoot() { return m_Root; }
    std::vector<Light>& GetLights() { return m_Lights; }

    TextureManager* GetTextureManager() { return m_TextureManager.get(); }
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

private:
    void SetupScene();
    void RenderQuad(); // For fullscreen quad in lighting pass
    void RenderCube(); // For cubemap rendering

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
    std::shared_ptr<Texture> m_Texture; // Default texture
    Camera* m_Camera;
    std::unique_ptr<Skybox> m_Skybox;
    std::unique_ptr<CascadedShadowMap> m_CSM;
    std::unique_ptr<Shader> m_PointShadowShader;
    std::vector<std::unique_ptr<CubemapShadow>> m_PointShadows;
    std::vector<std::unique_ptr<ShadowMap>> m_SpotShadows;
    std::unique_ptr<TextureManager> m_TextureManager;
    std::unique_ptr<GBuffer> m_GBuffer;
    std::unique_ptr<PostProcessing> m_PostProcessing;
    std::unique_ptr<SSAO> m_SSAO;
    bool m_SSAOEnabled;
    
    unsigned int m_QuadVAO, m_QuadVBO;
    
    std::shared_ptr<GameObject> m_Root;
    std::vector<Light> m_Lights;
    std::vector<std::unique_ptr<LightProbe>> m_LightProbes;
    
    void BakeProbe(LightProbe* probe);
    void RenderSceneForward(Shader* shader);

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
};
