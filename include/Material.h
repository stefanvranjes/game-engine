#pragma once

#include "Math/Vec3.h"
#include "Texture.h"
#include "Shader.h"
#include <memory>

class Material {
public:
    enum Property {
        None = 0,
        Ambient = 1 << 0,
        Diffuse = 1 << 1,
        Specular = 1 << 2,
        Shininess = 1 << 3,
        Roughness = 1 << 4,
        Metallic = 1 << 5,
        HeightScale = 1 << 6,
        EmissiveColor = 1 << 7,
        TextureMap = 1 << 8,
        SpecularMap = 1 << 9,
        NormalMap = 1 << 10,
        RoughnessMap = 1 << 11,
        MetallicMap = 1 << 12,
        AOMap = 1 << 13,
        ORMMap = 1 << 14,
        HeightMap = 1 << 15,
        EmissiveMap = 1 << 16
    };

    Material() 
        : m_Ambient(1.0f, 1.0f, 1.0f)
        , m_Diffuse(1.0f, 1.0f, 1.0f)
        , m_Specular(0.5f, 0.5f, 0.5f)
        , m_Shininess(32.0f)
        , m_Roughness(0.5f)
        , m_Metallic(0.0f)
        , m_HeightScale(0.1f)
        , m_EmissiveColor(0.0f, 0.0f, 0.0f)
        , m_Overrides(0)
    {
    }

    // Parent/Child Relationship
    void SetParent(std::shared_ptr<Material> parent) { m_Parent = parent; }
    std::shared_ptr<Material> GetParent() const { return m_Parent; }

    // Getters (Resolve Child -> Parent -> Default)
    Vec3 GetAmbient() const { return (m_Overrides & Ambient) ? m_Ambient : (m_Parent ? m_Parent->GetAmbient() : m_Ambient); }
    Vec3 GetDiffuse() const { return (m_Overrides & Diffuse) ? m_Diffuse : (m_Parent ? m_Parent->GetDiffuse() : m_Diffuse); }
    Vec3 GetSpecular() const { return (m_Overrides & Specular) ? m_Specular : (m_Parent ? m_Parent->GetSpecular() : m_Specular); }
    float GetShininess() const { return (m_Overrides & Shininess) ? m_Shininess : (m_Parent ? m_Parent->GetShininess() : m_Shininess); }
    float GetRoughness() const { return (m_Overrides & Roughness) ? m_Roughness : (m_Parent ? m_Parent->GetRoughness() : m_Roughness); }
    float GetMetallic() const { return (m_Overrides & Metallic) ? m_Metallic : (m_Parent ? m_Parent->GetMetallic() : m_Metallic); }
    float GetHeightScale() const { return (m_Overrides & HeightScale) ? m_HeightScale : (m_Parent ? m_Parent->GetHeightScale() : m_HeightScale); }
    Vec3 GetEmissiveColor() const { return (m_Overrides & EmissiveColor) ? m_EmissiveColor : (m_Parent ? m_Parent->GetEmissiveColor() : m_EmissiveColor); }

    std::shared_ptr<Texture> GetTexture() const { return (m_Overrides & TextureMap) ? m_Texture : (m_Parent ? m_Parent->GetTexture() : nullptr); }
    std::shared_ptr<Texture> GetSpecularMap() const { return (m_Overrides & SpecularMap) ? m_SpecularMap : (m_Parent ? m_Parent->GetSpecularMap() : nullptr); }
    std::shared_ptr<Texture> GetNormalMap() const { return (m_Overrides & NormalMap) ? m_NormalMap : (m_Parent ? m_Parent->GetNormalMap() : nullptr); }
    std::shared_ptr<Texture> GetRoughnessMap() const { return (m_Overrides & RoughnessMap) ? m_RoughnessMap : (m_Parent ? m_Parent->GetRoughnessMap() : nullptr); }
    std::shared_ptr<Texture> GetMetallicMap() const { return (m_Overrides & MetallicMap) ? m_MetallicMap : (m_Parent ? m_Parent->GetMetallicMap() : nullptr); }
    std::shared_ptr<Texture> GetAOMap() const { return (m_Overrides & AOMap) ? m_AOMap : (m_Parent ? m_Parent->GetAOMap() : nullptr); }
    std::shared_ptr<Texture> GetORMMap() const { return (m_Overrides & ORMMap) ? m_ORMMap : (m_Parent ? m_Parent->GetORMMap() : nullptr); }
    std::shared_ptr<Texture> GetHeightMap() const { return (m_Overrides & HeightMap) ? m_HeightMap : (m_Parent ? m_Parent->GetHeightMap() : nullptr); }
    std::shared_ptr<Texture> GetEmissiveMap() const { return (m_Overrides & EmissiveMap) ? m_EmissiveMap : (m_Parent ? m_Parent->GetEmissiveMap() : nullptr); }

    // Setters (Set value and mark override)
    void SetAmbient(const Vec3& v) { m_Ambient = v; m_Overrides |= Ambient; }
    void SetDiffuse(const Vec3& v) { m_Diffuse = v; m_Overrides |= Diffuse; }
    void SetSpecular(const Vec3& v) { m_Specular = v; m_Overrides |= Specular; }
    void SetShininess(float v) { m_Shininess = v; m_Overrides |= Shininess; }
    void SetRoughness(float v) { m_Roughness = v; m_Overrides |= Roughness; }
    void SetMetallic(float v) { m_Metallic = v; m_Overrides |= Metallic; }
    void SetHeightScale(float v) { m_HeightScale = v; m_Overrides |= HeightScale; }
    void SetEmissiveColor(const Vec3& v) { m_EmissiveColor = v; m_Overrides |= EmissiveColor; }

    void SetTexture(std::shared_ptr<Texture> t) { m_Texture = t; m_Overrides |= TextureMap; }
    void SetSpecularMap(std::shared_ptr<Texture> t) { m_SpecularMap = t; m_Overrides |= SpecularMap; }
    void SetNormalMap(std::shared_ptr<Texture> t) { m_NormalMap = t; m_Overrides |= NormalMap; }
    void SetRoughnessMap(std::shared_ptr<Texture> t) { m_RoughnessMap = t; m_Overrides |= RoughnessMap; }
    void SetMetallicMap(std::shared_ptr<Texture> t) { m_MetallicMap = t; m_Overrides |= MetallicMap; }
    void SetAOMap(std::shared_ptr<Texture> t) { m_AOMap = t; m_Overrides |= AOMap; }
    void SetORMMap(std::shared_ptr<Texture> t) { m_ORMMap = t; m_Overrides |= ORMMap; }
    void SetHeightMap(std::shared_ptr<Texture> t) { m_HeightMap = t; m_Overrides |= HeightMap; }
    void SetEmissiveMap(std::shared_ptr<Texture> t) { m_EmissiveMap = t; m_Overrides |= EmissiveMap; }

    // Factory methods for common material presets
    static std::shared_ptr<Material> CreateMetal(float roughness = 0.3f, const Vec3& color = Vec3(0.9f, 0.9f, 0.9f)) {
        auto mat = std::make_shared<Material>();
        mat->SetDiffuse(color);
        mat->SetRoughness(roughness);
        mat->SetMetallic(1.0f);  // Fully metallic
        return mat;
    }

    static std::shared_ptr<Material> CreatePlastic(const Vec3& color = Vec3(0.8f, 0.2f, 0.2f), float roughness = 0.5f) {
        auto mat = std::make_shared<Material>();
        mat->SetDiffuse(color);
        mat->SetRoughness(roughness);
        mat->SetMetallic(0.0f);  // Non-metallic
        return mat;
    }

    static std::shared_ptr<Material> CreateWood(float roughness = 0.8f) {
        auto mat = std::make_shared<Material>();
        mat->SetDiffuse(Vec3(0.6f, 0.4f, 0.2f));  // Brown wood color
        mat->SetRoughness(roughness);
        mat->SetMetallic(0.0f);  // Non-metallic
        return mat;
    }

    static std::shared_ptr<Material> CreateConcrete(float roughness = 0.9f) {
        auto mat = std::make_shared<Material>();
        mat->SetDiffuse(Vec3(0.5f, 0.5f, 0.5f));  // Gray
        mat->SetRoughness(roughness);
        mat->SetMetallic(0.0f);  // Non-metallic
        return mat;
    }

    static std::shared_ptr<Material> CreateGlass(float roughness = 0.0f) {
        auto mat = std::make_shared<Material>();
        mat->SetDiffuse(Vec3(0.95f, 0.95f, 0.95f));  // Clear/white
        mat->SetRoughness(roughness);
        mat->SetMetallic(0.0f);  // Non-metallic
        return mat;
    }


    void Bind(Shader* shader) {
        if (shader) {
            // Resolve properties
            Vec3 ambient = GetAmbient();
            Vec3 diffuse = GetDiffuse();
            Vec3 specular = GetSpecular();
            float shininess = GetShininess();
            float roughness = GetRoughness();
            float metallic = GetMetallic();
            float heightScale = GetHeightScale();
            Vec3 emissiveColor = GetEmissiveColor();
            
            auto texture = GetTexture();
            auto specularMap = GetSpecularMap();
            auto normalMap = GetNormalMap();
            auto roughnessMap = GetRoughnessMap();
            auto metallicMap = GetMetallicMap();
            auto aoMap = GetAOMap();
            auto ormMap = GetORMMap();
            auto heightMap = GetHeightMap();
            auto emissiveMap = GetEmissiveMap();

            shader->SetVec3("material.ambient", ambient.x, ambient.y, ambient.z);
            shader->SetVec3("material.diffuse", diffuse.x, diffuse.y, diffuse.z);
            shader->SetVec3("material.specular", specular.x, specular.y, specular.z);
            shader->SetFloat("material.shininess", shininess);
            shader->SetFloat("material.roughness", roughness);
            shader->SetFloat("material.metallic", metallic);

            // Texture unit 0: Albedo/Diffuse
            if (texture) {
                texture->Bind(0);
                shader->SetInt("material.texture", 0);
                shader->SetInt("u_HasTexture", 1);
            } else {
                shader->SetInt("u_HasTexture", 0);
            }

            // Texture unit 1: Specular (legacy)
            if (specularMap) {
                specularMap->Bind(1);
                shader->SetInt("material.specularMap", 1);
                shader->SetInt("u_HasSpecularMap", 1);
            } else {
                shader->SetInt("u_HasSpecularMap", 0);
            }

            // Texture unit 2: Roughness
            if (roughnessMap) {
                roughnessMap->Bind(2);
                shader->SetInt("material.roughnessMap", 2);
                shader->SetInt("u_HasRoughnessMap", 1);
            } else {
                shader->SetInt("u_HasRoughnessMap", 0);
            }

            // Texture unit 3: Normal
            if (normalMap) {
                normalMap->Bind(3);
                shader->SetInt("material.normalMap", 3);
                shader->SetInt("u_HasNormalMap", 1);
            } else {
                shader->SetInt("u_HasNormalMap", 0);
            }

            // Texture unit 4: Metallic
            if (metallicMap) {
                metallicMap->Bind(4);
                shader->SetInt("material.metallicMap", 4);
                shader->SetInt("u_HasMetallicMap", 1);
            } else {
                shader->SetInt("u_HasMetallicMap", 0);
            }

            // Texture unit 5: Ambient Occlusion
            if (aoMap) {
                aoMap->Bind(5);
                shader->SetInt("material.aoMap", 5);
                shader->SetInt("u_HasAOMap", 1);
            } else {
                shader->SetInt("u_HasAOMap", 0);
            }

            // Texture unit 6: Combined ORM (Occlusion-Roughness-Metallic)
            // If ORM map is present, it takes priority over separate maps
            if (ormMap) {
                ormMap->Bind(6);
                shader->SetInt("material.ormMap", 6);
                shader->SetInt("u_HasORMMap", 1);
            } else {
                shader->SetInt("u_HasORMMap", 0);
            }

            // Texture unit 7: Height map for parallax occlusion mapping
            if (heightMap) {
                heightMap->Bind(7);
                shader->SetInt("material.heightMap", 7);
                shader->SetFloat("material.heightScale", heightScale);
                shader->SetInt("u_HasHeightMap", 1);
            } else {
                shader->SetInt("u_HasHeightMap", 0);
            }

            // Texture unit 8: Emissive map
            shader->SetVec3("material.emissiveColor", emissiveColor.x, emissiveColor.y, emissiveColor.z);
            if (emissiveMap) {
                emissiveMap->Bind(8);
                shader->SetInt("material.emissiveMap", 8);
                shader->SetInt("u_HasEmissiveMap", 1);
            } else {
                shader->SetInt("u_HasEmissiveMap", 0);
            }
        }
    }

private:
    std::shared_ptr<Material> m_Parent;
    unsigned int m_Overrides;

    Vec3 m_Ambient;
    Vec3 m_Diffuse;
    Vec3 m_Specular;
    float m_Shininess;
    float m_Roughness;
    float m_Metallic;
    float m_HeightScale;
    Vec3 m_EmissiveColor;

    std::shared_ptr<Texture> m_Texture;
    std::shared_ptr<Texture> m_SpecularMap;
    std::shared_ptr<Texture> m_NormalMap;
    std::shared_ptr<Texture> m_RoughnessMap;
    std::shared_ptr<Texture> m_MetallicMap;
    std::shared_ptr<Texture> m_AOMap;
    std::shared_ptr<Texture> m_ORMMap;
    std::shared_ptr<Texture> m_HeightMap;
    std::shared_ptr<Texture> m_EmissiveMap;
};

