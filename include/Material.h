#pragma once

#include "Math/Vec3.h"
#include "Texture.h"
#include "Shader.h"
#include <memory>

struct Material {
    Vec3 ambient;
    Vec3 diffuse;
    Vec3 specular;
    float shininess;
    float roughness;
    float metallic;
    float heightScale; // Scale factor for parallax effect (0.0 - 0.2 typical)
    Vec3 emissiveColor; // Emissive color (default black)

    std::shared_ptr<Texture> texture;
    std::shared_ptr<Texture> specularMap;
    std::shared_ptr<Texture> normalMap;
    std::shared_ptr<Texture> roughnessMap;
    std::shared_ptr<Texture> metallicMap;
    std::shared_ptr<Texture> aoMap;  // Ambient Occlusion map
    std::shared_ptr<Texture> ormMap; // Combined Occlusion-Roughness-Metallic (R=AO, G=Roughness, B=Metallic)
    std::shared_ptr<Texture> heightMap; // Height/Displacement map for parallax occlusion mapping
    std::shared_ptr<Texture> emissiveMap; // Emissive map

    Material() 
        : ambient(1.0f, 1.0f, 1.0f)
        , diffuse(1.0f, 1.0f, 1.0f)
        , specular(0.5f, 0.5f, 0.5f)
        , shininess(32.0f)
        , roughness(0.5f)
        , metallic(0.0f)
        , heightScale(0.1f)
        , emissiveColor(0.0f, 0.0f, 0.0f)
    {
    }

    // Factory methods for common material presets
    static std::shared_ptr<Material> CreateMetal(float roughness = 0.3f, const Vec3& color = Vec3(0.9f, 0.9f, 0.9f)) {
        auto mat = std::make_shared<Material>();
        mat->diffuse = color;
        mat->roughness = roughness;
        mat->metallic = 1.0f;  // Fully metallic
        return mat;
    }

    static std::shared_ptr<Material> CreatePlastic(const Vec3& color = Vec3(0.8f, 0.2f, 0.2f), float roughness = 0.5f) {
        auto mat = std::make_shared<Material>();
        mat->diffuse = color;
        mat->roughness = roughness;
        mat->metallic = 0.0f;  // Non-metallic
        return mat;
    }

    static std::shared_ptr<Material> CreateWood(float roughness = 0.8f) {
        auto mat = std::make_shared<Material>();
        mat->diffuse = Vec3(0.6f, 0.4f, 0.2f);  // Brown wood color
        mat->roughness = roughness;
        mat->metallic = 0.0f;  // Non-metallic
        return mat;
    }

    static std::shared_ptr<Material> CreateConcrete(float roughness = 0.9f) {
        auto mat = std::make_shared<Material>();
        mat->diffuse = Vec3(0.5f, 0.5f, 0.5f);  // Gray
        mat->roughness = roughness;
        mat->metallic = 0.0f;  // Non-metallic
        return mat;
    }

    static std::shared_ptr<Material> CreateGlass(float roughness = 0.0f) {
        auto mat = std::make_shared<Material>();
        mat->diffuse = Vec3(0.95f, 0.95f, 0.95f);  // Clear/white
        mat->roughness = roughness;
        mat->metallic = 0.0f;  // Non-metallic
        return mat;
    }


    void Bind(Shader* shader) {
        if (shader) {
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
};

