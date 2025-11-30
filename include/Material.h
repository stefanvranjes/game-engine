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
    std::shared_ptr<Texture> texture;
    std::shared_ptr<Texture> specularMap;
    std::shared_ptr<Texture> normalMap;
    std::shared_ptr<Texture> roughnessMap;
    std::shared_ptr<Texture> metallicMap;

    Material() 
        : ambient(1.0f, 1.0f, 1.0f)
        , diffuse(1.0f, 1.0f, 1.0f)
        , specular(0.5f, 0.5f, 0.5f)
        , shininess(32.0f)
        , roughness(0.5f)
        , metallic(0.0f)
    {
    }

    void Bind(Shader* shader) {
        if (shader) {
            shader->SetVec3("material.ambient", ambient.x, ambient.y, ambient.z);
            shader->SetVec3("material.diffuse", diffuse.x, diffuse.y, diffuse.z);
            shader->SetVec3("material.specular", specular.x, specular.y, specular.z);
            shader->SetFloat("material.shininess", shininess);
            shader->SetFloat("material.roughness", roughness);
            shader->SetFloat("material.metallic", metallic);

            if (texture) {
                texture->Bind(0);
                shader->SetInt("material.texture", 0);
                shader->SetInt("u_HasTexture", 1);
            } else {
                shader->SetInt("u_HasTexture", 0);
            }

            if (specularMap) {
                specularMap->Bind(1);
                shader->SetInt("material.specularMap", 1);
                shader->SetInt("u_HasSpecularMap", 1);
            } else {
                shader->SetInt("u_HasSpecularMap", 0);
            }

            if (normalMap) {
                normalMap->Bind(3);
                shader->SetInt("material.normalMap", 3);
                shader->SetInt("u_HasNormalMap", 1);
            } else {
                shader->SetInt("u_HasNormalMap", 0);
            }
        }
    }
};
