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
    std::shared_ptr<Texture> texture;

    Material() 
        : ambient(1.0f, 1.0f, 1.0f)
        , diffuse(1.0f, 1.0f, 1.0f)
        , specular(0.5f, 0.5f, 0.5f)
        , shininess(32.0f)
    {
    }

    void Bind(Shader* shader) {
        if (shader) {
            shader->SetVec3("material.ambient", ambient.x, ambient.y, ambient.z);
            shader->SetVec3("material.diffuse", diffuse.x, diffuse.y, diffuse.z);
            shader->SetVec3("material.specular", specular.x, specular.y, specular.z);
            shader->SetFloat("material.shininess", shininess);

            if (texture) {
                texture->Bind(0);
                shader->SetInt("material.texture", 0);
                shader->SetInt("u_HasTexture", 1);
            } else {
                shader->SetInt("u_HasTexture", 0);
            }
        }
    }
};
