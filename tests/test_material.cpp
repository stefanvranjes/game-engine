#include <gtest/gtest.h>
#include "Material.h"
#include "Texture.h"
#include "Math/Vec3.h"

class MaterialTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(MaterialTest, DefaultMaterial) {
    Material mat;
    // Basic property check
    EXPECT_NEAR(mat.GetMetallic(), 0.0f, 1e-5f);
}

TEST_F(MaterialTest, SetGetDiffuse) {
    Material mat;
    Vec3 diffuse(0.8f, 0.8f, 0.8f);
    mat.SetDiffuse(diffuse);
    
    Vec3 retrieved = mat.GetDiffuse();
    EXPECT_NEAR(retrieved.x, 0.8f, 1e-5f);
    EXPECT_NEAR(retrieved.y, 0.8f, 1e-5f);
    EXPECT_NEAR(retrieved.z, 0.8f, 1e-5f);
}

TEST_F(MaterialTest, SetGetMetallic) {
    Material mat;
    mat.SetMetallic(0.5f);
    EXPECT_NEAR(mat.GetMetallic(), 0.5f, 1e-5f);
}

TEST_F(MaterialTest, SetGetRoughness) {
    Material mat;
    mat.SetRoughness(0.3f);
    EXPECT_NEAR(mat.GetRoughness(), 0.3f, 1e-5f);
}

TEST_F(MaterialTest, ValidMetallicRange) {
    Material mat;
    mat.SetMetallic(0.0f);
    EXPECT_NEAR(mat.GetMetallic(), 0.0f, 1e-5f);
    
    mat.SetMetallic(1.0f);
    EXPECT_NEAR(mat.GetMetallic(), 1.0f, 1e-5f);
}

TEST_F(MaterialTest, SetGetEmissive) {
    Material mat;
    Vec3 emissive(1.0f, 0.5f, 0.2f);
    mat.SetEmissiveColor(emissive);
    
    Vec3 retrieved = mat.GetEmissiveColor();
    EXPECT_NEAR(retrieved.x, 1.0f, 1e-5f);
    EXPECT_NEAR(retrieved.y, 0.5f, 1e-5f);
    EXPECT_NEAR(retrieved.z, 0.2f, 1e-5f);
}

TEST_F(MaterialTest, MultipleMaterials) {
    Material mat1;
    Material mat2;
    
    Vec3 color1(1.0f, 0.0f, 0.0f);
    Vec3 color2(0.0f, 1.0f, 0.0f);
    
    mat1.SetDiffuse(color1);
    mat2.SetDiffuse(color2);
    
    EXPECT_NEAR(mat1.GetDiffuse().x, 1.0f, 1e-5f);
    EXPECT_NEAR(mat2.GetDiffuse().y, 1.0f, 1e-5f);
}

TEST_F(MaterialTest, Inheritance) {
    auto parent = std::make_shared<Material>();
    auto child = std::make_shared<Material>();
    
    parent->SetDiffuse(Vec3(1, 0, 0));
    child->SetParent(parent);
    
    // Child should inherit diffuse
    EXPECT_NEAR(child->GetDiffuse().x, 1.0f, 1e-5f);
    
    // Override child
    child->SetDiffuse(Vec3(0, 1, 0));
    EXPECT_NEAR(child->GetDiffuse().x, 0.0f, 1e-5f);
    EXPECT_NEAR(child->GetDiffuse().y, 1.0f, 1e-5f);
}
