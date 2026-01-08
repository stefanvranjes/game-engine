#include <gtest/gtest.h>
#include "MaterialNew.h"
#include "Texture.h"
#include "Math/Vec3.h"

class MaterialTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(MaterialTest, DefaultMaterial) {
    MaterialNew mat("DefaultMaterial");
    EXPECT_EQ(mat.GetName(), "DefaultMaterial");
}

TEST_F(MaterialTest, SetGetAlbedo) {
    MaterialNew mat("TestMaterial");
    glm::vec3 albedo(0.8f, 0.8f, 0.8f);
    mat.SetAlbedo(albedo);
    
    glm::vec3 retrievedAlbedo = mat.GetAlbedo();
    EXPECT_NEAR(retrievedAlbedo.x, 0.8f, 1e-5f);
    EXPECT_NEAR(retrievedAlbedo.y, 0.8f, 1e-5f);
    EXPECT_NEAR(retrievedAlbedo.z, 0.8f, 1e-5f);
}

TEST_F(MaterialTest, SetGetMetallic) {
    MaterialNew mat("TestMaterial");
    mat.SetMetallic(0.5f);
    EXPECT_NEAR(mat.GetMetallic(), 0.5f, 1e-5f);
}

TEST_F(MaterialTest, SetGetRoughness) {
    MaterialNew mat("TestMaterial");
    mat.SetRoughness(0.3f);
    EXPECT_NEAR(mat.GetRoughness(), 0.3f, 1e-5f);
}

TEST_F(MaterialTest, ValidMetallicRange) {
    MaterialNew mat("TestMaterial");
    
    // Test boundary values
    mat.SetMetallic(0.0f);
    EXPECT_NEAR(mat.GetMetallic(), 0.0f, 1e-5f);
    
    mat.SetMetallic(1.0f);
    EXPECT_NEAR(mat.GetMetallic(), 1.0f, 1e-5f);
}

TEST_F(MaterialTest, ValidRoughnessRange) {
    MaterialNew mat("TestMaterial");
    
    // Test boundary values
    mat.SetRoughness(0.0f);
    EXPECT_NEAR(mat.GetRoughness(), 0.0f, 1e-5f);
    
    mat.SetRoughness(1.0f);
    EXPECT_NEAR(mat.GetRoughness(), 1.0f, 1e-5f);
}

TEST_F(MaterialTest, SetGetEmissive) {
    MaterialNew mat("TestMaterial");
    glm::vec3 emissive(1.0f, 0.5f, 0.2f);
    mat.SetEmissive(emissive);
    
    glm::vec3 retrievedEmissive = mat.GetEmissive();
    EXPECT_NEAR(retrievedEmissive.x, 1.0f, 1e-5f);
    EXPECT_NEAR(retrievedEmissive.y, 0.5f, 1e-5f);
    EXPECT_NEAR(retrievedEmissive.z, 0.2f, 1e-5f);
}

TEST_F(MaterialTest, AlbedoColorBounds) {
    MaterialNew mat("TestMaterial");
    
    // Test valid color range (0-1)
    glm::vec3 color(0.5f, 0.5f, 0.5f);
    mat.SetAlbedo(color);
    
    glm::vec3 retrieved = mat.GetAlbedo();
    EXPECT_GE(retrieved.x, 0.0f);
    EXPECT_LE(retrieved.x, 1.0f);
    EXPECT_GE(retrieved.y, 0.0f);
    EXPECT_LE(retrieved.y, 1.0f);
    EXPECT_GE(retrieved.z, 0.0f);
    EXPECT_LE(retrieved.z, 1.0f);
}

TEST_F(MaterialTest, MultipleMaterials) {
    MaterialNew mat1("Material1");
    MaterialNew mat2("Material2");
    
    glm::vec3 color1(1.0f, 0.0f, 0.0f);
    glm::vec3 color2(0.0f, 1.0f, 0.0f);
    
    mat1.SetAlbedo(color1);
    mat2.SetAlbedo(color2);
    
    glm::vec3 retrieved1 = mat1.GetAlbedo();
    glm::vec3 retrieved2 = mat2.GetAlbedo();
    
    EXPECT_NEAR(retrieved1.x, 1.0f, 1e-5f);
    EXPECT_NEAR(retrieved2.y, 1.0f, 1e-5f);
}

TEST_F(MaterialTest, PBRPropertiesIndependence) {
    MaterialNew mat("TestMaterial");
    
    mat.SetMetallic(0.8f);
    mat.SetRoughness(0.2f);
    glm::vec3 albedo(0.5f, 0.5f, 0.5f);
    mat.SetAlbedo(albedo);
    
    // Verify all properties are independent
    EXPECT_NEAR(mat.GetMetallic(), 0.8f, 1e-5f);
    EXPECT_NEAR(mat.GetRoughness(), 0.2f, 1e-5f);
    EXPECT_NEAR(mat.GetAlbedo().x, 0.5f, 1e-5f);
}
