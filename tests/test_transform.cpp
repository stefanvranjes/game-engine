#include <gtest/gtest.h>
#include "Transform.h"
#include "Math/Vector3.h"
#include "Math/Quaternion.h"

class TransformTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(TransformTest, DefaultConstructor) {
    Transform t;
    glm::vec3 pos = t.GetPosition();
    EXPECT_EQ(pos.x, 0.0f);
    EXPECT_EQ(pos.y, 0.0f);
    EXPECT_EQ(pos.z, 0.0f);
}

TEST_F(TransformTest, SetPosition) {
    Transform t;
    glm::vec3 newPos(5.0f, 10.0f, 15.0f);
    t.SetPosition(newPos);
    
    glm::vec3 pos = t.GetPosition();
    EXPECT_EQ(pos.x, 5.0f);
    EXPECT_EQ(pos.y, 10.0f);
    EXPECT_EQ(pos.z, 15.0f);
}

TEST_F(TransformTest, SetScale) {
    Transform t;
    glm::vec3 newScale(2.0f, 3.0f, 4.0f);
    t.SetScale(newScale);
    
    glm::vec3 scale = t.GetScale();
    EXPECT_EQ(scale.x, 2.0f);
    EXPECT_EQ(scale.y, 3.0f);
    EXPECT_EQ(scale.z, 4.0f);
}

TEST_F(TransformTest, SetRotation) {
    Transform t;
    glm::quat rotation = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    t.SetRotation(rotation);
    
    glm::quat rot = t.GetRotation();
    EXPECT_NEAR(glm::length(rot), 1.0f, 1e-6f);  // Should be normalized
}

TEST_F(TransformTest, Translate) {
    Transform t;
    t.SetPosition(glm::vec3(5.0f, 0.0f, 0.0f));
    t.Translate(glm::vec3(3.0f, 0.0f, 0.0f));
    
    glm::vec3 pos = t.GetPosition();
    EXPECT_EQ(pos.x, 8.0f);
    EXPECT_EQ(pos.y, 0.0f);
    EXPECT_EQ(pos.z, 0.0f);
}

TEST_F(TransformTest, GetForwardVector) {
    Transform t;
    // Default forward should be -Z
    glm::vec3 forward = t.GetForward();
    EXPECT_NEAR(forward.x, 0.0f, 1e-5f);
    EXPECT_NEAR(forward.y, 0.0f, 1e-5f);
    EXPECT_NEAR(forward.z, -1.0f, 1e-5f);
}

TEST_F(TransformTest, GetRightVector) {
    Transform t;
    // Default right should be X
    glm::vec3 right = t.GetRight();
    EXPECT_NEAR(right.x, 1.0f, 1e-5f);
    EXPECT_NEAR(right.y, 0.0f, 1e-5f);
    EXPECT_NEAR(right.z, 0.0f, 1e-5f);
}

TEST_F(TransformTest, GetUpVector) {
    Transform t;
    // Default up should be Y
    glm::vec3 up = t.GetUp();
    EXPECT_NEAR(up.x, 0.0f, 1e-5f);
    EXPECT_NEAR(up.y, 1.0f, 1e-5f);
    EXPECT_NEAR(up.z, 0.0f, 1e-5f);
}

TEST_F(TransformTest, GetMatrix) {
    Transform t;
    t.SetPosition(glm::vec3(5.0f, 10.0f, 15.0f));
    t.SetScale(glm::vec3(2.0f, 2.0f, 2.0f));
    
    glm::mat4 matrix = t.GetMatrix();
    EXPECT_TRUE(glm::length(matrix[3]) > 0.0f);  // Should have position data
}

TEST_F(TransformTest, ParentChildHierarchy) {
    Transform parent;
    Transform child;
    
    parent.SetPosition(glm::vec3(5.0f, 0.0f, 0.0f));
    child.SetPosition(glm::vec3(3.0f, 0.0f, 0.0f));
    
    // Verify individual positions
    EXPECT_EQ(parent.GetPosition().x, 5.0f);
    EXPECT_EQ(child.GetPosition().x, 3.0f);
}

TEST_F(TransformTest, LocalToWorldConversion) {
    Transform t;
    glm::vec3 localPoint(1.0f, 0.0f, 0.0f);
    t.SetPosition(glm::vec3(5.0f, 0.0f, 0.0f));
    
    // Local point should be transformed by transform matrix
    glm::vec4 worldPoint = t.GetMatrix() * glm::vec4(localPoint, 1.0f);
    EXPECT_EQ(worldPoint.x, 6.0f);
    EXPECT_EQ(worldPoint.y, 0.0f);
    EXPECT_EQ(worldPoint.z, 0.0f);
}
