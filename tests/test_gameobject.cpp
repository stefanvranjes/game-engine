#include <gtest/gtest.h>
#include "GameObject.h"
#include "Transform.h"

class GameObjectTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(GameObjectTest, CreateGameObject) {
    auto obj = std::make_shared<GameObject>("TestObject");
    EXPECT_TRUE(obj != nullptr);
    EXPECT_EQ(obj->GetName(), "TestObject");
}

TEST_F(GameObjectTest, SetGetPosition) {
    auto obj = std::make_shared<GameObject>("TestObject");
    glm::vec3 pos(5.0f, 10.0f, 15.0f);
    obj->GetTransform().SetPosition(pos);
    
    glm::vec3 retrievedPos = obj->GetTransform().GetPosition();
    EXPECT_EQ(retrievedPos.x, 5.0f);
    EXPECT_EQ(retrievedPos.y, 10.0f);
    EXPECT_EQ(retrievedPos.z, 15.0f);
}

TEST_F(GameObjectTest, SetGetScale) {
    auto obj = std::make_shared<GameObject>("TestObject");
    glm::vec3 scale(2.0f, 3.0f, 4.0f);
    obj->GetTransform().SetScale(scale);
    
    glm::vec3 retrievedScale = obj->GetTransform().GetScale();
    EXPECT_EQ(retrievedScale.x, 2.0f);
    EXPECT_EQ(retrievedScale.y, 3.0f);
    EXPECT_EQ(retrievedScale.z, 4.0f);
}

TEST_F(GameObjectTest, SetActive) {
    auto obj = std::make_shared<GameObject>("TestObject");
    EXPECT_TRUE(obj->IsActive());  // Default should be active
    
    obj->SetActive(false);
    EXPECT_FALSE(obj->IsActive());
    
    obj->SetActive(true);
    EXPECT_TRUE(obj->IsActive());
}

TEST_F(GameObjectTest, ObjectNameChange) {
    auto obj = std::make_shared<GameObject>("OriginalName");
    EXPECT_EQ(obj->GetName(), "OriginalName");
}

TEST_F(GameObjectTest, MultipleGameObjects) {
    auto obj1 = std::make_shared<GameObject>("Object1");
    auto obj2 = std::make_shared<GameObject>("Object2");
    auto obj3 = std::make_shared<GameObject>("Object3");
    
    EXPECT_EQ(obj1->GetName(), "Object1");
    EXPECT_EQ(obj2->GetName(), "Object2");
    EXPECT_EQ(obj3->GetName(), "Object3");
    
    EXPECT_NE(obj1, obj2);
    EXPECT_NE(obj2, obj3);
    EXPECT_NE(obj1, obj3);
}

TEST_F(GameObjectTest, PositionPersistence) {
    auto obj = std::make_shared<GameObject>("TestObject");
    
    // Set position multiple times
    obj->GetTransform().SetPosition(glm::vec3(1.0f, 2.0f, 3.0f));
    EXPECT_EQ(obj->GetTransform().GetPosition().x, 1.0f);
    
    obj->GetTransform().SetPosition(glm::vec3(4.0f, 5.0f, 6.0f));
    EXPECT_EQ(obj->GetTransform().GetPosition().x, 4.0f);
}

TEST_F(GameObjectTest, RotationHandling) {
    auto obj = std::make_shared<GameObject>("TestObject");
    glm::quat rotation = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    obj->GetTransform().SetRotation(rotation);
    
    glm::quat retrievedRotation = obj->GetTransform().GetRotation();
    EXPECT_NEAR(glm::length(retrievedRotation), 1.0f, 1e-6f);
}

TEST_F(GameObjectTest, ScalePersistence) {
    auto obj = std::make_shared<GameObject>("TestObject");
    
    obj->GetTransform().SetScale(glm::vec3(2.0f, 2.0f, 2.0f));
    glm::vec3 scale1 = obj->GetTransform().GetScale();
    EXPECT_EQ(scale1.x, 2.0f);
    
    obj->GetTransform().SetScale(glm::vec3(3.0f, 3.0f, 3.0f));
    glm::vec3 scale2 = obj->GetTransform().GetScale();
    EXPECT_EQ(scale2.x, 3.0f);
}
