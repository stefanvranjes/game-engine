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
    Vec3 pos(5.0f, 10.0f, 15.0f);
    obj->GetTransform().SetPosition(pos);
    
    Vec3 retrievedPos = obj->GetTransform().GetPosition();
    EXPECT_EQ(retrievedPos.x, 5.0f);
    EXPECT_EQ(retrievedPos.y, 10.0f);
    EXPECT_EQ(retrievedPos.z, 15.0f);
}

TEST_F(GameObjectTest, SetGetScale) {
    auto obj = std::make_shared<GameObject>("TestObject");
    Vec3 scale(2.0f, 3.0f, 4.0f);
    obj->GetTransform().SetScale(scale);
    
    Vec3 retrievedScale = obj->GetTransform().GetScale();
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
    obj->GetTransform().SetPosition(Vec3(1.0f, 2.0f, 3.0f));
    EXPECT_EQ(obj->GetTransform().GetPosition().x, 1.0f);
    
    obj->GetTransform().SetPosition(Vec3(4.0f, 5.0f, 6.0f));
    EXPECT_EQ(obj->GetTransform().GetPosition().x, 4.0f);
}

TEST_F(GameObjectTest, RotationHandling) {
    auto obj = std::make_shared<GameObject>("TestObject");
    Vec3 rotation(0.0f, 90.0f, 0.0f);
    obj->GetTransform().SetRotation(rotation);
    
    Vec3 retrievedRotation = obj->GetTransform().GetRotation();
    // Simple check as we are using Euler angles
    EXPECT_EQ(retrievedRotation.y, 90.0f);
}

TEST_F(GameObjectTest, ScalePersistence) {
    auto obj = std::make_shared<GameObject>("TestObject");
    
    obj->GetTransform().SetScale(Vec3(2.0f, 2.0f, 2.0f));
    Vec3 scale1 = obj->GetTransform().GetScale();
    EXPECT_EQ(scale1.x, 2.0f);
    
    obj->GetTransform().SetScale(Vec3(3.0f, 3.0f, 3.0f));
    Vec3 scale2 = obj->GetTransform().GetScale();
    EXPECT_EQ(scale2.x, 3.0f);
}
