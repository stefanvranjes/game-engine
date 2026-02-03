#include <gtest/gtest.h>
#include "Transform.h"
#include "Math/Vec3.h"
#include "Math/Quat.h"

class TransformTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(TransformTest, DefaultConstructor) {
    Transform t;
    Vec3 pos = t.GetPosition();
    EXPECT_EQ(pos.x, 0.0f);
    EXPECT_EQ(pos.y, 0.0f);
    EXPECT_EQ(pos.z, 0.0f);
}

TEST_F(TransformTest, SetPosition) {
    Transform t;
    Vec3 newPos(5.0f, 10.0f, 15.0f);
    t.SetPosition(newPos);
    
    Vec3 pos = t.GetPosition();
    EXPECT_EQ(pos.x, 5.0f);
    EXPECT_EQ(pos.y, 10.0f);
    EXPECT_EQ(pos.z, 15.0f);
}

TEST_F(TransformTest, SetScale) {
    Transform t;
    Vec3 newScale(2.0f, 3.0f, 4.0f);
    t.SetScale(newScale);
    
    Vec3 scale = t.GetScale();
    EXPECT_EQ(scale.x, 2.0f);
    EXPECT_EQ(scale.y, 3.0f);
    EXPECT_EQ(scale.z, 4.0f);
}

TEST_F(TransformTest, SetRotation) {
    Transform t;
    Vec3 newRot(90.0f, 0.0f, 0.0f);
    t.SetRotation(newRot);
    
    Vec3 rot = t.GetRotation();
    EXPECT_EQ(rot.x, 90.0f);
}

TEST_F(TransformTest, GetMatrix) {
    Transform t;
    t.SetPosition(Vec3(5.0f, 10.0f, 15.0f));
    t.SetScale(Vec3(2.0f, 2.0f, 2.0f));
    
    Mat4 matrix = t.GetMatrix();
    // In column-major, translation is in [12], [13], [14]
    EXPECT_EQ(matrix.m[12], 5.0f);
    EXPECT_EQ(matrix.m[13], 10.0f);
    EXPECT_EQ(matrix.m[14], 15.0f);
}
