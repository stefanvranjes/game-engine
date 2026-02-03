#include <gtest/gtest.h>
#include "Math/Mat4.h"
#include "Math/Vec3.h"
#include "Math/Quat.h"

class MathTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Vector3 tests
TEST_F(MathTest, Vector3DefaultConstructor) {
    Vec3 v;
    EXPECT_EQ(v.x, 0.0f);
    EXPECT_EQ(v.y, 0.0f);
    EXPECT_EQ(v.z, 0.0f);
}

TEST_F(MathTest, Vector3ParameterizedConstructor) {
    Vec3 v(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(v.x, 1.0f);
    EXPECT_EQ(v.y, 2.0f);
    EXPECT_EQ(v.z, 3.0f);
}

TEST_F(MathTest, Vector3Addition) {
    Vec3 v1(1.0f, 2.0f, 3.0f);
    Vec3 v2(4.0f, 5.0f, 6.0f);
    Vec3 result = v1 + v2;
    EXPECT_EQ(result.x, 5.0f);
    EXPECT_EQ(result.y, 7.0f);
    EXPECT_EQ(result.z, 9.0f);
}

TEST_F(MathTest, Vector3Subtraction) {
    Vec3 v1(5.0f, 6.0f, 7.0f);
    Vec3 v2(1.0f, 2.0f, 3.0f);
    Vec3 result = v1 - v2;
    EXPECT_EQ(result.x, 4.0f);
    EXPECT_EQ(result.y, 4.0f);
    EXPECT_EQ(result.z, 4.0f);
}

TEST_F(MathTest, Vector3DotProduct) {
    Vec3 v1(1.0f, 2.0f, 3.0f);
    Vec3 v2(4.0f, 5.0f, 6.0f);
    float dot = v1.Dot(v2);
    EXPECT_EQ(dot, 32.0f);  // 1*4 + 2*5 + 3*6 = 32
}

TEST_F(MathTest, Vector3CrossProduct) {
    Vec3 v1(1.0f, 0.0f, 0.0f);
    Vec3 v2(0.0f, 1.0f, 0.0f);
    Vec3 result = v1.Cross(v2);
    EXPECT_EQ(result.x, 0.0f);
    EXPECT_EQ(result.y, 0.0f);
    EXPECT_EQ(result.z, 1.0f);
}

TEST_F(MathTest, Vector3Length) {
    Vec3 v(3.0f, 4.0f, 0.0f);
    float len = v.Length();
    EXPECT_NEAR(len, 5.0f, 1e-6f);
}

TEST_F(MathTest, Vector3Normalization) {
    Vec3 v(3.0f, 4.0f, 0.0f);
    Vec3 normalized = v.Normalized();
    EXPECT_NEAR(normalized.Length(), 1.0f, 1e-6f);
}

// Matrix4 tests
TEST_F(MathTest, Matrix4Identity) {
    Mat4 m;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_EQ(m.m[i * 4 + j], 1.0f);
            } else {
                EXPECT_EQ(m.m[i * 4 + j], 0.0f);
            }
        }
    }
}

TEST_F(MathTest, Matrix4Translation) {
    Vec3 translation(5.0f, 10.0f, 15.0f);
    Mat4 m = Mat4::Translate(translation);
    
    Vec3 point(0.0f, 0.0f, 0.0f);
    Vec3 transformed = m * point;
    
    EXPECT_EQ(transformed.x, 5.0f);
    EXPECT_EQ(transformed.y, 10.0f);
    EXPECT_EQ(transformed.z, 15.0f);
}

TEST_F(MathTest, Matrix4Scale) {
    Vec3 scale(2.0f, 3.0f, 4.0f);
    Mat4 m = Mat4::Scale(scale);
    
    Vec3 point(1.0f, 1.0f, 1.0f);
    Vec3 transformed = m * point;
    
    EXPECT_EQ(transformed.x, 2.0f);
    EXPECT_EQ(transformed.y, 3.0f);
    EXPECT_EQ(transformed.z, 4.0f);
}

TEST_F(MathTest, Matrix4Multiplication) {
    Mat4 m1 = Mat4::Translate(Vec3(5.0f, 0.0f, 0.0f));
    Mat4 m2 = Mat4::Translate(Vec3(3.0f, 0.0f, 0.0f));
    Mat4 result = m1 * m2;
    
    Vec3 point(0.0f, 0.0f, 0.0f);
    Vec3 transformed = result * point;
    
    EXPECT_NEAR(transformed.x, 8.0f, 1e-6f);  // Combined translation
}

// Quaternion tests
TEST_F(MathTest, QuaternionIdentity) {
    Quat q = Quat::Identity();
    EXPECT_EQ(q.w, 1.0f);
    EXPECT_EQ(q.x, 0.0f);
    EXPECT_EQ(q.y, 0.0f);
    EXPECT_EQ(q.z, 0.0f);
}

TEST_F(MathTest, QuaternionRotation) {
    // In our engine, Mat4::FromQuaternion exists
    Quat q = Quat::Identity(); // Replace with axis angle if available
    Mat4 rotationMatrix = Mat4::FromQuaternion(q);
    
    Vec3 point(1.0f, 0.0f, 0.0f);
    Vec3 rotated = rotationMatrix * point;
    
    EXPECT_NEAR(rotated.x, 1.0f, 1e-5f);
}
