#include <gtest/gtest.h>
#include "Math/Mat4.h"
#include "Math/Vec3.h"
#include "Math/Quat.h"

class MathTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// Vector3 tests
TEST_F(MathTest, Vector3DefaultConstructor) {
    glm::vec3 v;
    EXPECT_EQ(v.x, 0.0f);
    EXPECT_EQ(v.y, 0.0f);
    EXPECT_EQ(v.z, 0.0f);
}

TEST_F(MathTest, Vector3ParameterizedConstructor) {
    glm::vec3 v(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(v.x, 1.0f);
    EXPECT_EQ(v.y, 2.0f);
    EXPECT_EQ(v.z, 3.0f);
}

TEST_F(MathTest, Vector3Addition) {
    glm::vec3 v1(1.0f, 2.0f, 3.0f);
    glm::vec3 v2(4.0f, 5.0f, 6.0f);
    glm::vec3 result = v1 + v2;
    EXPECT_EQ(result.x, 5.0f);
    EXPECT_EQ(result.y, 7.0f);
    EXPECT_EQ(result.z, 9.0f);
}

TEST_F(MathTest, Vector3Subtraction) {
    glm::vec3 v1(5.0f, 6.0f, 7.0f);
    glm::vec3 v2(1.0f, 2.0f, 3.0f);
    glm::vec3 result = v1 - v2;
    EXPECT_EQ(result.x, 4.0f);
    EXPECT_EQ(result.y, 4.0f);
    EXPECT_EQ(result.z, 4.0f);
}

TEST_F(MathTest, Vector3DotProduct) {
    glm::vec3 v1(1.0f, 2.0f, 3.0f);
    glm::vec3 v2(4.0f, 5.0f, 6.0f);
    float dot = glm::dot(v1, v2);
    EXPECT_EQ(dot, 32.0f);  // 1*4 + 2*5 + 3*6 = 32
}

TEST_F(MathTest, Vector3CrossProduct) {
    glm::vec3 v1(1.0f, 0.0f, 0.0f);
    glm::vec3 v2(0.0f, 1.0f, 0.0f);
    glm::vec3 result = glm::cross(v1, v2);
    EXPECT_EQ(result.x, 0.0f);
    EXPECT_EQ(result.y, 0.0f);
    EXPECT_EQ(result.z, 1.0f);
}

TEST_F(MathTest, Vector3Length) {
    glm::vec3 v(3.0f, 4.0f, 0.0f);
    float len = glm::length(v);
    EXPECT_NEAR(len, 5.0f, 1e-6f);
}

TEST_F(MathTest, Vector3Normalization) {
    glm::vec3 v(3.0f, 4.0f, 0.0f);
    glm::vec3 normalized = glm::normalize(v);
    EXPECT_NEAR(glm::length(normalized), 1.0f, 1e-6f);
}

// Matrix4 tests
TEST_F(MathTest, Matrix4Identity) {
    glm::mat4 m = glm::mat4(1.0f);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_EQ(m[i][j], 1.0f);
            } else {
                EXPECT_EQ(m[i][j], 0.0f);
            }
        }
    }
}

TEST_F(MathTest, Matrix4Translation) {
    glm::vec3 translation(5.0f, 10.0f, 15.0f);
    glm::mat4 m = glm::translate(glm::mat4(1.0f), translation);
    
    glm::vec4 point(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 transformed = m * point;
    
    EXPECT_EQ(transformed.x, 5.0f);
    EXPECT_EQ(transformed.y, 10.0f);
    EXPECT_EQ(transformed.z, 15.0f);
}

TEST_F(MathTest, Matrix4Scale) {
    glm::vec3 scale(2.0f, 3.0f, 4.0f);
    glm::mat4 m = glm::scale(glm::mat4(1.0f), scale);
    
    glm::vec4 point(1.0f, 1.0f, 1.0f, 1.0f);
    glm::vec4 transformed = m * point;
    
    EXPECT_EQ(transformed.x, 2.0f);
    EXPECT_EQ(transformed.y, 3.0f);
    EXPECT_EQ(transformed.z, 4.0f);
}

TEST_F(MathTest, Matrix4Multiplication) {
    glm::mat4 m1 = glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 0.0f, 0.0f));
    glm::mat4 m2 = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0.0f, 0.0f));
    glm::mat4 result = m1 * m2;
    
    glm::vec4 point(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 transformed = result * point;
    
    EXPECT_NEAR(transformed.x, 8.0f, 1e-6f);  // Combined translation
}

// Quaternion tests
TEST_F(MathTest, QuaternionIdentity) {
    glm::quat q = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_EQ(q.w, 1.0f);
    EXPECT_EQ(q.x, 0.0f);
    EXPECT_EQ(q.y, 0.0f);
    EXPECT_EQ(q.z, 0.0f);
}

TEST_F(MathTest, QuaternionRotation) {
    glm::quat q = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 rotationMatrix = glm::mat4_cast(q);
    
    glm::vec4 point(1.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 rotated = rotationMatrix * point;
    
    EXPECT_NEAR(rotated.x, 0.0f, 1e-5f);
    EXPECT_NEAR(rotated.y, 1.0f, 1e-5f);
    EXPECT_NEAR(rotated.z, 0.0f, 1e-5f);
}

TEST_F(MathTest, QuaternionInterpolation) {
    glm::quat q1 = glm::angleAxis(glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat q2 = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    
    glm::quat interpolated = glm::slerp(q1, q2, 0.5f);
    EXPECT_TRUE(glm::length(interpolated) > 0.99f);  // Should still be normalized
}
