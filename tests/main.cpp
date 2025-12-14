#include <gtest/gtest.h>

// Google Test entry point
// Individual test files are linked in CMakeLists.txt
// Run tests with: ./tests --gtest_filter=TestName.*
// Run with verbose output: ./tests --gtest_verbose
// Run specific test: ./tests --gtest_filter=TestClass.TestMethod

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
