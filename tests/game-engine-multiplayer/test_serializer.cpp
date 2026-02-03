#include <gtest/gtest.h>
#include "serialization/Serializer.hpp"
#include "Message.hpp"
#include <vector>

class SerializerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(SerializerTest, SerializeSimpleMessage) {
    Serializer serializer;
    Message msg(MessageType::Chat, "Hello World");
    
    std::vector<char> buffer = serializer.serialize(msg);
    EXPECT_FALSE(buffer.empty());
    
    Message deserialized = serializer.deserialize(buffer);
    EXPECT_EQ(deserialized.GetType(), MessageType::Chat);
    EXPECT_EQ(deserialized.GetContent(), "Hello World");
}

TEST_F(SerializerTest, SerializeDifferentTypes) {
    Serializer serializer;
    
    // Test Join
    {
        Message msg(MessageType::Join, "Player1");
        auto buffer = serializer.serialize(msg);
        auto result = serializer.deserialize(buffer);
        EXPECT_EQ(result.GetType(), MessageType::Join);
        EXPECT_EQ(result.GetContent(), "Player1");
    }
    
    // Test Move (assuming Input type uses content for data)
    {
        Message msg(MessageType::Input, "W");
        auto buffer = serializer.serialize(msg);
        auto result = serializer.deserialize(buffer);
        EXPECT_EQ(result.GetType(), MessageType::Input);
        EXPECT_EQ(result.GetContent(), "W");
    }
}

TEST_F(SerializerTest, EmptyContent) {
    Serializer serializer;
    Message msg(MessageType::Ping, "");
    
    auto buffer = serializer.serialize(msg);
    auto result = serializer.deserialize(buffer);
    
    EXPECT_EQ(result.GetType(), MessageType::Ping);
    EXPECT_TRUE(result.GetContent().empty());
}

TEST_F(SerializerTest, LargeContent) {
    Serializer serializer;
    std::string large(1000, 'A');
    Message msg(MessageType::State, large);
    
    auto buffer = serializer.serialize(msg);
    auto result = serializer.deserialize(buffer);
    
    EXPECT_EQ(result.GetType(), MessageType::State);
    EXPECT_EQ(result.GetContent(), large);
}

TEST_F(SerializerTest, RoundTrip) {
    Serializer serializer;
    Message original(MessageType::Snapshot, "Game State Data");
    
    auto buffer = serializer.serialize(original);
    
    // Verify buffer format (Type + 4 bytes length + content)
    // Type (1) + Len (4) + Content (15) = 20 bytes
    // Expected size might vary depending on implementation details
    EXPECT_GE(buffer.size(), 5); 
    
    Message result = serializer.deserialize(buffer);
    
    EXPECT_EQ(result.GetType(), original.GetType());
    EXPECT_EQ(result.GetContent(), original.GetContent());
}
