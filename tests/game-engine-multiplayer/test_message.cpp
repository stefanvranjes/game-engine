#include <gtest/gtest.h>
#include "Message.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class NetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(NetworkTest, MessageCreation) {
    Message msg(MessageType::Chat, "Hello, World!");
    EXPECT_EQ(msg.GetType(), MessageType::Chat);
    EXPECT_EQ(msg.GetContent(), "Hello, World!");
}

TEST_F(NetworkTest, MessageSerialization) {
    Message msg(MessageType::State, "test_data");
    auto serialized = msg.serialize();
    
    EXPECT_FALSE(serialized.empty());
    EXPECT_GT(serialized.size(), 0);
}

TEST_F(NetworkTest, MessageDeserialization) {
    Message original(MessageType::Chat, "Test message");
    auto serialized = original.serialize();
    
    Message deserialized = Message::deserialize(serialized);
    EXPECT_EQ(deserialized.GetType(), original.GetType());
    EXPECT_EQ(deserialized.GetContent(), original.GetContent());
}

TEST_F(NetworkTest, MessageTypeChat) {
    Message msg(MessageType::Chat, "Hello");
    EXPECT_EQ(msg.GetType(), MessageType::Chat);
}

TEST_F(NetworkTest, MessageTypeState) {
    json state = {
        {"position", {1.0, 2.0, 3.0}},
        {"rotation", {0.0, 0.0, 0.0, 1.0}}
    };
    Message msg(MessageType::State, state.dump());
    EXPECT_EQ(msg.GetType(), MessageType::State);
}

TEST_F(NetworkTest, MessageTypePing) {
    Message msg(MessageType::Ping, "");
    EXPECT_EQ(msg.GetType(), MessageType::Ping);
}

TEST_F(NetworkTest, MessageTypeJoin) {
    json joinData = {{"player_name", "Player1"}};
    Message msg(MessageType::Join, joinData.dump());
    EXPECT_EQ(msg.GetType(), MessageType::Join);
}

TEST_F(NetworkTest, MessageTypeLeave) {
    Message msg(MessageType::Leave, "");
    EXPECT_EQ(msg.GetType(), MessageType::Leave);
}

TEST_F(NetworkTest, MultipleMessageSerialization) {
    Message msg1(MessageType::Chat, "Message 1");
    Message msg2(MessageType::State, "State data");
    Message msg3(MessageType::Ping, "");
    
    auto ser1 = msg1.serialize();
    auto ser2 = msg2.serialize();
    auto ser3 = msg3.serialize();
    
    EXPECT_NE(ser1, ser2);
    EXPECT_NE(ser2, ser3);
    EXPECT_NE(ser1, ser3);
}

TEST_F(NetworkTest, JSONPayloadParsing) {
    json payload = {
        {"x", 10.5},
        {"y", 20.3},
        {"z", 30.1}
    };
    
    Message msg(MessageType::State, payload.dump());
    auto parsed = json::parse(msg.GetContent());
    
    EXPECT_NEAR(parsed["x"].get<double>(), 10.5, 1e-5);
    EXPECT_NEAR(parsed["y"].get<double>(), 20.3, 1e-5);
    EXPECT_NEAR(parsed["z"].get<double>(), 30.1, 1e-5);
}

TEST_F(NetworkTest, ComplexMessagePayload) {
    json payload = {
        {"player", {
            {"id", 1},
            {"name", "TestPlayer"},
            {"position", {1.0, 2.0, 3.0}},
            {"health", 100}
        }},
        {"action", "move"}
    };
    
    Message msg(MessageType::State, payload.dump());
    auto parsed = json::parse(msg.GetContent());
    
    EXPECT_EQ(parsed["player"]["id"], 1);
    EXPECT_EQ(parsed["player"]["name"], "TestPlayer");
    EXPECT_EQ(parsed["action"], "move");
}

TEST_F(NetworkTest, EmptyMessageContent) {
    Message msg(MessageType::Ping, "");
    EXPECT_TRUE(msg.GetContent().empty());
}

TEST_F(NetworkTest, LargeMessagePayload) {
    std::string largePayload(10000, 'a');
    Message msg(MessageType::State, largePayload);
    
    EXPECT_EQ(msg.GetContent().size(), 10000);
    auto serialized = msg.serialize();
    EXPECT_GT(serialized.size(), 10000);  // Should include header
}
