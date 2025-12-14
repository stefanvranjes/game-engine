#include <gtest/gtest.h>
#include "serialization/Serializer.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class SerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(SerializationTest, SerializeSimpleData) {
    Serializer serializer;
    json data = {{"type", "test"}, {"value", 42}};
    
    auto serialized = serializer.serialize(data);
    EXPECT_FALSE(serialized.empty());
}

TEST_F(SerializationTest, DeserializeSimpleData) {
    Serializer serializer;
    json original = {{"type", "test"}, {"value", 42}};
    auto serialized = serializer.serialize(original);
    
    auto deserialized = serializer.deserialize(serialized);
    EXPECT_EQ(deserialized["type"], "test");
    EXPECT_EQ(deserialized["value"], 42);
}

TEST_F(SerializationTest, SerializeVector) {
    Serializer serializer;
    json data = {
        {"position", {1.0, 2.0, 3.0}},
        {"velocity", {0.5, -0.5, 1.0}}
    };
    
    auto serialized = serializer.serialize(data);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_NEAR(deserialized["position"][0].get<double>(), 1.0, 1e-5);
    EXPECT_NEAR(deserialized["position"][1].get<double>(), 2.0, 1e-5);
    EXPECT_NEAR(deserialized["position"][2].get<double>(), 3.0, 1e-5);
}

TEST_F(SerializationTest, SerializeComplexObject) {
    Serializer serializer;
    json player = {
        {"id", 1},
        {"name", "Player1"},
        {"position", {10.5, 20.3, 30.1}},
        {"health", 100},
        {"mana", 50},
        {"inventory", {"sword", "shield", "potion"}}
    };
    
    auto serialized = serializer.serialize(player);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_EQ(deserialized["id"], 1);
    EXPECT_EQ(deserialized["name"], "Player1");
    EXPECT_EQ(deserialized["health"], 100);
}

TEST_F(SerializationTest, SerializeArray) {
    Serializer serializer;
    json array = {
        {{"id", 1}, {"name", "Item1"}},
        {{"id", 2}, {"name", "Item2"}},
        {{"id", 3}, {"name", "Item3"}}
    };
    
    auto serialized = serializer.serialize(array);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_EQ(deserialized.size(), 3);
    EXPECT_EQ(deserialized[0]["name"], "Item1");
}

TEST_F(SerializationTest, SerializeNestedStructure) {
    Serializer serializer;
    json nested = {
        {"player", {
            {"name", "Hero"},
            {"stats", {
                {"strength", 20},
                {"dexterity", 15},
                {"constitution", 18}
            }}
        }},
        {"world", {
            {"level", 5},
            {"difficulty", "hard"}
        }}
    };
    
    auto serialized = serializer.serialize(nested);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_EQ(deserialized["player"]["name"], "Hero");
    EXPECT_EQ(deserialized["player"]["stats"]["strength"], 20);
    EXPECT_EQ(deserialized["world"]["difficulty"], "hard");
}

TEST_F(SerializationTest, SerializeBoolean) {
    Serializer serializer;
    json data = {
        {"is_active", true},
        {"is_visible", false}
    };
    
    auto serialized = serializer.serialize(data);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_TRUE(deserialized["is_active"].get<bool>());
    EXPECT_FALSE(deserialized["is_visible"].get<bool>());
}

TEST_F(SerializationTest, SerializeFloatingPoint) {
    Serializer serializer;
    json data = {
        {"pi", 3.14159265359},
        {"euler", 2.71828182846}
    };
    
    auto serialized = serializer.serialize(data);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_NEAR(deserialized["pi"].get<double>(), 3.14159265359, 1e-8);
    EXPECT_NEAR(deserialized["euler"].get<double>(), 2.71828182846, 1e-8);
}

TEST_F(SerializationTest, SerializeNull) {
    Serializer serializer;
    json data = {
        {"value", nullptr},
        {"name", "test"}
    };
    
    auto serialized = serializer.serialize(data);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_TRUE(deserialized["value"].is_null());
    EXPECT_EQ(deserialized["name"], "test");
}

TEST_F(SerializationTest, RoundTripSerialization) {
    Serializer serializer;
    json original = {
        {"int_val", 42},
        {"float_val", 3.14},
        {"string_val", "hello"},
        {"bool_val", true},
        {"array_val", {1, 2, 3, 4, 5}}
    };
    
    auto serialized = serializer.serialize(original);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_EQ(original, deserialized);
}

TEST_F(SerializationTest, EmptyObjectSerialization) {
    Serializer serializer;
    json empty = json::object();
    
    auto serialized = serializer.serialize(empty);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_TRUE(deserialized.is_object());
    EXPECT_EQ(deserialized.size(), 0);
}

TEST_F(SerializationTest, EmptyArraySerialization) {
    Serializer serializer;
    json empty = json::array();
    
    auto serialized = serializer.serialize(empty);
    auto deserialized = serializer.deserialize(serialized);
    
    EXPECT_TRUE(deserialized.is_array());
    EXPECT_EQ(deserialized.size(), 0);
}
