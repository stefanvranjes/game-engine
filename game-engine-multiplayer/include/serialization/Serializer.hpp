#ifndef SERIALIZER_HPP
#define SERIALIZER_HPP

#include <vector>
#include <string>

class Message;

class Serializer {
public:
    Serializer();
    std::vector<char> serialize(const Message& message);
    Message deserialize(const std::vector<char>& buffer);
};

#endif // SERIALIZER_HPP
