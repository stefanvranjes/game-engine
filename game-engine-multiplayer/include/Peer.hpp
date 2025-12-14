class Peer {
public:
    Peer(const std::string& id);
    void connect(const std::string& address, int port);
    void disconnect();
    void sendMessage(const std::string& message);
    std::string receiveMessage();
    std::string getId() const;

private:
    std::string id;
    bool connected;
};