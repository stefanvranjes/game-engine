#!/bin/bash

# Master node user data script

# Update system
apt-get update
apt-get upgrade -y

# Install dependencies
apt-get install -y \
    build-essential \
    cmake \
    git \
    nvidia-driver-525 \
    nvidia-cuda-toolkit \
    libasio-dev \
    lz4 \
    liblz4-dev \
    protobuf-compiler \
    libprotobuf-dev

# Clone and build game engine
cd /opt
git clone https://github.com/your-repo/game-engine.git
cd game-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)

# Create systemd service
cat > /etc/systemd/system/physics-master.service <<EOF
[Unit]
Description=Physics Simulation Master
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/game-engine/build
ExecStart=/opt/game-engine/build/GameEngine --distributed-master --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable physics-master
systemctl start physics-master

echo "Master node setup complete"
