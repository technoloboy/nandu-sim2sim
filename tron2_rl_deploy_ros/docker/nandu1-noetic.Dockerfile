FROM ros:noetic-ros-base-focal
RUN sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g; s|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list && \
    apt-get -o Acquire::Retries=3 update && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
    build-essential cmake python3-numpy python3-yaml python3-nose \
    libeigen3-dev libssl-dev \
    ros-noetic-ros-control ros-noetic-ros-controllers \
    ros-noetic-kdl-parser ros-noetic-urdf ros-noetic-robot-state-publisher \
    ros-noetic-realtime-tools && rm -rf /var/lib/apt/lists/*
WORKDIR /ws
