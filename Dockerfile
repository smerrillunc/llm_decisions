# 1. Start from an official NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. Install Python 3.11, pip, and other common utilities
# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python and python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Ensure pip is up-to-date
RUN pip3 install --upgrade pip

# 3. Set up a working directory
WORKDIR /app

# 4. Copy all current repository contents into the working directory
COPY . .

# 5. Install Python packages.
# Install PyTorch compatible with CUDA 11.8 first.
# Using a specific version (e.g., PyTorch 2.1.2) for reproducibility.
RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 6. Ensure Unsloth can be correctly installed and built.
# Use the Unsloth auto-install script to determine and run the optimal install command.
# The script prints the pip install command, so we use eval to execute it.
# Ensure python3 points to python3.11 for the script.
RUN /bin/bash -c "eval $(wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python3 -)"

# Install other requirements from requirements.txt
RUN pip3 install -r requirements.txt

# 7. Set appropriate environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Update LD_LIBRARY_PATH for CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

# 8. The default command should ideally be bash
CMD ["bash"]
