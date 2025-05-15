FROM tensorflow/tensorflow:2.13.0-gpu
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV CXXFLAGS="-std=c++17"

# Update and install required packages
RUN apt-get update &&  apt-get install -y \
    build-essential libpng-dev libgl1-mesa-glx libsm6 libxext6 libxrender-dev cuda-toolkit-11-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install a C++17 compatible compiler
RUN apt-get update && apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install tensorflow_datasets apache_beam && \
    pip install numpy matplotlib imageio scikit-image 'opencv-python>=3.0,<4.0' && \
    pip install etils pypng easydict flow-vis einops hydra-core && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install path.py fast_slic tensorboardX

# link CUDA and cuDNN libraries
RUN ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib

WORKDIR /workspace
