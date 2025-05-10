#!/bin/bash

# Script para detectar las capacidades CUDA de la GPU

# Crear programa temporal para detectar la arquitectura
cat > /tmp/detect_cuda.cu << 'EOF'
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaDeviceProp prop;
    
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    int major = prop.major;
    int minor = prop.minor;
    
    std::cout << major << minor << std::endl;
    
    return 0;
}
EOF

# Compilar y ejecutar
nvcc -o /tmp/detect_cuda /tmp/detect_cuda.cu
architecture=$(/tmp/detect_cuda)

# Limpiar archivos temporales
rm /tmp/detect_cuda.cu /tmp/detect_cuda

# Compilar con la arquitectura detectada
cd build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$architecture"
make -j$(nproc)