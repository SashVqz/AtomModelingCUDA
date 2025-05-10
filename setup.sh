#!/bin/bash

# Script para configurar el proyecto de simulación de partículas 3D

echo "=== Setting up Particle Simulation 3D ==="

# Actualizar repositorios
echo "Updating package repositories..."
sudo apt-get update

# Instalar dependencias básicas
echo "Installing basic dependencies..."
sudo apt-get install -y build-essential cmake git

# Instalar dependencias gráficas
echo "Installing graphics dependencies..."
sudo apt-get install -y libglew-dev libglfw3-dev

# Instalar GLM manualmente si no está disponible en el repositorio
echo "Installing GLM..."
sudo apt-get install -y libglm-dev

# Si GLM no se puede instalar desde el repositorio, instalarlo manualmente
if ! dpkg -l | grep -q libglm-dev; then
    echo "Installing GLM manually..."
    git clone https://github.com/g-truc/glm.git /tmp/glm
    cd /tmp/glm
    cmake . -DCMAKE_INSTALL_PREFIX=/usr/local
    sudo make install
    cd -
    rm -rf /tmp/glm
fi

# Verificar si CUDA ya está instalado
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Installing CUDA..."
    
    # Descargar e instalar CUDA (Ubuntu)
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda
    
    # Configurar variables de entorno
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
else
    echo "CUDA already installed"
fi

# Crear estructura de directorios si no existe
echo "Creating directory structure..."
mkdir -p build
mkdir -p src/shaders
mkdir -p include

# Limpiar compilación anterior si existe
cd build
rm -rf *

# Compilar el proyecto
echo "Building the project..."
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Verificar si la compilación fue exitosa
if [ $? -eq 0 ]; then
    echo "=== Build successful! ==="
    echo "You can run the simulation with: ./ParticleSimulation3D"
else
    echo "=== Build failed! ==="
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "=== Setup complete! ==="
echo "Controls:"
echo "  WASD - Move camera horizontally"
echo "  Space - Move camera up"
echo "  Left Shift - Move camera down"
echo "  Mouse - Look around"
echo "  Mouse Wheel - Zoom in/out"
echo "  ESC - Exit"
echo ""
echo "To run the simulation: cd build && ./ParticleSimulation3D"