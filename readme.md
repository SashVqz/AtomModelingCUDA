# 3D Particle Simulation with CUDA

## Requirements

- NVIDIA GPU with CUDA support (Compute Capability 3.5+)
- Ubuntu 20.04+ (or similar Linux distribution)
- CUDA Toolkit 11.0+
- OpenGL 3.3+
- CMake 3.15+

## Dependencies

- CUDA Runtime
- OpenGL
- GLEW (OpenGL Extension Wrangler)
- GLFW3 (Graphics Library Framework)
- GLM (OpenGL Mathematics)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/particle-simulation-3d.git
cd particle-simulation-3d
```

2. Run the setup script:
```bash
chmod +x setup.sh ./setup.sh
```

3. Or install manually:
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y libglew-dev libglfw3-dev libglm-dev cmake

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
make -j$(nproc)
```

## Usage

Run the simulation:
```bash
cd build
./ParticleSimulation3D
```

### Controls

- **WASD**: Move camera horizontally
- **Space**: Move camera up
- **Left Shift**: Move camera down
- **Mouse**: Look around
- **Mouse Wheel**: Zoom in/out
- **ESC**: Exit

## Project Structure

```
particle-simulation-3d/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── setup.sh
├── include/
│   ├── camera.h
│   └── particles.h
└── src/
    ├── main.cpp
    ├── camera.cpp
    ├── particles.cu
    ├── renderer.cpp
    └── shaders/
        ├── particle.vs.glsl
        └── particle.fs.glsl
```

## Performance Tuning

You can adjust the number of particles by modifying the `NUM_PARTICLES` macro in `src/main.cpp`. The simulation is optimized for up to 10,000 particles, but this depends on your GPU capabilities.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CUDA Programming Guide by NVIDIA
- OpenGL Tutorial by LearnOpenGL
- GLM Documentation

## Contributing

Feel free to submit issues and enhancement requests!