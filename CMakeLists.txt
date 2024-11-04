cmake_minimum_required(VERSION 3.10)
project(ParticleSimulation3D)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLM REQUIRED)

include_directories(include ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS})

set(SOURCES
    src/main.cpp
    src/particles.cu
    src/renderer.cpp
    src/camera.cpp
)

cuda_add_executable(ParticleSimulation3D ${SOURCES})
target_link_libraries(ParticleSimulation3D ${OPENGL_LIBRARIES} GLEW::GLEW glfw ${GLM_LIBRARIES} CUDA::cudart)