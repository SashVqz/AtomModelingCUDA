cmake_minimum_required(VERSION 3.15)
project(ParticleSimulation3D LANGUAGES CXX CUDA)

# C++17 para features modernas
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Buscar dependencias
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Buscar CUDA de manera más simple
find_package(CUDAToolkit REQUIRED)

# Includes
include_directories(
    src
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    /usr/include  # Para GLM
    /usr/local/include  # Para GLM
)

# Headers
set(HEADERS
    src/particles.h
    src/camera.h
)

# Sources
set(SOURCES
    src/main.cpp
    src/renderer.cpp
    src/camera.cpp
)

# CUDA sources - IMPORTANTE: especificar que es un archivo CUDA
set(CUDA_SOURCES
    src/particles.cu
)

# Marcar explícitamente el archivo CUDA
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Agregar ejecutable
add_executable(ParticleSimulation3D 
    ${SOURCES} 
    ${CUDA_SOURCES}
    ${HEADERS}
)

# Configurar propiedades CUDA
set_property(TARGET ParticleSimulation3D PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# No especificar CUDA_ARCHITECTURES, dejar que se autodetecte
if(NOT CMAKE_CUDA_ARCHITECTURES)
    # Si no se puede autodetectar, usar un conjunto básico
    set_property(TARGET ParticleSimulation3D PROPERTY CUDA_ARCHITECTURES "35;50;61;70;75")
endif()

# Enlaces
target_link_libraries(ParticleSimulation3D 
    PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart
)

# Flags de compilación CUDA
target_compile_options(ParticleSimulation3D PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++14>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ParticleSimulation3D PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-g>
        $<$<COMPILE_LANGUAGE:CUDA>:-G>
    )
endif()