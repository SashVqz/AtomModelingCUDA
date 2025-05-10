#include "particles.h" 
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void initParticlesKernel(Particle* particles, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        curandState state;
        curand_init(1234, i, 0, &state);
        
        // Inicializar posiciones dentro de una esfera
        float r = curand_uniform(&state) * 1.0f;
        float theta = curand_uniform(&state) * 2.0f * M_PI;
        float phi = curand_uniform(&state) * M_PI;
        
        particles[i].x = r * sinf(phi) * cosf(theta);
        particles[i].y = r * sinf(phi) * sinf(theta);
        particles[i].z = r * cosf(phi);
        
        // Inicializar velocidades
        particles[i].vx = (curand_uniform(&state) - 0.5f) * 0.2f;
        particles[i].vy = (curand_uniform(&state) - 0.5f) * 0.2f;
        particles[i].vz = (curand_uniform(&state) - 0.5f) * 0.2f;
        
        // Asignar tipo
        particles[i].type = i % 3;
    }
}

__global__ void updateParticlesKernel(Particle* particles, int numParticles, float deltaTime) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    // Parámetros de simulación
    const float attractionForce = 0.005f;
    const float collisionRadius = 0.05f;
    const float dampingFactor = 0.99f;
    const float centerForce = 0.001f;
    
    // Centro estático (podríamos hacerlo dinámico después)
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;

    // Memoria compartida para optimizar accesos
    __shared__ Particle shared_particles[BLOCK_SIZE];
    
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    // Iterar sobre todos los bloques
    for (int tile = 0; tile < gridDim.x; tile++) {
        // Cargar partículas del bloque actual a memoria compartida
        if (tile * blockDim.x + threadIdx.x < numParticles) {
            shared_particles[threadIdx.x] = particles[tile * blockDim.x + threadIdx.x];
        }
        __syncthreads();

        // Calcular fuerzas con partículas en el bloque actual
        for (int j = 0; j < blockDim.x && tile * blockDim.x + j < numParticles; j++) {
            if (tile * blockDim.x + j != i) {
                float dx = shared_particles[j].x - particles[i].x;
                float dy = shared_particles[j].y - particles[i].y;
                float dz = shared_particles[j].z - particles[i].z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                if (dist > 0.001f && dist < collisionRadius) {
                    // Atracción entre partículas del mismo tipo
                    if (particles[i].type == shared_particles[j].type) {
                        float force = attractionForce / (dist * dist);
                        fx += force * dx;
                        fy += force * dy;
                        fz += force * dz;
                    } else {
                        // Repulsión entre diferentes tipos
                        float force = -attractionForce / (dist * dist);
                        fx += force * dx;
                        fy += force * dy;
                        fz += force * dz;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Atracción hacia el centro
    float dx = centerX - particles[i].x;
    float dy = centerY - particles[i].y;
    float dz = centerZ - particles[i].z;
    fx += dx * centerForce;
    fy += dy * centerForce;
    fz += dz * centerForce;

    // Actualizar velocidad
    particles[i].vx += fx * deltaTime;
    particles[i].vy += fy * deltaTime;
    particles[i].vz += fz * deltaTime;

    // Aplicar amortiguamiento
    particles[i].vx *= dampingFactor;
    particles[i].vy *= dampingFactor;
    particles[i].vz *= dampingFactor;

    // Actualizar posición
    particles[i].x += particles[i].vx * deltaTime;
    particles[i].y += particles[i].vy * deltaTime;
    particles[i].z += particles[i].vz * deltaTime;

    // Rebotes en paredes con amortiguamiento
    const float wallDamping = 0.8f;
    if (particles[i].x < -2.0f) { particles[i].x = -2.0f; particles[i].vx = -particles[i].vx * wallDamping; }
    if (particles[i].x > 2.0f) { particles[i].x = 2.0f; particles[i].vx = -particles[i].vx * wallDamping; }
    if (particles[i].y < -2.0f) { particles[i].y = -2.0f; particles[i].vy = -particles[i].vy * wallDamping; }
    if (particles[i].y > 2.0f) { particles[i].y = 2.0f; particles[i].vy = -particles[i].vy * wallDamping; }
    if (particles[i].z < -2.0f) { particles[i].z = -2.0f; particles[i].vz = -particles[i].vz * wallDamping; }
    if (particles[i].z > 2.0f) { particles[i].z = 2.0f; particles[i].vz = -particles[i].vz * wallDamping; }
}

void initParticles(Particle* particles, int numParticles) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    initParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(particles, numParticles);
    cudaDeviceSynchronize();
}

void updateParticles(Particle* particles, int numParticles, float deltaTime) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    updateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(particles, numParticles, deltaTime);
    cudaDeviceSynchronize();
}