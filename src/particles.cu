#include "particles.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

__global__ void initParticlesKernel(Particle* particles, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        curandState state;
        curand_init(1234, i, 0, &state);
        particles[i].x = curand_uniform(&state) * 2.0f - 1.0f;
        particles[i].y = curand_uniform(&state) * 2.0f - 1.0f;
        particles[i].z = curand_uniform(&state) * 2.0f - 1.0f;
        particles[i].vx = curand_uniform(&state) * 0.1f - 0.05f;
        particles[i].vy = curand_uniform(&state) * 0.1f - 0.05f;
        particles[i].vz = curand_uniform(&state) * 0.1f - 0.05f;
        particles[i].type = i % 3;  // 3 types of particles
    }
}

__global__ void updateParticlesKernel(Particle* particles, int numParticles, float deltaTime, float attractionForce, float collisionRadius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float t = clock64() * 0.0001f;
    float centerX = sinf(t);
    float centerY = cosf(t);
    float centerZ = sinf(t) * cosf(t);

    for (int j = 0; j < numParticles; j++) {
        if (i != j) {
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dz = particles[j].z - particles[i].z;
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);

            // central gravity
            if (dist < collisionRadius) {
                float force = attractionForce / dist;
                particles[i].vx += force * dx;
                particles[i].vy += force * dy;
                particles[i].vz += force * dz;
            }

            if (dist < collisionRadius) {
                float tempVx = particles[i].vx;
                particles[i].vx = particles[j].vx;
                particles[j].vx = tempVx;

                float tempVy = particles[i].vy;
                particles[i].vy = particles[j].vy;
                particles[j].vy = tempVy;

                float tempVz = particles[i].vz;
                particles[i].vz = particles[j].vz;
                particles[j].vz = tempVz;
            }
        }
    }

    float dx = centerX - particles[i].x;
    float dy = centerY - particles[i].y;
    float dz = centerZ - particles[i].z;
    particles[i].vx += dx * 0.0001f;
    particles[i].vy += dy * 0.0001f;
    particles[i].vz += dz * 0.0001f;

    // update position
    particles[i].x += particles[i].vx * deltaTime;
    particles[i].y += particles[i].vy * deltaTime;
    particles[i].z += particles[i].vz * deltaTime;

    // bounce on walls
    if (particles[i].x < -1.0f || particles[i].x > 1.0f) particles[i].vx *= -1;
    if (particles[i].y < -1.0f || particles[i].y > 1.0f) particles[i].vy *= -1;
    if (particles[i].z < -1.0f || particles[i].z > 1.0f) particles[i].vz *= -1;
}

void initParticles(Particle* particles, int numParticles) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    initParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(particles, numParticles);
    cudaDeviceSynchronize();
}

void updateParticles(Particle* particles, int numParticles, float deltaTime) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    updateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(particles, numParticles, deltaTime, 0.05f, 0.001f);
    cudaDeviceSynchronize();
}