#ifndef PARTICLES_H
#define PARTICLES_H

struct Particle {
    float x, y, z;       // Position
    float vx, vy, vz;    // Velocity
    int type;         
};

void initParticles(Particle* particles, int numParticles);
void updateParticles(Particle* particles, int numParticles, float deltaTime);

#endif