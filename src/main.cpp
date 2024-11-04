#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include "particles.h"
#include "camera.h"

#define NUM_PARTICLES 1000

Particle* particles;
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
 
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "3D Particle Simulation", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    particles = new Particle[NUM_PARTICLES];
    initParticles(particles, NUM_PARTICLES);

    // Load shaders
    GLuint shaderProgram = loadShaders("src/shaders/particle.vs.glsl", "src/shaders/particle.fs.glsl");

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * sizeof(Particle), particles, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    glEnableVertexAttribArray(0);

    while (!glfwWindowShouldClose(window)) {
        float deltaTime = 0.01f;  // time between current frame and last frame
        updateParticles(particles, NUM_PARTICLES, deltaTime);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * sizeof(Particle), particles, GL_DYNAMIC_DRAW);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);

        glUniform3f(glGetUniformLocation(shaderProgram, "particleColor"), 1.0f, 0.5f, 0.2f);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    delete[] particles;

    glfwTerminate();
    return 0;
}