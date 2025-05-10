#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in int aType;

uniform mat4 projection;
uniform mat4 view;

out vec3 vColor;
flat out int vType;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = 8.0;
    vType = aType;
    
    // Asignar color basado en el tipo
    if (aType == 0) {
        vColor = vec3(1.0, 0.3, 0.3);  // Rojo
    } else if (aType == 1) {
        vColor = vec3(0.3, 1.0, 0.3);  // Verde
    } else {
        vColor = vec3(0.3, 0.3, 1.0);  // Azul
    }
}