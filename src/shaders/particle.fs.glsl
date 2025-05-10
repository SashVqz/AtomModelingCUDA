#version 330 core
out vec4 FragColor;

in vec3 vColor;
flat in int vType;

void main() {
    // Crear efecto circular para las partículas
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    float dist = length(circCoord);
    
    if (dist > 1.0) {
        discard;
    }
    
    // Aplicar efecto de glow
    float glow = 1.0 - smoothstep(0.5, 1.0, dist);
    vec3 finalColor = vColor * glow;
    
    // Agregar un poco de transparencia
    float alpha = glow * 0.8;
    
    FragColor = vec4(finalColor, alpha);
}