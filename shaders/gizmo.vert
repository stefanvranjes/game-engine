#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 color;

out vec3 TexCoords; // Not really used
out vec3 FragColor;

void main()
{
    FragColor = color;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
