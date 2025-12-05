#version 430 core

in vec4 v_Color;
in vec2 v_TexCoord;

out vec4 FragColor;

uniform sampler2D u_Texture;
uniform int u_HasTexture;

void main() {
    vec4 texColor = vec4(1.0);
    if (u_HasTexture != 0) {
        texColor = texture(u_Texture, v_TexCoord);
    }
    
    FragColor = v_Color * texColor;
}
