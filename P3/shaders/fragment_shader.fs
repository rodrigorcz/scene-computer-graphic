#version 330 core

uniform vec3 lightPos1;
uniform vec3 lightPos2;
uniform vec3 lightPosTV; 
       
uniform vec3 viewPos;
uniform float ka, kd, ks, ns;
uniform sampler2D samplerTexture;

uniform vec3 houseMin;
uniform vec3 houseMax;

uniform bool isLight1Internal;
uniform bool isLight2Internal;

vec3 lightColor1 = vec3(1.0);              // Luz branca
vec3 lightColor2 = vec3(1.0);              // Luz branca
vec3 lightColorTV = vec3(0.1, 0.3, 1.0);   // Azul 

varying vec2 out_texture;
varying vec3 out_normal;
varying vec3 out_fragPos;

float intensity = 10.0;

bool isInsideBox(vec3 point, vec3 boxMin, vec3 boxMax) {
    return all(greaterThanEqual(point, boxMin)) && all(lessThanEqual(point, boxMax));
}

void main() {
    vec3 ambient = ka * vec3(0.7);
    vec3 result = ambient;

    vec3 norm = normalize(out_normal);
    vec3 viewDir = normalize(viewPos - out_fragPos);

    bool is_inside_house = isInsideBox(out_fragPos, houseMin, houseMax);
    bool light1_outside  = any(lessThan(lightPos1, houseMin)) || any(greaterThan(lightPos1, houseMax));
    bool light2_outside  = any(lessThan(lightPos2, houseMin)) || any(greaterThan(lightPos2, houseMax));

    // --- Luz 1 ---
    if (!(is_inside_house && light1_outside)) {
        vec3 lightDir1 = normalize(lightPos1 - out_fragPos);
        float distance1 = length(lightPos1 - out_fragPos);
        float attenuation1 = 1.0 / (distance1 * distance1); 
        
        float diff1 = max(dot(norm, lightDir1), 0.0);
        vec3 diffuse1 = kd * diff1 * lightColor1 * attenuation1 * intensity;

        vec3 reflectDir1 = reflect(-lightDir1, norm);
        float spec1 = pow(max(dot(viewDir, reflectDir1), 0.0), ns);
        vec3 specular1 = ks * spec1 * lightColor1 * attenuation1 * intensity;

        result += diffuse1 + specular1;
    }

    // --- Luz 2 ---
    if (!(is_inside_house && light2_outside)) {
        vec3 lightDir2 = normalize(lightPos2 - out_fragPos);
        float distance2 = length(lightPos2 - out_fragPos);
        float attenuation2 = 1.0 / (distance2 * distance2); 

        float diff2 = max(dot(norm, lightDir2), 0.0);
        vec3 diffuse2 = kd * diff2 * lightColor2 * attenuation2 * intensity;

        vec3 reflectDir2 = reflect(-lightDir2, norm);
        float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), ns);
        vec3 specular2 = ks * spec2 * lightColor2 * attenuation2 * intensity;

        result += diffuse2 + specular2;
    }

    // --- Luz da TV ---
    if (!(is_inside_house && light2_outside)) {
        vec3 lightDirTV = normalize(lightPosTV - out_fragPos);
        float distanceTV = length(lightPosTV - out_fragPos);
        float attenuationTV = 1.0 / (distanceTV * distanceTV); 

        float diffTV = max(dot(norm, lightDirTV), 0.0);
        vec3 diffuseTV = kd * diffTV * lightColorTV * attenuationTV * (intensity * 0.8);

        vec3 reflectDirTV = reflect(-lightDirTV, norm);
        float specTV = pow(max(dot(viewDir, reflectDirTV), 0.0), ns);
        vec3 specularTV = ks * specTV * lightColorTV * attenuationTV * (intensity * 0.8);

        result += diffuseTV + specularTV;
    }
    result = clamp(result, 0.0, 1.0);
    gl_FragColor = vec4(result, 1.0) * texture2D(samplerTexture, out_texture);
}