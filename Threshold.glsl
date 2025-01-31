#version 460
#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 4,local_size_y = 4,local_size_z = 4) in;

// Input and output data
layout(r32i, binding = 0) uniform iimage3D dataTexture;

uniform int32_t threshold;



void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
    int w = imageSize(dataTexture).x;
    int h = imageSize(dataTexture).y;

    int32_t value = int32_t(imageLoad(dataTexture, gid).r);

    if (value < threshold) {
        value = int32_t(gid.x + gid.y*w +gid.z*w*h);
    } else {
        value = int32_t(-1);
    }


    imageStore(dataTexture, gid, ivec4(value, 0, 0, 0));
}