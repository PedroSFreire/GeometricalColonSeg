#version 460
#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 4,local_size_y = 4,local_size_z = 4) in;

// Input and output data
layout(r32i, binding = 0) uniform iimage3D dataTexture;



void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
    ivec3 values[7];
    int i;
    int val;
    int32_t min;
    // Read the 16-bit signed integer from the texture
    min = int32_t(imageLoad(dataTexture, gid).r);
    if(min < 0)
            return;
    values[0] = gid;
    values[1] = gid + ivec3(1,0,0);
    values[2] = gid + ivec3(0,1,0);
    values[3] = gid + ivec3(0,0,1);
    values[4] = gid + ivec3(-1,0,0);
    values[5] = gid + ivec3(0,-1,0);
    values[6] = gid + ivec3(0,0,-1);


    for(i=0;i<7;i++){
        val = int32_t(imageLoad(dataTexture, values[i]).r);
        if(val < min && val>0)
                min = val;
    }
    for(i=0;i<7;i++){
        val = int32_t(imageLoad(dataTexture, values[i]).r);
        if(val>0){
            imageStore(dataTexture, values[i], ivec4(min, 0, 0, 0));
        }
    }
}