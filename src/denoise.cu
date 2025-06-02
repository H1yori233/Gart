#include "denoise.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

static glm::vec3* dev_image_temp = NULL;

#if OPTIMIZED_GBUFFER
// octahedron normal compression

// X^2 + Y^2 + Z^2 = 1 -> we can only store X and Y, but need SIGN of Z
__device__ 
glm::vec2 encodeNormal(glm::vec3 normal) {
    // project positions in the sphere onto a octahedron (which |X'| + |Y'| + |Z'| = 1)
    glm::vec2 p = glm::vec2(normal.x, normal.y) / 
                 (fabsf(normal.x) + fabsf(normal.y) + fabsf(normal.z));
    float x, y;
    if (normal.z < 0.f) {
        // |p'.x| = |p.x| + |p.z| = |p.x| - p.z
        if (p.x >= 0.f) {
            x = (1.0f - fabsf(p.y));
        } else {
            x = -(1.0f - fabsf(p.y));
        }
        
        // |p'.y| = |p.y| + |p.z| = |p.y| - p.z
        if (p.y >= 0.f) {
            y = (1.0f - fabsf(p.x));
        } else {
            y = -(1.0f - fabsf(p.x));
        }
    } else {
        x = p.x;
        y = p.y;
        // so p.z = 1 - |p.x| - |p.y|
    }
    return glm::vec2(x, y);
}

__device__ 
glm::vec3 decodeNormal(glm::vec2 encoded) {
    // |X'| + |Y'| + |Z'| = 1 -> p.z = 1 - |p.x| - |p.y|
    glm::vec3 n = glm::vec3(encoded.x, encoded.y, 
                            1.0f - fabsf(encoded.x) - fabsf(encoded.y));
    float t = fmaxf(-n.z, 0.f);

    glm::vec3 result = n;
    if (n.x >= 0.f) {
        result.x -= t;
    } else {
        result.x += t;
    }
    
    if (n.y >= 0.f) {
        result.y -= t;
    } else {
        result.y += t;
    }
    return glm::normalize(result);
}

#endif

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, 
#if OPTIMIZED_GBUFFER
                             glm::mat4 invViewMatrix,
#endif
                             GBufferPixel* gBuffer, GBufferPixelType type) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        float timeToIntersect = gBuffer[index].t * 256.0;
        pbo[index].w = 0;
#if OPTIMIZED_GBUFFER
        glm::vec3 pack = gBuffer[index].pack;
        glm::vec3 normal = decodeNormal(glm::vec2(pack.x, pack.y));
        float ndc_x = (2.0f * x) / resolution.x - 1.0f;
        float ndc_y = (2.0f * y) / resolution.y - 1.0f;
        glm::vec4 tmp = invViewMatrix * glm::vec4(ndc_x, ndc_y, pack.z, 1.0f);
        glm::vec3 position = glm::vec3(tmp.x, tmp.y, tmp.z) / tmp.w;
        if(type == GBUFFER_PIXEL_TYPE_TIME) {
            pbo[index].x = timeToIntersect;
            pbo[index].y = timeToIntersect;
            pbo[index].z = timeToIntersect;
        }
        else if(type == GBUFFER_PIXEL_TYPE_NORMAL) {
            glm::vec3 mapped = normal * 255.f;
            pbo[index].x = glm::clamp(abs(mapped.x), 0.f, 255.f);
            pbo[index].y = glm::clamp(abs(mapped.y), 0.f, 255.f);
            pbo[index].z = glm::clamp(abs(mapped.z), 0.f, 255.f);
        } else if(type == GBUFFER_PIXEL_TYPE_POSITION) {
            glm::vec3 mapped = position * 25.f;
            pbo[index].x = glm::clamp(abs(mapped.x), 0.f, 255.f);
            pbo[index].y = glm::clamp(abs(mapped.y), 0.f, 255.f);
            pbo[index].z = glm::clamp(abs(mapped.z), 0.f, 255.f);
        }
#else
        glm::vec3 position = gBuffer[index].position;
        glm::vec3 normal = gBuffer[index].normal;
        if(type == GBUFFER_PIXEL_TYPE_TIME) {
            pbo[index].x = timeToIntersect;
            pbo[index].y = timeToIntersect;
            pbo[index].z = timeToIntersect;
        }
        else if(type == GBUFFER_PIXEL_TYPE_NORMAL) {
            glm::vec3 mapped = normal * 255.f;
            pbo[index].x = glm::clamp(abs(mapped.x), 0.f, 255.f);
            pbo[index].y = glm::clamp(abs(mapped.y), 0.f, 255.f);
            pbo[index].z = glm::clamp(abs(mapped.z), 0.f, 255.f);
        } else if(type == GBUFFER_PIXEL_TYPE_POSITION) {
            glm::vec3 mapped = position * 25.f;
            pbo[index].x = glm::clamp(abs(mapped.x), 0.f, 255.f);
            pbo[index].y = glm::clamp(abs(mapped.y), 0.f, 255.f);
            pbo[index].z = glm::clamp(abs(mapped.z), 0.f, 255.f);
        }
#endif
    }
}

__global__ void generateGBuffer_kernel (
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
#if OPTIMIZED_GBUFFER
    glm::mat4 viewMatrix,
#endif
    GBufferPixel* gBuffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        gBuffer[idx].t = shadeableIntersections[idx].t;
        glm::vec3 position = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
        glm::vec3 normal = shadeableIntersections[idx].surfaceNormal;
#if OPTIMIZED_GBUFFER
        glm::vec4 tmp = viewMatrix * glm::vec4(position, 1.0f);
        glm::vec3 viewPos = glm::vec3(tmp.x, tmp.y, tmp.z) / tmp.w;
        glm::vec3 pack = glm::vec3(encodeNormal(normal), viewPos.z);
        gBuffer[idx].pack = pack;
#else
        gBuffer[idx].position = position;
        gBuffer[idx].normal = normal;
#endif
    }
}

__global__ void atrous(
    int num_paths,
    glm::vec3* input_image,
    glm::vec3* output_image,
    GBufferPixel* gBuffer,
    glm::ivec2 resolution,
    int stepwidth,
#if OPTIMIZED_GBUFFER
    glm::mat4 invViewMatrix,
#endif
    float c_phi, float n_phi, float p_phi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        int x = idx % resolution.x;
        int y = idx / resolution.x;
        float h[5] = { 1/16.f, 1/4.f, 3/8.f, 1/4.f, 1/16.f };

        glm::vec3 cval = input_image[idx];
#if OPTIMIZED_GBUFFER
        glm::vec3 pack = gBuffer[idx].pack;
        glm::vec3 nval = decodeNormal(glm::vec2(pack.x, pack.y));

        float ndc_x = (2.0f * x) / resolution.x - 1.0f;
        float ndc_y = (2.0f * y) / resolution.y - 1.0f;
        glm::vec4 tmp = invViewMatrix * glm::vec4(ndc_x, ndc_y, pack.z, 1.0f);
        glm::vec3 pval = glm::vec3(tmp.x, tmp.y, tmp.z) / tmp.w;
#else
        glm::vec3 nval = gBuffer[idx].normal;
        glm::vec3 pval = gBuffer[idx].position;
#endif

        glm::vec3 sum = glm::vec3(0.0f);
        float cum_w = 0.0f;
        for(int dx = -2; dx <= 2; dx++) 
        {
            for(int dy = -2; dy <= 2; dy++)
            {
                int n_x = x + stepwidth * dx;
                int n_y = y + stepwidth * dy;

                if(n_x < 0 || n_x >= resolution.x || n_y < 0 || n_y >= resolution.y) continue;   
                int n_index = n_x + n_y * resolution.x;
                glm::vec3 ctmp = input_image[n_index];
#if OPTIMIZED_GBUFFER
                glm::vec3 packtmp = gBuffer[n_index].pack;
                glm::vec3 ntmp = decodeNormal(glm::vec2(packtmp.x, packtmp.y));
                float ndc_x = (2.0f * n_x) / resolution.x - 1.0f;
                float ndc_y = (2.0f * n_y) / resolution.y - 1.0f;
                glm::vec4 tmp = invViewMatrix * glm::vec4(ndc_x, ndc_y, packtmp.z, 1.0f);
                glm::vec3 ptmp = glm::vec3(tmp.x, tmp.y, tmp.z) / tmp.w;
#else
                glm::vec3 ntmp = gBuffer[n_index].normal;
                glm::vec3 ptmp = gBuffer[n_index].position;
#endif

                glm::vec3 t = cval - ctmp;
                float dist2 = dot(t, t);
                float c_w = fminf(exp(-(dist2) / c_phi), 1.0);
                
                t = nval - ntmp;
                dist2 = fmaxf(dot(t, t) / (stepwidth * stepwidth), 0.0f);
                float n_w = fminf(exp(-(dist2) / n_phi), 1.0);

                t = pval - ptmp;
                dist2 = dot(t, t);
                float p_w = fminf(exp(-(dist2) / p_phi), 1.0);

                float kernel_weight = h[dx + 2] * h[dy + 2];
                float weight = c_w * n_w * p_w * kernel_weight;
                sum += ctmp * weight;
                cum_w += weight;
            }
        }

        if(cum_w > 0.0f) {
            output_image[idx] = sum / cum_w;
        } else {
            output_image[idx] = cval;
        }
    }
}

// Host functions
void denoiseInit(int pixelcount)
{
    cudaMalloc(&dev_image_temp, pixelcount * sizeof(glm::vec3));
    checkCUDAError("denoiseInit");
}

void denoiseFree()
{
    cudaFree(dev_image_temp);
    checkCUDAError("denoiseFree");
}

void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
#if OPTIMIZED_GBUFFER
    glm::mat4 viewMatrix,
#endif
    GBufferPixel* gBuffer)
{
    const int blockSize1d = 128;
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    
    generateGBuffer_kernel<<<numblocksPathSegmentTracing, blockSize1d>>> (
        num_paths, 
        shadeableIntersections, 
        pathSegments, 
#if OPTIMIZED_GBUFFER
        viewMatrix,
#endif
        gBuffer
    );
    checkCUDAError("generateGBuffer");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo, GBufferPixelType type, 
                 const Camera& cam,
#if OPTIMIZED_GBUFFER
                 glm::mat4 viewMatrix,
#endif
                 GBufferPixel* dev_gBuffer) {
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

#if OPTIMIZED_GBUFFER
    glm::mat4 invViewMatrix = glm::inverse(viewMatrix);
#endif

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(
        pbo, cam.resolution,
#if OPTIMIZED_GBUFFER
        invViewMatrix,
#endif
        dev_gBuffer, type
    );
    checkCUDAError("showGBuffer");
}

void applyDenoising(Camera cam, int filterSize, float c_phi, 
                    float n_phi, float p_phi,
#if OPTIMIZED_GBUFFER
                    glm::mat4 viewMatrix,
#endif
                    glm::vec3* dev_image,
                    GBufferPixel* dev_gBuffer) {
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int blockSize1d = 128;
    const dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

    glm::vec3* input = dev_image;
    glm::vec3* output = dev_image_temp;

#if OPTIMIZED_GBUFFER
    glm::mat4 invViewMatrix = glm::inverse(viewMatrix);
#endif

    const int numPasses = glm::ceil(glm::log2(filterSize));
    for(int pass = 0; pass < numPasses; pass++) {
        int stepwidth = 1 << pass;
        atrous<<<numBlocksPixels, blockSize1d>>>(
            pixelcount,
            input,
            output,
            dev_gBuffer,
            cam.resolution,
            stepwidth,
#if OPTIMIZED_GBUFFER
            invViewMatrix,
#endif
            c_phi, n_phi, p_phi
        );

        checkCUDAError("atrous denoising");
        cudaDeviceSynchronize();

        glm::vec3* temp = input;
        input = output;
        output = temp;
    }
    
    if(input != dev_image) {
        cudaMemcpy(dev_image, input, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        checkCUDAError("copy final denoised result");
    }
}
