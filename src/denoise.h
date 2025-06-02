#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "sceneStructs.h"

// Forward declarations
void denoiseInit(int pixelcount);
void denoiseFree();

void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
#if OPTIMIZED_GBUFFER
    glm::mat4 viewMatrix,
#endif
    GBufferPixel* gBuffer);

void showGBuffer(uchar4* pbo, GBufferPixelType type, 
                 const Camera& cam,
#if OPTIMIZED_GBUFFER
                 glm::mat4 viewMatrix,
#endif
                 GBufferPixel* dev_gBuffer);

void applyDenoising(Camera cam, int filterSize, float c_phi, 
                    float n_phi, float p_phi,
#if OPTIMIZED_GBUFFER
                    glm::mat4 viewMatrix,
#endif
                    glm::vec3* dev_image,
                    GBufferPixel* dev_gBuffer);

// Utility functions for octahedron normal compression
#if OPTIMIZED_GBUFFER
__host__ __device__ glm::vec2 encodeNormal(glm::vec3 normal);
__host__ __device__ glm::vec3 decodeNormal(glm::vec2 encoded);
#endif 