#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define ERRORCHECK 1
#define SORT_BY_MATERIAL 1  // Toggle for material-based sorting optimization
#define RUSSIAN_ROULETTE 1  // Toggle for Russian Roulette
#define RR_DEPTH 5
#define OPTIMIZED_GBUFFER 1

enum IntegratorType {
    INTEGRATOR_FAKE = 0,
    INTEGRATOR_NAIVE = 1,
    INTEGRATOR_NEE = 2,
    INTEGRATOR_COUNT  // for array size
};

static const char* INTEGRATOR_NAMES[INTEGRATOR_COUNT] = {
    "fake",
    "naive", 
    "nee"
};

enum GeomType
{
    SPHERE,
    CUBE,
    QUAD,
    TRIANGLE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle
{
    glm::vec3 v0, v1, v2; // vertices
    glm::vec3 n0, n1, n2; // normals at vertices (for smooth shading)
    glm::vec2 t0, t1, t2; // texture coordinates
    bool hasVertexNormals; // whether to use vertex normals for smooth shading
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    
    // For triangle geometry
    Triangle triangle;
};

struct LightInfo {
    int geomid;
    float power;
    float area;
    glm::vec3 emission;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
    IntegratorType integrator;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    int materialId;
};

// CHECKITOUT - a simple struct for storing scene geometry information per-pixel.
// What information might be helpful for guiding a denoising filter?
struct GBufferPixel {
    float t;
#if OPTIMIZED_GBUFFER
    glm::vec3 pack;
#else
    glm::vec3 position;
    glm::vec3 normal;
#endif
};

enum GBufferPixelType {
    GBUFFER_PIXEL_TYPE_TIME,
    GBUFFER_PIXEL_TYPE_NORMAL,
    GBUFFER_PIXEL_TYPE_POSITION,
};
