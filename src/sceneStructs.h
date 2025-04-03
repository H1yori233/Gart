#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle
{
    glm::vec3 v0, v1, v2;
    glm::vec3 n0, n1, n2;
    glm::vec2 t0, t1, t2;
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
    Triangle triangle;
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
    float lensRadius, focalDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
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

// Heavily Ref: https://cs87-dartmouth.github.io/Fall2024/assignment4.html
struct ONB
{
    // Three ortho-normal vectors that form the basis for a local coordinate system
    glm::vec3 s; // The tangent vector
    glm::vec3 t; // The bi-tangent vector
    glm::vec3 n; // The normal vector

    __host__ __device__ ONB();

    /*
        Build an ONB from a single vector.

        Sets ONB::n to a normalized version of \p n_ and computes ONB::s and ONB::t automatically to form a right-handed
        orthonormal basis
    */
    __host__ __device__ ONB(const glm::vec3 n_);

    /*
        Initialize an ONB from a surface tangent \p s and normal \p n.

        \param [in] s_   Surface tangent
        \param [in] n_   Surface normal
    */
    __host__ __device__ ONB(const glm::vec3 &s_, const glm::vec3 &n_);

    // Initialize an ONB from three orthonormal vectors
    __host__ __device__ ONB(const glm::vec3 &s, const glm::vec3 &t, const glm::vec3 &n);

    // Convert from world coordinates to local coordinates
    __host__ __device__ glm::vec3 toLocal(const glm::vec3 &v) const;

    // Convert from local coordinates to world coordinates
    __host__ __device__ glm::vec3 toWorld(const glm::vec3 &v) const;
};

