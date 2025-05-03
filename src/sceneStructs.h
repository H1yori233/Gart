#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
// #include "texture.h"

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


struct ONB
{
    // Three orthonormal basis vectors stored in an array
    glm::vec3 axis[3];

    __host__ __device__ inline ONB() {}
    __host__ __device__ inline ONB(const glm::vec3& normal) 
    {
        axis[2] = glm::normalize(normal);
        glm::vec3 a = (std::fabs(axis[2].x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        axis[1] = glm::normalize(glm::cross(axis[2], a));
        axis[0] = glm::cross(axis[2], axis[1]);
    }

    __host__ __device__ inline const glm::vec3& tangent() const { return axis[0]; }
    __host__ __device__ inline const glm::vec3& bitangent() const { return axis[1]; }
    __host__ __device__ inline const glm::vec3& normal() const { return axis[2]; }

    __host__ __device__ inline glm::vec3 localToWorld(const glm::vec3& a) const 
    {
        return a.x * axis[0] + a.y * axis[1] + a.z * axis[2];
    }

    __host__ __device__ inline glm::vec3 worldToLocal(const glm::vec3& a) const 
    {
        return glm::vec3(
            glm::dot(a, axis[0]),
            glm::dot(a, axis[1]),
            glm::dot(a, axis[2])
        );
    }
};

