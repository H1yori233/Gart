#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

// This part comes from PBRT v4
__host__ __device__ inline int permutationElement(uint32_t i, uint32_t l, uint32_t p)
{
    uint32_t w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    do {
        i ^= p;
        i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | p >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);
    return (i + p) % l;
}

__host__ __device__ inline glm::vec2 randomCircle(float rv)
{
    auto angle = 2.f * PI * rv;
    return glm::vec2(glm::cos(angle), glm::sin(angle));
}

__host__ __device__ inline float pdfCircle()
{
    return 0.5f / PI;
}

__host__ __device__ inline glm::vec2 randomDisk(float a, float b)
{
    auto phi = a * PI * 2;
    auto r = sqrtf(b);
    return glm::vec2(glm::cos(phi) * r, glm::sin(phi) * r);
}

__host__ __device__ inline float pdfDisk(const glm::vec2 &p)
{
    return 1.f / PI;
}

__host__ __device__ inline glm::vec3 randomSphere(float a, float b)
{
    float z = 1.f - 2.f * b;
    float r = sqrtf(glm::max(0.f, 1.f - z * z));
    glm::vec2 circle = randomCircle(a);
    return glm::vec3(circle.x * r, circle.y * r, z);
}

__host__ __device__ inline float pdfSphere()
{
    return 0.25f / PI;
}

__host__ __device__ inline glm::vec3 randomHemisphere(float a, float b)
{
    auto cos_theta = b;
    auto sin_theta = sqrtf(glm::max(0.f, 1.f - cos_theta * cos_theta));
    glm::vec2 circle = randomCircle(a);
    return glm::vec3(circle.x * sin_theta, circle.y * sin_theta, cos_theta);
}

__host__ __device__ inline float pdfHemisphere(const glm::vec3 &v)
{
    return 0.5f / PI;
}

__host__ __device__ inline glm::vec3 randomHemisphereCosine(float a, float b)
{
    auto phi = a * PI * 2;
    auto r = sqrtf(b);
    return glm::vec3(glm::cos(phi) * r, glm::sin(phi) * r, sqrtf(glm::max(0.f, 1.f - b)));
}

__host__ __device__ inline float pdfHemisphereCosine(const glm::vec3 &v)
{
    return v.z / PI;
}

__host__ __device__ inline glm::vec3 randomHemisphereCosinePower(float exponent, float a, float b)
{
    auto phi = a * PI * 2;
    auto cos_theta = std::pow(b, 1.f / (exponent + 1.f));
    auto r = sqrtf(glm::max(0.f, 1.f - cos_theta * cos_theta));
    return glm::vec3(glm::cos(phi) * r, glm::sin(phi) * r, cos_theta);
}

__host__ __device__ inline float pdfHemisphereCosinePower(float exponent, float cosine)
{
    return (exponent + 1.f) * (1.f / (2.f * PI)) * powf(cosine, exponent);
}

__host__ __device__ inline glm::vec3 randomSphereCap(float cos_theta_max, float a, float b)
{
    auto phi = a * PI * 2;
    auto cos_theta = glm::mix(cos_theta_max, 1.f, b);
    auto r = sqrtf(glm::max(0.f, 1.f - cos_theta * cos_theta));
    return glm::vec3(glm::cos(phi) * r, glm::sin(phi) * r, cos_theta);
}

__host__ __device__ inline float pdfSphereCap(float cos_theta_max)
{
    return (0.5f / PI) / (1.f - cos_theta_max);
}

__host__ __device__ inline glm::vec2 randomTriangle(float a, float b)
{
    auto b0 = a;
    auto b1 = b;

    if(b0 + b1 > 1.f)
    {
        b0 = 1.f - b0;
        b1 = 1.f - b1;
    }

    return glm::vec2(b0, b1);
}

__host__ __device__ inline glm::vec3 randomTriangle(const glm::vec3 &v0, 
    const glm::vec3 &v1, const glm::vec3 &v2, float a, float b)
{
    auto b0 = a;
    auto b1 = b;

    if(b0 + b1 > 1.f)
    {
        b0 = 1.f - b0;
        b1 = 1.f - b1;
    }

    auto b2 = 1 - b0 - b1;
    return b0 * v0 + b1 * v1 + b2 * v2;
}

__host__ __device__ inline float pdfTriangle(const glm::vec3 &v0, 
    const glm::vec3 &v1, const glm::vec3 &v2)
{
    float area = 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
    return 1.f / area;
}
