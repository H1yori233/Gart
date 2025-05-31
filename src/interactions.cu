#include "interactions.h"
#include "random.h"
#include "texture.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng));      // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal + 
           cos(around) * over * perpendicularDirection1 + 
           sin(around) * over * perpendicularDirection2;
}

// Heavily Ref: https://github.com/cs87-dartmouth/darts-2024/blob/main/src/materials/material.cpp
__host__ __device__ float fresnelDielectric(float cos_theta_i, 
    float eta_i, float eta_t)
{
    // Using Sahl-Snell's law, calculate the squared sine of the angle between the normal and the transmitted ray
    float eta          = eta_i / eta_t;
    float sin_theta_t2 = eta * eta * (1 - cos_theta_i * cos_theta_i);

    // Total internal reflection!
    if (sin_theta_t2 > 1.0f)
        return 1.0f;

    float cos_theta_t = sqrtf(1.0f - sin_theta_t2);

    float Rs = (eta_i * cos_theta_i - eta_t * cos_theta_t) / 
               (eta_i * cos_theta_i + eta_t * cos_theta_t);
    float Rp = (eta_t * cos_theta_i - eta_i * cos_theta_t) / 
               (eta_t * cos_theta_i + eta_i * cos_theta_t);

    return 0.5f * (Rs * Rs + Rp * Rp);
}

__host__ __device__ void scatterRay(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    const glm::vec2 &uv,
    const DevTexturePool &texture_pool,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (m.hasReflective > 0.f)
    {
        auto wi = pathSegment.ray.direction;
        auto wo = glm::reflect(wi, normal);
        wo = glm::normalize(wo);

        pathSegment.ray.origin = intersect + EPSILON * wo;
        pathSegment.ray.direction = wo;
        pathSegment.color *= eval(m.color, uv, 0.0f, texture_pool);
    }
    else if (m.hasRefractive > 0.f)
    {
        auto wi = pathSegment.ray.direction;
        float cos_theta_i = -glm::dot(wi, normal);
        float sin_theta_i = sqrtf(1 - cos_theta_i * cos_theta_i);

        float ior, refl;
        if (cos_theta_i > 0)
        {
            ior = 1 / m.indexOfRefraction;
            refl = fresnelDielectric(cos_theta_i, 1.f, m.indexOfRefraction);
        }
        else
        {
            ior = m.indexOfRefraction;
            refl = fresnelDielectric(cos_theta_i, m.indexOfRefraction, 1.f);
        }

        // Schlick's approximation
        // float r0 = (1 - ior) / (1 + ior);
        // r0 = r0 * r0;
        // float refl = r0 + (1 - r0) * std::pow((1 - cos_theta_i), 5);
        
        auto wo = wi;
        if ((ior * sin_theta_i > 1) || (u01(rng) < refl)) 
        {
            wo = glm::reflect(wi, normal);
        }
        else 
        {
            wo = glm::refract(wi, normal, ior);
        }
        wo = glm::normalize(wo);
        
        // Very Strange, EPSILON = 0.00001f wont work
        // only 0.0001f works.
        pathSegment.ray.origin = intersect + 0.0001f * wo;
        pathSegment.ray.direction = wo;
        pathSegment.color *= eval(m.color, uv, 0.0f, texture_pool);
    }
    else
    {
        // auto wo = calculateRandomDirectionInHemisphere(normal, rng);
        // wo = glm::normalize(wo); 
        auto lo     = randomHemisphereCosine(u01(rng), u01(rng));

        auto onb    = ONB(normal);
        auto wo     = glm::normalize(onb.localToWorld(lo));

        pathSegment.ray.origin = intersect + EPSILON * wo;
        pathSegment.ray.direction = wo;

        // * pdf and then / pdf, so ignore it
        pathSegment.color *= eval(m.color, uv, 0.0f, texture_pool);
    }

    pathSegment.remainingBounces--;
}
