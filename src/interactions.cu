#include "interactions.h"
#include "utilities.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
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

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    if(m.hasReflective > .0f) {
        auto wo = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.direction = wo;
    } else {
        auto wo = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.direction = wo;
    }

    pathSegment.ray.origin = intersect + .0001f * normal; // prevent self-intersection
    pathSegment.color *= m.color;
}


// ------------------------------------------------------------

__host__ __device__ int sampleLight(
    const LightInfo* lightInfos,
    int numLights,
    float totalPower,
    thrust::default_random_engine& rng,
    float& selectionPdf)
{
    if (numLights == 0 || totalPower <= 0.0f) {
        selectionPdf = 0.0f;
        return -1;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    
    // Importance Sampling based on Light Power
    float randomPower = u01(rng) * totalPower;
    float cumulativePower = 0.0f;
    for (int i = 0; i < numLights; i++) {
        cumulativePower += lightInfos[i].power;
        if (randomPower <= cumulativePower) {
            selectionPdf = lightInfos[i].power / totalPower;
            return i;
        }
    }

    // If randomPower is greater than totalPower
    // return the last light
    selectionPdf = lightInfos[numLights - 1].power / totalPower;
    return numLights - 1;
}

__host__ __device__ glm::vec3 samplePointOnGeom(
    const Geom& geom,
    glm::vec3 shadingPoint,
    thrust::default_random_engine& rng,
    glm::vec3& lightNormal,
    float& areaPdf)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (geom.type == GeomType::SPHERE) {
        glm::vec3 center = glm::vec3(geom.transform * glm::vec4(0, 0, 0, 1));
        float radius = glm::length(glm::vec3(geom.transform * glm::vec4(1, 0, 0, 0)));

        // uniform sampling on the surface of the sphere
        float z = 1.0f - 2.0f * u01(rng);
        float r = sqrt(1.0f - z * z);
        float phi = 2.0f * PI * u01(rng);

        glm::vec3 localPoint(r * cos(phi), r * sin(phi), z);
        glm::vec3 worldPoint = center + radius * localPoint;
        lightNormal = glm::normalize(localPoint);

        float area = 4.0f * PI * radius * radius;
        areaPdf = 1.0f / area;
        return worldPoint;
    } else if (geom.type == CUBE) {
        glm::vec3 scaleVec = glm::vec3(
            glm::length(glm::vec3(geom.transform[0])),
            glm::length(glm::vec3(geom.transform[1])),
            glm::length(glm::vec3(geom.transform[2]))
        );

        // Check if this is a thin/planar light (same logic as scene.cpp)
        float min_dim = glm::min(scaleVec.x, glm::min(scaleVec.y, scaleVec.z));
        float max_dim = glm::max(scaleVec.x, glm::max(scaleVec.y, scaleVec.z));
        float ratio = min_dim / max_dim;

        if (ratio < 0.1f) {
            // Thin cube - sample from the thinnest face
            float u = u01(rng);
            float v = u01(rng);
            
            glm::vec3 localPoint;
            glm::vec4 localNormal4;
            float area;
            
            if (scaleVec.x == min_dim) {
                // Sample from YZ face (thinnest in X direction)
                localPoint = glm::vec3(-0.5f, -0.5f + u, -0.5f + v);
                localNormal4 = glm::vec4(-1, 0, 0, 0);
                area = scaleVec.y * scaleVec.z;
            } else if (scaleVec.y == min_dim) {
                // Sample from XZ face (thinnest in Y direction) 
                localPoint = glm::vec3(-0.5f + u, -0.5f, -0.5f + v);
                localNormal4 = glm::vec4(0, -1, 0, 0);
                area = scaleVec.x * scaleVec.z;
            } else {
                // Sample from XY face (thinnest in Z direction)
                localPoint = glm::vec3(-0.5f + u, -0.5f + v, -0.5f);
                localNormal4 = glm::vec4(0, 0, -1, 0);
                area = scaleVec.x * scaleVec.y;
            }
            
            glm::vec4 worldPoint4 = geom.transform * glm::vec4(localPoint, 1.0f);
            lightNormal = glm::normalize(glm::vec3(geom.invTranspose * localNormal4));
            areaPdf = 1.0f / area;
            return glm::vec3(worldPoint4);
        } else {
            // Full cube - sample from all 6 faces based on their relative areas
            float total_area = 2.0f * (scaleVec.x * scaleVec.y + 
                                       scaleVec.y * scaleVec.z + 
                                       scaleVec.x * scaleVec.z);
            
            // Calculate cumulative areas for each face pair
            float area_xy = scaleVec.x * scaleVec.y;
            float area_yz = scaleVec.y * scaleVec.z;
            float area_xz = scaleVec.x * scaleVec.z;
            
            float cum_xy = 2.0f * area_xy;
            float cum_yz = cum_xy + 2.0f * area_yz;
            float cum_xz = cum_yz + 2.0f * area_xz;
            
            float r = u01(rng) * total_area;
            float u = u01(rng);
            float v = u01(rng);
            
            glm::vec3 localPoint;
            glm::vec4 localNormal4;
            
            if (r < cum_xy) {
                // Sample from XY faces (top/bottom)
                if (r < area_xy) {
                    // Bottom face (z = -0.5)
                    localPoint = glm::vec3(-0.5f + u, -0.5f + v, -0.5f);
                    localNormal4 = glm::vec4(0, 0, -1, 0);
                } else {
                    // Top face (z = 0.5)
                    localPoint = glm::vec3(-0.5f + u, -0.5f + v, 0.5f);
                    localNormal4 = glm::vec4(0, 0, 1, 0);
                }
            } else if (r < cum_yz) {
                // Sample from YZ faces (left/right)
                if (r < cum_xy + area_yz) {
                    // Left face (x = -0.5)
                    localPoint = glm::vec3(-0.5f, -0.5f + u, -0.5f + v);
                    localNormal4 = glm::vec4(-1, 0, 0, 0);
                } else {
                    // Right face (x = 0.5)
                    localPoint = glm::vec3(0.5f, -0.5f + u, -0.5f + v);
                    localNormal4 = glm::vec4(1, 0, 0, 0);
                }
            } else {
                // Sample from XZ faces (front/back)
                if (r < cum_yz + area_xz) {
                    // Front face (y = -0.5)
                    localPoint = glm::vec3(-0.5f + u, -0.5f, -0.5f + v);
                    localNormal4 = glm::vec4(0, -1, 0, 0);
                } else {
                    // Back face (y = 0.5)
                    localPoint = glm::vec3(-0.5f + u, 0.5f, -0.5f + v);
                    localNormal4 = glm::vec4(0, 1, 0, 0);
                }
            }
            
            glm::vec4 worldPoint4 = geom.transform * glm::vec4(localPoint, 1.0f);
            lightNormal = glm::normalize(glm::vec3(geom.invTranspose * localNormal4));
            areaPdf = 1.0f / total_area;
            return glm::vec3(worldPoint4);
        }
    } else if (geom.type == QUAD) {
        float u = u01(rng);
        float v = u01(rng);

        // Sample uniformly on the unit quad [-0.5, 0.5] x [-0.5, 0.5] at z=0
        glm::vec3 localPoint(-0.5f + u, -0.5f + v, 0.0f);
        glm::vec4 worldPoint4 = geom.transform * glm::vec4(localPoint, 1.0f);
        glm::vec4 localNormal4(0, 0, 1, 0); // Normal pointing in +z direction
        lightNormal = glm::normalize(glm::vec3(geom.invTranspose * localNormal4));
        
        // Calculate area using cross product of transformed edges
        glm::vec3 edge1 = glm::vec3(geom.transform * glm::vec4(1, 0, 0, 0)); // X direction
        glm::vec3 edge2 = glm::vec3(geom.transform * glm::vec4(0, 1, 0, 0)); // Y direction
        float area = glm::length(glm::cross(edge1, edge2));
        areaPdf = 1.0f / area;
        return glm::vec3(worldPoint4);
    } else if (geom.type == TRIANGLE) {
        float u = u01(rng);
        float v = u01(rng);
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }

        glm::vec3 point = geom.triangle.v0 + 
                          u * (geom.triangle.v1 - geom.triangle.v0) + 
                          v * (geom.triangle.v2 - geom.triangle.v0);
        if (geom.triangle.hasVertexNormals) {
            // interpolate normal
            lightNormal = glm::normalize(
                geom.triangle.n0 + 
                u * (geom.triangle.n1 - geom.triangle.n0) + 
                v * (geom.triangle.n2 - geom.triangle.n0)
            );
        } else {
            // compute normal
            lightNormal = glm::normalize(glm::cross(
                geom.triangle.v1 - geom.triangle.v0,
                geom.triangle.v2 - geom.triangle.v0
            ));
        }

        float area = 0.5f * glm::length(glm::cross(
            geom.triangle.v1 - geom.triangle.v0,
            geom.triangle.v2 - geom.triangle.v0
        ));
        areaPdf = 1.0f / area;
        return point;
    }
    
    areaPdf = 1.0f;
    lightNormal = glm::vec3(0, 1, 0);
    return glm::vec3(0.0f);
}

__device__ bool shadowTest(Ray ray, Geom* geoms, int geoms_size, float max_distance) {
    for (int i = 0; i < geoms_size; i++) {
        float t;
        glm::vec3 tmp_intersect, tmp_normal;
        bool outside;
        
        if (geoms[i].type == SPHERE) {
            t = sphereIntersectionTest(geoms[i], ray, tmp_intersect, tmp_normal, outside);
        } else if (geoms[i].type == CUBE) {
            t = boxIntersectionTest(geoms[i], ray, tmp_intersect, tmp_normal, outside);
        } else if (geoms[i].type == QUAD) {
            t = quadIntersectionTest(geoms[i], ray, tmp_intersect, tmp_normal, outside);
        } else if (geoms[i].type == TRIANGLE) {
            t = triangleIntersectionTest(geoms[i], ray, tmp_intersect, tmp_normal, outside);
        }
        
        if (t > 0.001f && t < max_distance - 0.001f) {
            return true;
        }
    }
    return false;
}

// ------------------------------------------------------------

__host__ __device__ glm::vec3 sampleBSDF(
    const Material& material,
    glm::vec3 wi,
    glm::vec3 normal,
    thrust::default_random_engine& rng,
    glm::vec3& wo,
    float& pdf)
{
    if (material.hasReflective > 0.0f) {
        wo = glm::reflect(-wi, normal);
        pdf = 1.0f; // delta should be infinite, but we use 1.0f to represent it
        
        float cosTheta = glm::dot(wo, normal);
        if (cosTheta > 0.0f) {
            return material.color * material.hasReflective;
        } else {
            pdf = 0.0f;
            return glm::vec3(0.0f);
        }
    } else {
        wo = calculateRandomDirectionInHemisphere(normal, rng);
        float cosTheta = glm::dot(wo, normal);
        if (cosTheta > 0.0f) {
            pdf = cosTheta / PI;
            return material.color / PI;
        } else {
            pdf = 0.0f;
            return glm::vec3(0.0f);
        }
    }
}

__host__ __device__ glm::vec3 evaluateBSDF(
    const Material& material,
    glm::vec3 wi,
    glm::vec3 wo, 
    glm::vec3 normal)
{
    float cosTheta = glm::dot(wo, normal);
    if (cosTheta <= 0.0f) return glm::vec3(0.0f);
    
    if (material.hasReflective > 0.0f) {
        glm::vec3 reflection = glm::reflect(-wi, normal);
        float dotProduct = glm::dot(wo, reflection);
        if (dotProduct > 0.999f) {
            return material.color * material.hasReflective;
        }
        return glm::vec3(0.0f);
    } else {
        return material.color / PI;
    }
}

__host__ __device__ float pdfBSDF(
    const Material& material,
    glm::vec3 wi,
    glm::vec3 wo,
    glm::vec3 normal)
{
    float cosTheta = glm::dot(wo, normal);
    if (cosTheta <= 0.0f) return 0.0f;
    
    if (material.hasReflective > 0.0f) {
        return 0.0f;
    } else {
        return cosTheta / PI;
    }
}
