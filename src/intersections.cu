#include "intersections.h"
#include "random.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

// __host__ __device__ glm::vec3 boxSample(
//     Geom box,
//     Ray& r,
//     Material material,
//     thrust::default_random_engine& rng,
//     float& pdf)
// {
//     return glm::vec3(0.f, 0.f, 0.f);
// }

// __host__ __device__ float boxPDF(
//     Geom box,
//     Ray r,
//     glm::vec3& intersectionPoint,
//     glm::vec3& normal,
//     bool& outside)
// {
//     float t = boxIntersectionTest(box, r, intersectionPoint, normal, outside);
//     return 0.0f;
// }

// --------------------------------------------------------------- //

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// __host__ __device__ glm::vec3 sphereSample(
//     Geom sphere,
//     Ray& r,
//     Material material,
//     thrust::default_random_engine& rng,
//     float& pdf)
// {
//     thrust::uniform_real_distribution<float> u01(0, 1);

//     glm::vec3 center = multiplyMV(sphere.transform, glm::vec4(0, 0, 0, 1.0f));
//     glm::vec3 oc = center - r.origin;
//     float d = glm::length(oc);
//     float radius = 0.5f;
//     float transformedRadius = radius * 
//         glm::length(multiplyMV(sphere.transform, glm::vec4(radius, 0, 0, 0.0f)));
    
//     float c = d * d - transformedRadius * transformedRadius;
//     float cos_theta_max = c < 0.f ? -1.f : sqrt(c) / d;
//     glm::vec3 lo = randomSphereCap(cos_theta_max, u01(rng), u01(rng));

//     ONB onb = ONB(glm::normalize(oc));
//     r.direction = glm::normalize(onb.localToWorld(lo));
//     pdf = pdfSphereCap(cos_theta_max);

//     return material.color * material.emittance / pdf;
// }

// __host__ __device__ float spherePDF(
//     Geom sphere,
//     Ray r,
//     glm::vec3& intersectionPoint,
//     glm::vec3& normal,
//     bool& outside)
// {
//     float t = sphereIntersectionTest(sphere, r, intersectionPoint, normal, outside);
//     if (t > 0) 
//     {
//         glm::vec3 center = multiplyMV(sphere.transform, glm::vec4(0, 0, 0, 1.0f));
//         glm::vec3 oc = center - r.origin;
//         float d = glm::length(oc);
//         float radius = 0.5f;
//         float transformedRadius = radius * 
//             glm::length(multiplyMV(sphere.transform, glm::vec4(radius, 0, 0, 0.0f)));
        
//         float c = d * d - transformedRadius * transformedRadius;
//         float cos_theta_max = c < 0.f ? -1.f  : sqrt(c) / d;
//         return pdfSphereCap(cos_theta_max);
//     }
    
//     return 0.f;
// }

// --------------------------------------------------------------- //

__host__ __device__ float triangleIntersectionTest(
    Geom triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{
    auto tri = triangle.triangle;
    auto S  = r.origin  - tri.v0;
    auto E1 = tri.v1    - tri.v0;
    auto E2 = tri.v2    - tri.v0;

    //   t  = det([S, E1, E2])  / det(-d, E1, E2)
    //      = (S x E1) * E2     / E1 * (d x E2) 
    auto S1     = glm::cross(r.direction, E2);
    auto S2     = glm::cross(S, E1);
    float S1E1  = glm::dot(S1, E1);
    if (glm::abs(S1E1) < 1e-6)
    {
        return -1.0f;
    }

    float t     = glm::dot(S2, E2)  / S1E1;
    float b1    = glm::dot(S1, S)   / S1E1;
    float b2    = glm::dot(S2, r.direction) / S1E1;
    float b0    = 1 - b1 - b2;
    uv    = b0 * tri.t0 + b1 * tri.t1 + b2 * tri.t2;
    if (t < 0.0f || b0 < 0.f || b1 < 0.f || b2 < 0.f) 
    {
        return -1.0f;
    }

    intersectionPoint = r.origin + t * r.direction;
    normal = glm::normalize(glm::cross(E1, E2));
    outside = (glm::dot(r.direction, normal) < 0.0f);
    // if (!outside) 
    // {
    //     normal = -normal;
    // }
    
    if (tri.n0 != glm::vec3(0.0f) || 
        tri.n1 != glm::vec3(0.0f) || 
        tri.n2 != glm::vec3(0.0f)) 
    {
        glm::vec3 shadingNormal = glm::normalize(b0 * tri.n0 + 
            b1 * tri.n1 + b2 * tri.n2);
        // if (glm::dot(normal, shadingNormal) < 0.0f) 
        // {
        //     shadingNormal = -shadingNormal;
        // }
        normal = shadingNormal;
    }
    
    return t;
}

// __host__ __device__ glm::vec3 triangleSample(
//     Geom triangle,
//     Ray& r,
//     Material material,
//     thrust::default_random_engine& rng,
//     float& pdf)
// {
//     thrust::uniform_real_distribution<float> u01(0, 1);
    
//     Triangle tri = triangle.triangle;
//     glm::vec3 v0 = tri.v0;
//     glm::vec3 v1 = tri.v1;
//     glm::vec3 v2 = tri.v2;

//     glm::vec3 E1 = v1 - v0;
//     glm::vec3 E2 = v2 - v0;
//     glm::vec3 normal = glm::normalize(glm::cross(E1, E2));
//     glm::vec3 samplePoint = randomTriangle(v0, v1, v2, u01(rng), u01(rng));
    
//     glm::vec3 direction = samplePoint - r.origin;
//     float distanceSquared = glm::dot(direction, direction);
//     float distance = sqrtf(distanceSquared);
//     r.direction = direction / distance;

//     float pdfArea = pdfTriangle(v0, v1, v2);
//     float cosine = glm::abs(glm::dot(normal, r.direction));
//     pdf = distanceSquared / (cosine * (1.0f / pdfArea));

//     return material.color * material.emittance / pdf;
// }

// __host__ __device__ float trianglePDF(
//     Geom triangle,
//     Ray r,
//     glm::vec3& intersectionPoint,
//     glm::vec3& normal,
//     bool& outside)
// {
//     float t = triangleIntersectionTest(triangle, r, intersectionPoint, normal, outside);
//     if (t > 0) 
//     {
//         Triangle tri = triangle.triangle;
//         glm::vec3 v0 = tri.v0;
//         glm::vec3 v1 = tri.v1;
//         glm::vec3 v2 = tri.v2;
//         float pdfArea = pdfTriangle(v0, v1, v2);
        
//         float distanceSquared = t * t;
//         float cosine = glm::abs(glm::dot(glm::normalize(r.direction), normal));
//         return distanceSquared / (cosine * (1.0f / pdfArea));
//     }
//     return 0.0f;
// }
