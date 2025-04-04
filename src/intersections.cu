#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
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

__host__ __device__ glm::vec3 boxSample(
    Geom box,
    Ray& r,
    Material material,
    thrust::default_random_engine& rng,
    float& pdf)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    int face = min(int(u01(rng) * 6), 5);

    float u1 = u01(rng) * 2.0f - 1.0f;
    float u2 = u01(rng) * 2.0f - 1.0f;

    float size = 0.5f;
    glm::vec3 localPoint;
    switch (face) {
        case 0: // +x
            localPoint = glm::vec3(size, u1 * size, u2 * size);
            break;
        case 1: // -x
            localPoint = glm::vec3(-size, u1 * size, u2 * size);
            break;
        case 2: // +y
            localPoint = glm::vec3(u1 * size, size, u2 * size);
            break;
        case 3: // -y
            localPoint = glm::vec3(u1 * size, -size, u2 * size);
            break;
        case 4: // +z
            localPoint = glm::vec3(u1 * size, u2 * size, size);
            break;
        case 5: // -z
            localPoint = glm::vec3(u1 * size, u2 * size, -size);
            break;
    }
    
    glm::vec3 worldPoint = multiplyMV(box.transform, glm::vec4(localPoint, 1.0f));
    glm::vec3 direction = worldPoint - r.origin;
    float distanceSquared = glm::dot(direction, direction);
    float distance = sqrtf(distanceSquared);
    
    r.direction = direction / distance;
    glm::vec3 localNormal;
    switch (face) {
        case 0: localNormal = glm::vec3(1, 0, 0); break;
        case 1: localNormal = glm::vec3(-1, 0, 0); break;
        case 2: localNormal = glm::vec3(0, 1, 0); break;
        case 3: localNormal = glm::vec3(0, -1, 0); break;
        case 4: localNormal = glm::vec3(0, 0, 1); break;
        case 5: localNormal = glm::vec3(0, 0, -1); break;
    }
    glm::vec3 worldNormal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(localNormal, 0.0f)));
    
    float area = 0.0f;
    glm::vec3 rightVec      = multiplyMV(box.transform, 
        glm::vec4(1, 0, 0, 0)) - multiplyMV(box.transform, glm::vec4(0, 0, 0, 0));
    glm::vec3 upVec         = multiplyMV(box.transform, 
        glm::vec4(0, 1, 0, 0)) - multiplyMV(box.transform, glm::vec4(0, 0, 0, 0));
    glm::vec3 forwardVec    = multiplyMV(box.transform, 
        glm::vec4(0, 0, 1, 0)) - multiplyMV(box.transform, glm::vec4(0, 0, 0, 0));
    float areaXY = glm::length(glm::cross(rightVec, upVec));
    float areaXZ = glm::length(glm::cross(rightVec, forwardVec));
    float areaYZ = glm::length(glm::cross(upVec, forwardVec));
    
    area = 2.0f * (areaXY + areaXZ + areaYZ);
    float cosine = glm::abs(glm::dot(worldNormal, r.direction));
    pdf = distanceSquared / (cosine * area);
    
    return material.color * material.emittance / pdf;
}

__host__ __device__ float boxPDF(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float t = boxIntersectionTest(box, r, intersectionPoint, normal, outside);
    if (t > 0) 
    {
        glm::vec3 rightVec      = multiplyMV(box.transform, 
            glm::vec4(1, 0, 0, 0)) - multiplyMV(box.transform, glm::vec4(0, 0, 0, 0));
        glm::vec3 upVec         = multiplyMV(box.transform, 
            glm::vec4(0, 1, 0, 0)) - multiplyMV(box.transform, glm::vec4(0, 0, 0, 0));
        glm::vec3 forwardVec    = multiplyMV(box.transform, 
            glm::vec4(0, 0, 1, 0)) - multiplyMV(box.transform, glm::vec4(0, 0, 0, 0));
        
        float areaXY = glm::length(glm::cross(rightVec, upVec));
        float areaXZ = glm::length(glm::cross(rightVec, forwardVec));
        float areaYZ = glm::length(glm::cross(upVec, forwardVec));
        float area = 2.0f * (areaXY + areaXZ + areaYZ);

        float distanceSquared = t * t;
        float cosine = glm::abs(glm::dot(glm::normalize(r.direction), normal));
        return distanceSquared / (cosine * area);
    }
    return 0.0f;
}

// --------------------------------------------------------------- //

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
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

__host__ __device__ glm::vec3 sphereSample(
    Geom sphere,
    Ray& r,
    Material material,
    thrust::default_random_engine& rng,
    float& pdf)
{
    glm::vec3 center = multiplyMV(sphere.transform, glm::vec4(0, 0, 0, 1.0f));
    glm::vec3 oc = center - r.origin;
    
    float d = glm::length(oc);
    float radius = 0.5f;
    float transformedRadius = radius * 
        glm::length(multiplyMV(sphere.transform, glm::vec4(radius, 0, 0, 0.0f)));
    
    float c = d * d - transformedRadius * transformedRadius;
    float cos_theta_max = c < 0.f ? -1.f : sqrt(c) / d;

    // sample sphere cap
    thrust::uniform_real_distribution<float> u01(0, 1);
    float phi = 2.0f * PI * u01(rng);
    float t = u01(rng);
    float cos_theta = cos_theta_max * (1.0f - t) + 1.0f * t;
    float r_ = sqrtf(glm::max(0.f, 1.f - cos_theta * cos_theta));
    glm::vec3 lo = glm::vec3(glm::cos(phi) * r_, glm::sin(phi) * r_, cos_theta);

    ONB onb = ONB(glm::normalize(oc));
    r.direction = glm::normalize(onb.toWorld(lo));
    pdf = 1.0f / (2.0f * PI * (1.0f - cos_theta_max));

    return material.color * material.emittance / pdf;
}

__host__ __device__ float spherePDF(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float t = sphereIntersectionTest(sphere, r, intersectionPoint, normal, outside);
    if (t > 0) 
    {
        glm::vec3 center = multiplyMV(sphere.transform, glm::vec4(0, 0, 0, 1.0f));
        glm::vec3 oc = center - r.origin;

        float d = glm::length(oc);
        float radius = 0.5f;
        float transformedRadius = radius * 
            glm::length(multiplyMV(sphere.transform, glm::vec4(radius, 0, 0, 0.0f)));
        
        float c = d * d - transformedRadius * transformedRadius;
        float cos_theta_max = c < 0.f ? -1.f  : sqrt(c) / d;
        return 1.f / (2.f * PI * (1.f  - cos_theta_max));
    }
    
    return 0.f;
}

// --------------------------------------------------------------- //

__host__ __device__ float triangleIntersectionTest(
    Geom triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
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

    // First, check for intersection and fill in the hit distance t
    float t     = glm::dot(S2, E2)  / S1E1;
    float b1    = glm::dot(S1, S)   / S1E1;
    float b2    = glm::dot(S2, r.direction) / S1E1;

    // You should also compute the u/v (i.e. the beta/gamma barycentric coordinates) of the hit point
    // (Moller-Trumbore gives you this for free)
    float b0    = 1 - b1 - b2;
    auto  uv    = b0 * tri.t0 + b1 * tri.t1 + b2 * tri.t2;

    // check if the distance t is valid and the barycentric coordinates are within the triangle
    if (t < 0.0f || b0 < 0.f || b1 < 0.f || b2 < 0.f) 
    {
        return -1.0f;
    }

    intersectionPoint = r.origin + t * r.direction;

    // Fill in the gn with the geometric normal of the triangle
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

__host__ __device__ glm::vec3 triangleSample(
    Geom triangle,
    Ray& r,
    Material material,
    thrust::default_random_engine& rng,
    float& pdf)
{
    Triangle tri = triangle.triangle;
    glm::vec3 v0 = tri.v0;
    glm::vec3 v1 = tri.v1;
    glm::vec3 v2 = tri.v2;

    glm::vec3 E1 = v1 - v0;
    glm::vec3 E2 = v2 - v0;
    glm::vec3 normal = glm::normalize(glm::cross(E1, E2));
    
    // Sample Triangle
    thrust::uniform_real_distribution<float> u01(0, 1);
    float b0 = u01(rng);
    float b1 = u01(rng);
    if(b0 + b1 > 1.f)
    {
        b0 = 1.f - b0;
        b1 = 1.f - b1;
    }
    float b2 = 1 - b0 - b1;

    glm::vec3 samplePoint = b0 * v0 + b1 * v1 + b2 * v2;
    glm::vec3 direction = samplePoint - r.origin;
    float distanceSquared = glm::dot(direction, direction);
    float distance = sqrtf(distanceSquared);
    r.direction = direction / distance;

    float area = 0.5f * glm::length(glm::cross(E1, E2));
    float cosine = glm::abs(glm::dot(normal, r.direction));
    pdf = distanceSquared / (cosine * area);

    return material.color * material.emittance / pdf;
}

__host__ __device__ float trianglePDF(
    Geom triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float t = triangleIntersectionTest(triangle, r, intersectionPoint, normal, outside);
    if (t > 0) 
    {
        Triangle tri = triangle.triangle;
        glm::vec3 v0 = tri.v0;
        glm::vec3 v1 = tri.v1;
        glm::vec3 v2 = tri.v2;
        
        glm::vec3 E1 = v1 - v0;
        glm::vec3 E2 = v2 - v0;
        float area = 0.5f * glm::length(glm::cross(E1, E2));
        
        float distanceSquared = t * t;
        float cosine = glm::abs(glm::dot(glm::normalize(r.direction), normal));
        return distanceSquared / (cosine * area);
    }
    return 0.0f;
}
