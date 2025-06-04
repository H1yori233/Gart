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
            float ta = fminf(t1, t2);
            float tb = fmaxf(t1, t2);
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
        t = fminf(t1, t2);
        outside = true;
    }
    else
    {
        t = fmaxf(t1, t2);
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

__host__ __device__ float quadIntersectionTest(
    Geom quad,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    // Transform ray to object space
    glm::vec3 ro = multiplyMV(quad.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(quad.inverseTransform, glm::vec4(r.direction, 0.0f)));

    // Check if ray is parallel to the quad (z=0 plane)
    if (glm::abs(rd.z) < EPSILON)
        return -1;

    // Calculate intersection with z=0 plane
    float t = -ro.z / rd.z;
    
    if (t <= EPSILON)
        return -1;

    // Calculate intersection point in object space
    glm::vec3 objspaceIntersection = ro + rd * t;

    // Check if intersection point is within quad bounds [-0.5, 0.5] in x and y
    if (glm::abs(objspaceIntersection.x) > 0.5f || glm::abs(objspaceIntersection.y) > 0.5f)
        return -1;

    // Project hit point onto plane to reduce floating-point error
    objspaceIntersection.z = 0.0f;

    // Transform intersection point back to world space
    intersectionPoint = multiplyMV(quad.transform, glm::vec4(objspaceIntersection, 1.0f));

    // Calculate normal (always pointing in +z direction in object space)
    glm::vec3 objspaceNormal = glm::vec3(0, 0, 1);
    normal = glm::normalize(multiplyMV(quad.invTranspose, glm::vec4(objspaceNormal, 0.0f)));

    // Check if ray is coming from outside (negative dot product with normal)
    outside = glm::dot(r.direction, normal) < 0;
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    Geom triangle_geom,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    // Transform ray to object space
    glm::vec3 ro = multiplyMV(triangle_geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(triangle_geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    const Triangle& tri = triangle_geom.triangle;
    
    // Möller–Trumbore intersection algorithm
    glm::vec3 edge1, edge2, h, s, q;
    float a, f, u, v;
    
    edge1 = tri.v1 - tri.v0;
    edge2 = tri.v2 - tri.v0;
    h = glm::cross(rd, edge2);
    a = glm::dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON)
        return -1; // Ray is parallel to triangle
    
    f = 1.0f / a;
    s = ro - tri.v0;
    u = f * glm::dot(s, h);
    
    if (u < 0.0f || u > 1.0f)
        return -1;
    
    q = glm::cross(s, edge1);
    v = f * glm::dot(rd, q);
    
    if (v < 0.0f || u + v > 1.0f)
        return -1;
    
    // At this stage we can compute t to find out where the intersection point is on the line
    float t = f * glm::dot(edge2, q);
    
    if (t > EPSILON) // Ray intersection
    {
        glm::vec3 objspaceIntersection = ro + rd * t;
        
        // Transform intersection point back to world space
        intersectionPoint = multiplyMV(triangle_geom.transform, glm::vec4(objspaceIntersection, 1.0f));
        
        // Calculate normal
        if (tri.hasVertexNormals)
        {
            // Interpolate vertex normals using barycentric coordinates
            float w = 1.0f - u - v;
            glm::vec3 interpolatedNormal = w * tri.n0 + u * tri.n1 + v * tri.n2;
            normal = glm::normalize(multiplyMV(triangle_geom.invTranspose, glm::vec4(interpolatedNormal, 0.0f)));
        }
        else
        {
            // Use face normal
            glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));
            normal = glm::normalize(multiplyMV(triangle_geom.invTranspose, glm::vec4(faceNormal, 0.0f)));
        }
        
        // Check if ray is coming from outside
        outside = glm::dot(r.direction, normal) < 0;
        if (!outside)
        {
            normal = -normal;
        }
        
        return glm::length(r.origin - intersectionPoint);
    }
    
    return -1; // No intersection
}
