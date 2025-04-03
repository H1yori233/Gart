#include "sceneStructs.h"

__host__ __device__ ONB::ONB()
{
}

__host__ __device__ ONB::ONB(const glm::vec3 n_)
{
    n = glm::normalize(n_);
    glm::vec3 a = (abs(n.x) > 0.9) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    t = glm::normalize(glm::cross(a, n));
    s = glm::normalize(glm::cross(t, n));
}

__host__ __device__ ONB::ONB(const glm::vec3 &s_, const glm::vec3 &n_)
{
    s = s_;
    n = n_;
    t = glm::normalize(glm::cross(n, s));
}

__host__ __device__ ONB::ONB(const glm::vec3 &s, const glm::vec3 &t, const glm::vec3 &n)
{
    this->s = s;
    this->t = t;
    this->n = n;
}

__host__ __device__ glm::vec3 ONB::toLocal(const glm::vec3 &v) const
{
    // 正交矩阵的逆矩阵是它的转置
    // [s, t, n]^T * v
    return glm::vec3(glm::dot(v, s), glm::dot(v, t), glm::dot(v, n));
}

__host__ __device__ glm::vec3 ONB::toWorld(const glm::vec3 &v) const
{
    // 变换 [s, t, n] * v
    //      s * v 
    return v.x * s + v.y * t + v.z * n;
} 