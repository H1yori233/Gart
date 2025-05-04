#pragma once

#include "image.h"
#include "mipmap.h"
#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <algorithm>
#include <cuda/std/variant>
#include <cuda/std/functional>

// --------------------------- Host-side TexturePool ---------------------------

/// Can be replaced by a more advanced texture caching system,
/// where we only load images from files when necessary.
/// See OpenImageIO for example https://github.com/OpenImageIO/oiio
struct TexturePool {
    std::map<std::string, int> image1s_map;
    std::map<std::string, int> image3s_map;

    std::vector<Mipmap1> image1s;
    std::vector<Mipmap3> image3s;
};

inline bool texture_id_exists(const TexturePool &pool, const std::string &texture_name) {
    return pool.image1s_map.find(texture_name) != pool.image1s_map.end() ||
           pool.image3s_map.find(texture_name) != pool.image3s_map.end();
}

inline int insert_image1(TexturePool &pool, const std::string &texture_name, const std::string &filename) {
    if (pool.image1s_map.find(texture_name) != pool.image1s_map.end()) {
        // We don't check if img is the same as the one in the cache!
        return pool.image1s_map[texture_name];
    }
    int id = (int)pool.image1s.size();
    pool.image1s_map[texture_name] = id;
    pool.image1s.push_back(make_mipmap(imread1(filename)));
    return id;
}

inline int insert_image1(TexturePool &pool, const std::string &texture_name, const Image1 &img) {
    if (pool.image1s_map.find(texture_name) != pool.image1s_map.end()) {
        // We don't check if img is the same as the one in the cache!
        return pool.image1s_map[texture_name];
    }
    int id = (int)pool.image1s.size();
    pool.image1s_map[texture_name] = id;
    pool.image1s.push_back(make_mipmap(img));
    return id;
}

inline int insert_image3(TexturePool &pool, const std::string &texture_name, const std::string &filename) {
    if (pool.image3s_map.find(texture_name) != pool.image3s_map.end()) {
        // We don't check if img is the same as the one in the cache!
        return pool.image3s_map[texture_name];
    }
    int id = (int)pool.image3s.size();
    pool.image3s_map[texture_name] = id;
    pool.image3s.push_back(make_mipmap(imread3(filename)));
    return id;
}

inline int insert_image3(TexturePool &pool, const std::string &texture_name, const Image3 &img) {
    if (pool.image3s_map.find(texture_name) != pool.image3s_map.end()) {
        // We don't check if img is the same as the one in the cache!
        return pool.image3s_map[texture_name];
    }
    int id = (int)pool.image3s.size();
    pool.image3s_map[texture_name] = id;
    pool.image3s.push_back(make_mipmap(img));
    return id;
}

// --------------------------- Device-side DevTexturePool ---------------------------

struct DevTexturePool {
    Mipmap1* image1s;
    Mipmap3* image3s;
    int      num1, num3;
};

// Host utility to upload textures to GPU
inline void uploadTexturePool(DevTexturePool &devPool, const TexturePool &hostPool) {
    devPool.num1 = (int)hostPool.image1s.size();
    devPool.num3 = (int)hostPool.image3s.size();
    cudaMalloc(&devPool.image1s, sizeof(Mipmap1) * devPool.num1);
    cudaMalloc(&devPool.image3s, sizeof(Mipmap3) * devPool.num3);
    cudaMemcpy(devPool.image1s, hostPool.image1s.data(), sizeof(Mipmap1) * devPool.num1, cudaMemcpyHostToDevice);
    cudaMemcpy(devPool.image3s, hostPool.image3s.data(), sizeof(Mipmap3) * devPool.num3, cudaMemcpyHostToDevice);
}

// --------------------------- Accessors ---------------------------

inline const Mipmap1 &get_img1(const TexturePool &pool, int texture_id) {
    assert(texture_id >= 0 && texture_id < (int)pool.image1s.size());
    return pool.image1s[texture_id];
}

inline const Mipmap3 &get_img3(const TexturePool &pool, int texture_id) {
    assert(texture_id >= 0 && texture_id < (int)pool.image3s.size());
    return pool.image3s[texture_id];
}

__device__ inline const Mipmap1 &get_img1(const DevTexturePool &p, int texture_id) {
    return p.image1s[texture_id];
}
__device__ inline const Mipmap3 &get_img3(const DevTexturePool &p, int texture_id) {
    return p.image3s[texture_id];
}

// --------------------------- Texture Variants ---------------------------

template <typename T>
struct ConstantTexture {
    T value;
};

template <typename T>
struct ImageTexture {
    int texture_id;
    float uscale, vscale;
    float uoffset, voffset;
};

template <typename T>
struct CheckerboardTexture {
    T color0, color1;
    float uscale, vscale;
    float uoffset, voffset;
};

template <typename T>
__device__ inline const Mipmap<T> &get_img(const ImageTexture<T> &t, const DevTexturePool &pool) {
    return Mipmap<T>{};
}

template <>
__device__ inline const Mipmap<float> &get_img(const ImageTexture<float> &t, const DevTexturePool &pool) {
    return get_img1(pool, t.texture_id);
}

template <>
__device__ inline const Mipmap<glm::vec3> &get_img(const ImageTexture<glm::vec3> &t, const DevTexturePool &pool) {
    return get_img3(pool, t.texture_id);
}

template <typename T>
using Texture = cuda::std::variant<ConstantTexture<T>, ImageTexture<T>, CheckerboardTexture<T>>;

using Texture1 = Texture<float>;
using TextureSpectrum = Texture<glm::vec3>;

// --------------------------- Eval ---------------------------

template <typename T>
struct eval_texture_op {
    __host__ __device__ T operator()(const ConstantTexture<T> &t) const;
    __host__ __device__ T operator()(const ImageTexture<T> &t) const;
    __host__ __device__ T operator()(const CheckerboardTexture<T> &t) const;

    const glm::vec2 &uv;
    const float &footprint;
    // const TexturePool &pool;
    const DevTexturePool &dev_pool;
};

template <typename T>
__host__ __device__ T eval_texture_op<T>::operator()(const ConstantTexture<T> &t) const {
    return t.value;
}

template <typename T>
__host__ __device__ T eval_texture_op<T>::operator()(const ImageTexture<T> &t) const {
    const Mipmap<T> &img = get_img(t, dev_pool);
    glm::vec2 local_uv{modulo(uv[0] * t.uscale + t.uoffset, float(1)),
                     modulo(uv[1] * t.vscale + t.voffset, float(1))};
    float scaled_footprint = max(get_width(img), get_height(img)) * max(t.uscale, t.vscale) * footprint;
    float level = log2(max(scaled_footprint, float(1e-8f)));
    return lookup(img, local_uv[0], local_uv[1], level);
}

template <typename T>
__host__ __device__ T eval_texture_op<T>::operator()(const CheckerboardTexture<T> &t) const {
    glm::vec2 local_uv{modulo(uv[0] * t.uscale + t.uoffset, float(1)),
                     modulo(uv[1] * t.vscale + t.voffset, float(1))};
    int x = 2 * modulo((int)(local_uv.x * 2), 2) - 1,
        y = 2 * modulo((int)(local_uv.y * 2), 2) - 1;

    if (x * y == 1) {
        return t.color0;
    } else {
        return t.color1;
    }
}

/// Evaluate the texture at location uv.
/// Footprint should be approximatedly min(du/dx, du/dy, dv/dx, dv/dy) for texture filtering.
template <typename T>
__host__ __device__ T eval(const Texture<T> &texture, const glm::vec2 &uv, float footprint, const DevTexturePool &pool) {
    return cuda::std::visit(eval_texture_op<T>{uv, footprint, pool}, texture);
}

// --------------------------- Constructors ---------------------------

inline ConstantTexture<glm::vec3> make_constant_spectrum_texture(const glm::vec3 &spec) {
    return ConstantTexture<glm::vec3>{spec};
}

inline ConstantTexture<float> make_constant_float_texture(float f) {
    return ConstantTexture<float>{f};
}

inline ImageTexture<glm::vec3> make_image_spectrum_texture(
        const std::string &texture_name,
        const std::string &filename,
        TexturePool &pool,
        float uscale = 1,
        float vscale = 1,
        float uoffset = 0,
        float voffset = 0) {
    return ImageTexture<glm::vec3>{insert_image3(pool, texture_name, filename),
        uscale, vscale, uoffset, voffset};
}

inline ImageTexture<glm::vec3> make_image_spectrum_texture(
        const std::string &texture_name,
        const Image3 &img,
        TexturePool &pool,
        float uscale = 1,
        float vscale = 1,
        float uoffset = 0,
        float voffset = 0) {
    return ImageTexture<glm::vec3>{insert_image3(pool, texture_name, img),
        uscale, vscale, uoffset, voffset};
}

inline ImageTexture<float> make_image_float_texture(
        const std::string &texture_name,
        const std::string &filename,
        TexturePool &pool,
        float uscale = 1,
        float vscale = 1,
        float uoffset = 0,
        float voffset = 0) {
    return ImageTexture<float>{insert_image1(pool, texture_name, filename),
        uscale, vscale, uoffset, voffset};
}

inline ImageTexture<float> make_image_float_texture(
        const std::string &texture_name,
        const Image1 &img,
        TexturePool &pool,
        float uscale = 1,
        float vscale = 1,
        float uoffset = 0,
        float voffset = 0) {
    return ImageTexture<float>{insert_image1(pool, texture_name, img),
        uscale, vscale, uoffset, voffset};
}

inline CheckerboardTexture<glm::vec3> make_checkerboard_spectrum_texture(
        const glm::vec3 &color0, const glm::vec3 &color1,
        float uscale = 1, float vscale = 1,
        float uoffset = 0, float voffset = 0) {
    return CheckerboardTexture<glm::vec3>{
        color0, color1, uscale, vscale, uoffset, voffset};
}

inline CheckerboardTexture<float> make_checkerboard_float_texture(
        float color0, float color1,
        float uscale = 1, float vscale = 1,
        float uoffset = 0, float voffset = 0) {
    return CheckerboardTexture<float>{
        color0, color1, uscale, vscale, uoffset, voffset};
}
