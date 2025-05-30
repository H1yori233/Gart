#pragma once

#include "image.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <vector>
#include <iostream>
#include <algorithm>

constexpr int c_max_mipmap_levels = 8;

template <typename T>
struct Mipmap {
    // std::vector<Image<T>> images;
    Image<T> images[c_max_mipmap_levels];
    int numLevels; 
};

template <typename T>
__host__ __device__ inline int get_width(const Mipmap<T> &mipmap) {
    assert(mipmap.numLevels > 0);
    return mipmap.images[0].getWidth() ;
}

template <typename T>
__host__ __device__ inline int get_height(const Mipmap<T> &mipmap) {
    assert(mipmap.numLevels > 0);
    return mipmap.images[0].getHeight();
}

template <typename T>
inline Mipmap<T> make_mipmap(const Image<T> &img) {
    Mipmap<T> mipmap;
    int size = max(img.getWidth() , img.getHeight());
    int num_levels = std::min((int)ceil(log2(float(size)) + 1), c_max_mipmap_levels);
    mipmap.images[0] = img;
    mipmap.numLevels = num_levels;
    for (int i = 1; i < num_levels; i++) {
        const Image<T> &prev_img = mipmap.images[i-1];
        int next_w = max(prev_img.getWidth()  / 2, 1);
        int next_h = max(prev_img.getHeight() / 2, 1);
        Image<T> next_img(next_w, next_h);
        for (int y = 0; y < next_img.getHeight(); y++) {
            for (int x = 0; x < next_img.getWidth() ; x++) {
                // 2x2 box filter
                next_img.setPixel(x, y,
                    (prev_img.getPixel(2 * x    , 2 * y    ) +
                     prev_img.getPixel(2 * x + 1, 2 * y    ) +
                     prev_img.getPixel(2 * x    , 2 * y + 1) +
                     prev_img.getPixel(2 * x + 1, 2 * y + 1)) / float(4));
            }
        }
        mipmap.images[i] = next_img;
    }
    return mipmap;
}

/// Bilinear lookup of a mipmap at location (uv) with an integer level
template <typename T>
__host__ __device__ inline T lookup(const Mipmap<T> &mipmap, float u, float v, int level) {
    assert(level >= 0 && level < mipmap.numLevels);
    // Bilinear interpolation
    // (-0.5 to match Mitsuba's coordinates)
    u = u * mipmap.images[level].getWidth()  - 0.5f;
    v = v * mipmap.images[level].getHeight() - 0.5f;
    int ufi = modulo(int(u), mipmap.images[level].getWidth() );
    int vfi = modulo(int(v), mipmap.images[level].getHeight());
    int uci = modulo(ufi + 1, mipmap.images[level].getWidth() );
    int vci = modulo(vfi + 1, mipmap.images[level].getHeight());
    float u_off = u - ufi;
    float v_off = v - vfi;
    T val_ff = mipmap.images[level].getPixel(ufi, vfi);
    T val_fc = mipmap.images[level].getPixel(ufi, vci);
    T val_cf = mipmap.images[level].getPixel(uci, vfi);
    T val_cc = mipmap.images[level].getPixel(uci, vci);
    return val_ff * (1 - u_off) * (1 - v_off) +
           val_fc * (1 - u_off) *      v_off +
           val_cf *      u_off  * (1 - v_off) +
           val_cc *      u_off  *      v_off;
}

/// Trilinear look of of a mipmap at (u, v, level)
template <typename T>
__host__ __device__ inline T lookup(const Mipmap<T> &mipmap, float u, float v, float level) {
    if (level <= 0) {
        return lookup(mipmap, u, v, 0);
    } else if (level < float(mipmap.numLevels - 1)) {
        int flevel = glm::clamp((int)floor(level), 0, mipmap.numLevels - 1);
        int clevel = glm::clamp(flevel + 1, 0, mipmap.numLevels - 1);
        float level_off = level - flevel;
        return lookup(mipmap, u, v, flevel) * (1 - level_off) +
               lookup(mipmap, u, v, clevel) *      level_off;
    } else {
        return lookup(mipmap, u, v, int(mipmap.numLevels - 1));
    }
}

using Mipmap1 = Mipmap<float>;
using Mipmap3 = Mipmap<glm::vec3>;
