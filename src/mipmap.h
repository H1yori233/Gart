#pragma once

#include "image.h"
#include "glm/glm.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

constexpr int c_max_mipmap_levels = 8;

template <typename T>
struct Mipmap {
    std::vector<Image<T>> images;
};

template <typename T>
inline int get_width(const Mipmap<T> &mipmap) {
    assert(mipmap.images.size() > 0);
    return mipmap.images[0].xSize;
}

template <typename T>
inline int get_height(const Mipmap<T> &mipmap) {
    assert(mipmap.images.size() > 0);
    return mipmap.images[0].ySize;
}

template <typename T>
inline Mipmap<T> make_mipmap(const Image<T> &img) {
    Mipmap<T> mipmap;
    int size = max(img.xSize, img.ySize);
    int num_levels = std::min((int)ceil(log2(float(size)) + 1), c_max_mipmap_levels);
    mipmap.images.push_back(img);
    for (int i = 1; i < num_levels; i++) {
        const Image<T> &prev_img = mipmap.images.back();
        int next_w = max(prev_img.xSize / 2, 1);
        int next_h = max(prev_img.ySize / 2, 1);
        Image<T> next_img(next_w, next_h);
        for (int y = 0; y < next_img.ySize; y++) {
            for (int x = 0; x < next_img.xSize; x++) {
                // 2x2 box filter
                next_img.setPixel(x, y,
                    (prev_img.getPixel(2 * x    , 2 * y    ) +
                     prev_img.getPixel(2 * x + 1, 2 * y    ) +
                     prev_img.getPixel(2 * x    , 2 * y + 1) +
                     prev_img.getPixel(2 * x + 1, 2 * y + 1)) / float(4));
            }
        }
        mipmap.images.push_back(next_img);
    }
    return mipmap;
}

/// Bilinear lookup of a mipmap at location (uv) with an integer level
template <typename T>
inline T lookup(const Mipmap<T> &mipmap, float u, float v, int level) {
    assert(level >= 0 && level < (int)mipmap.images.size());
    // Bilinear interpolation
    // (-0.5 to match Mitsuba's coordinates)
    u = u * mipmap.images[level].xSize - 0.5f;
    v = v * mipmap.images[level].ySize - 0.5f;
    int ufi = modulo(int(u), mipmap.images[level].xSize);
    int vfi = modulo(int(v), mipmap.images[level].ySize);
    int uci = modulo(ufi + 1, mipmap.images[level].xSize);
    int vci = modulo(vfi + 1, mipmap.images[level].ySize);
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
inline T lookup(const Mipmap<T> &mipmap, float u, float v, float level) {
    if (level <= 0) {
        return lookup(mipmap, u, v, 0);
    } else if (level < float(mipmap.images.size() - 1)) {
        // int flevel = std::clamp((int)floor(level), 0, (int)mipmap.images.size() - 1);
        // int clevel = std::clamp(flevel + 1, 0, (int)mipmap.images.size() - 1);
        int flevel = std::max(std::min((int)floor(level), (int)mipmap.images.size() - 1), 0);
        int clevel = std::max(std::min(flevel + 1, (int)mipmap.images.size() - 1), 0);
        float level_off = level - flevel;
        return lookup(mipmap, u, v, flevel) * (1 - level_off) +
               lookup(mipmap, u, v, clevel) *      level_off;
    } else {
        return lookup(mipmap, u, v, int(mipmap.images.size() - 1));
    }
}

using Mipmap1 = Mipmap<float>;
using Mipmap3 = Mipmap<glm::vec3>;
