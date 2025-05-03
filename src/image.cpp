#include <iostream>
#include <string>
#include <stb_image_write.h>
#include <algorithm>
#include <cctype>

#include "image.h"

template <typename T>
Image<T>::Image(int x, int y)
    : xSize(x), ySize(y), pixels(new T[x * y]) 
{}

template <typename T>
Image<T>::~Image()
{
    delete[] pixels;
}

template <typename T>
void Image<T>::setPixel(int x, int y, const T &pixel)
{
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

template <typename T>
T Image<T>::getPixel(int x, int y) const
{
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    return pixels[(y * xSize) + x];
}

// 特化版本：glm::vec3 的 savePNG
template <>
void Image3::savePNG(const std::string &baseFilename)
{
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++)
    {
        for (int x = 0; x < xSize; x++)
        {
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

// 特化版本：glm::vec3 的 saveHDR
template <>
void Image3::saveHDR(const std::string &baseFilename)
{
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}

// // 图像读取函数实现
// Image<float> imread1(const std::filesystem::path &filename) {
//     Image<float> img;
//     std::string extension = to_lowercase(filename.extension().string());
//     // JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
//     if (extension == ".jpg" ||
//           extension == ".png" ||
//           extension == ".tga" ||
//           extension == ".bmp" ||
//           extension == ".psd" ||
//           extension == ".gif" ||
//           extension == ".hdr" ||
//           extension == ".pic") {
//         int w, h, n;
// #ifdef _WINDOWS
//         float* data = stbi_loadf(filename.string().c_str(), &w, &h, &n, 1);
// #else
//         float *data = stbi_loadf(filename.c_str(), &w, &h, &n, 1);
// #endif
//         img = Image<float>(w, h);
//         if (data == nullptr) {
//             Error(std::string("Failure when loading image: ") + filename.string());
//         }
//         for (int i = 0; i < w * h; i++) {
//             img.setPixel(i % w, i / w, data[i]);
//         }
//         stbi_image_free(data);
//     } else if (extension == ".exr") {
// //         float* data = nullptr;
// //         int width;
// //         int height;
// //         const char* err = nullptr;
// // #ifdef _WINDOWS
// //         int ret = LoadEXR(&data, &width, &height, filename.string().c_str(), &err);
// // #else
// //         int ret = LoadEXR(&data, &width, &height, filename.c_str(), &err);
// // #endif
// //         if (ret != TINYEXR_SUCCESS) {
// //             std::cerr << "OpenEXR error: " << err << std::endl;
// //             FreeEXRErrorMessage(err);
// //             Error(std::string("Failure when loading image: ") + filename.string());
// //         }
// //         img = Image<float>(width, height);
// //         for (int i = 0; i < width * height; i++) {
// //             img.setPixel(i % width, i / width, (data[4 * i] + data[4 * i + 1] + data[4 * i + 2]) / 3);
// //         }
// //         free(data);
//     } else {
//         Error(std::string("Unsupported image format: ") + filename.string());
//     }
//     return img;
// }

// Image3 imread3(const std::filesystem::path &filename) {
//     Image3 img;
//     std::string extension = to_lowercase(filename.extension().string());
//     // JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
//     if (extension == ".jpg" ||
//           extension == ".png" ||
//           extension == ".tga" ||
//           extension == ".bmp" ||
//           extension == ".psd" ||
//           extension == ".gif" ||
//           extension == ".hdr" ||
//           extension == ".pic") {
//         int w, h, n;
// #ifdef _WINDOWS
//         float* data = stbi_loadf(filename.string().c_str(), &w, &h, &n, 3);
// #else
//         float* data = stbi_loadf(filename.c_str(), &w, &h, &n, 3);
// #endif
//         img = Image3(w, h);
//         if (data == nullptr) {
//             Error(std::string("Failure when loading image: ") + filename.string());
//         }
//         int j = 0;
//         for (int i = 0; i < w * h; i++) {
//             glm::vec3 pixel(data[j], data[j+1], data[j+2]);
//             img.setPixel(i % w, i / w, pixel);
//             j += 3;
//         }
//         stbi_image_free(data);
//     } else if (extension == ".exr") {
// //         float* data = nullptr;
// //         int width;
// //         int height;
// //         const char* err = nullptr;
// // #ifdef _WINDOWS
// //         int ret = LoadEXR(&data, &width, &height, filename.string().c_str(), &err);
// // #else
// //         int ret = LoadEXR(&data, &width, &height, filename.c_str(), &err);
// // #endif
// //         if (ret != TINYEXR_SUCCESS) {
// //             std::cerr << "OpenEXR error: " << err << std::endl;
// //             FreeEXRErrorMessage(err);
// //             Error(std::string("Failure when loading image: ") + filename.string());
// //         }
// //         img = Image3(width, height);
// //         for (int i = 0; i < width * height; i++) {
// //             glm::vec3 pixel(data[4 * i], data[4 * i + 1], data[4 * i + 2]);
// //             img.setPixel(i % width, i / width, pixel);
// //         }
// //         free(data);
//     } else {
//         Error(std::string("Unsupported image format: ") + filename.string());
//     }
//     return img;
// }

template class Image<float>;
template class Image<glm::vec2>;
template class Image<glm::vec3>;
