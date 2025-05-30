#include <iostream>
#include <string>
#include <stb_image_write.h>
#include <algorithm>
#include <cctype>

#include "image.h"

template <typename T>
Image<T>::Image()
    : xSize(0), ySize(0), pixels(nullptr) 
{}

template <typename T>
Image<T>::Image(int x, int y)
    : xSize(x), ySize(y), pixels(new T[x * y]) 
{
    // Initialize to zero
    for (int i = 0; i < x * y; i++) {
        pixels[i] = T();
    }
}

template <typename T>
Image<T>::~Image()
{
    delete[] pixels;
}

template <typename T>
void Image<T>::setPixel(int x, int y, const T &pixel)
{
    if (x >= 0 && y >= 0 && x < xSize && y < ySize) {
        pixels[(y * xSize) + x] = pixel;
    }
}

template <typename T>
T Image<T>::getPixel(int x, int y) const
{
    if (x >= 0 && y >= 0 && x < xSize && y < ySize) {
        return pixels[(y * xSize) + x];
    }
    return T();
}

// Special implementation for glm::vec3 savePNG
template <>
void Image3::savePNG(const std::string &baseFilename)
{
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++)
    {
        for (int x = 0; x < xSize; x++)
        {
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(0.0f), glm::vec3(1.0f)) * 255.0f;
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

// Special implementation for glm::vec3 saveHDR
template <>
void Image3::saveHDR(const std::string &baseFilename)
{
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}

// Special implementation for float savePNG
template <>
void Image1::savePNG(const std::string &baseFilename)
{
    unsigned char *bytes = new unsigned char[xSize * ySize];
    for (int i = 0; i < xSize * ySize; i++)
    {
        float pix = glm::clamp(pixels[i], 0.0f, 1.0f) * 255.0f;
        bytes[i] = (unsigned char) pix;
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 1, bytes, xSize);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

// Special implementation for float saveHDR
template <>
void Image1::saveHDR(const std::string &baseFilename)
{
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 1, pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}

/////////////////////////////////////////////////////////////////////////////

inline std::string to_lowercase(const std::string &str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

inline std::string get_extension(const std::string &filename) {
    size_t pos = filename.find_last_of('.');
    if (pos != std::string::npos) {
        return to_lowercase(filename.substr(pos));
    }
    return "";
}

// Image loading function implementations
Image1 imread1(const std::string &filename) {
    std::string extension = get_extension(filename);
    
    if (extension == ".jpg" ||
          extension == ".png" ||
          extension == ".tga" ||
          extension == ".bmp" ||
          extension == ".psd" ||
          extension == ".gif" ||
          extension == ".hdr" ||
          extension == ".pic") {
        int w, h, n;
        float* data = stbi_loadf(filename.c_str(), &w, &h, &n, 1);
        
        if (data == nullptr) {
            std::cerr << "Error: Failed to load image: " << filename << std::endl;
            Error(std::string("Failure when loading image: ") + filename);
        }
        
        Image1 img(w, h);
        for (int i = 0; i < w * h; i++) {
            img.setPixel(i % w, i / w, data[i]);
        }
        stbi_image_free(data);
        return img;
    } else {
        Error(std::string("Unsupported image format: ") + filename);
    }
    return Image1(1, 1); // Return minimal valid image on error
}

Image3 imread3(const std::string &filename) {
    std::string extension = get_extension(filename);
    
    if (extension == ".jpg" ||
          extension == ".png" ||
          extension == ".tga" ||
          extension == ".bmp" ||
          extension == ".psd" ||
          extension == ".gif" ||
          extension == ".hdr" ||
          extension == ".pic") {
        int w, h, n;
        float* data = stbi_loadf(filename.c_str(), &w, &h, &n, 3);
        
        if (data == nullptr) {
            std::cerr << "Error: Failed to load image: " << filename << std::endl;
            Error(std::string("Failure when loading image: ") + filename);
        }
        
        Image3 img(w, h);
        for (int i = 0; i < w * h; i++) {
            glm::vec3 pixel(data[3*i], data[3*i+1], data[3*i+2]);
            img.setPixel(i % w, i / w, pixel);
        }
        stbi_image_free(data);
        return img;
    } else {
        Error(std::string("Unsupported image format: ") + filename);
    }
    return Image3(1, 1); // Return minimal valid image on error
}

// Explicit template instantiations
template class Image<float>;
template class Image<glm::vec2>;
template class Image<glm::vec3>;
